import json
import logging
import os
import pickle
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_model import BaseModel, TorchModel
from data.patch_making import make_patch_dataset
from metrics.dota_eval import dota_eval
from metrics.dota_results_translator import DOTAResultsTranslator
from model_parts.losses.pixel_ce_loss import PixelCELoss
from model_parts.unet.unet import pad_before_infer
from models.position_net.pos_net_model import PosNetModel, PATCH_SIZE
from models.shape_net.data_loaders import LossMaskParams, ShapePatchProcessor
from models.shape_net.display import display_shape_inference, _pred_to_image, _pred_to_image2
from models.shape_net.mappings import ValueMapping, class_id_to_value, output_vector_to_value
from models.shape_net.shape_net import ShapeNet
from base.shapes.rectangle import sra_to_wla, rect_to_poly
from utils.data import get_dataset_base_path, fetch_data_paths, get_model_config_by_name, get_inference_path
from utils.files import make_gif, make_if_not_exist
from utils.logger import Logger
from utils.misc import timestamp
from utils.nms import nms_distance
from utils.training import update_metrics, print_metrics, startup_config, PatchBasedTrainer, Config


class ShapeNetModel(BaseModel, PatchBasedTrainer, TorchModel):
    def __init__(self, config: Config, train: bool, load=False, reuse_data=False, overwrite=False, dataset: str = None):
        self.config, self.logger, self.save_path = startup_config(config, 'shapenet', load_model=load,
                                                                  overwrite=overwrite)
        if not load:
            self.logger.clear()

        self.dataset = self.config['data_loader']["dataset"] if dataset is None else dataset
        self.temp_dataset = 'temp_' + self.config['model_name'] + '_' + timestamp()

        # get train config
        self.n_epochs = self.config['trainer']['n_epochs']
        self.n_classes = self.config['trainer']['n_classes']
        self.batch_size = self.config['trainer']['batch_size']
        self.figure_interval = self.config['trainer']['figure_interval']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self.model = ShapeNet(
            in_channels=3,
            hidden_dims=self.config['model']['hidden_dims'],
            out_features=3,
            out_feat_size=self.n_classes,
            device=self.device
        ).to(self.device)
        self.logger.log_model(checkpoint_interval=4, model=self.model)
        self.last_epoch = 0

        if load:
            self._load()

        self.optimizer = Adam(params=self.model.parameters(), lr=self.config['loss']['learning_rate'])
        self.criterion = PixelCELoss(
            focal_loss=self.config['loss'].get('focal_loss'),
            focal_loss_args=self.config['loss'].get('focal_loss_args')
        )
        self.loss_mask_params = LossMaskParams(
            mode=self.config['loss']['mask_mode'],
            mask_sigma=self.config['loss'].get('mask_sigma'),
            mask_cutoff_dist=self.config['loss'].get('mask_cutoff_dist'))

        self.rng = np.random.default_rng(42)

        self.s_mapping = ValueMapping(self.n_classes, self.config['mappings']['size_mapping_min'],
                                      self.config['mappings']['size_mapping_max'])
        self.r_mapping = ValueMapping(self.n_classes, 0, 1)
        self.angle_mapping = ValueMapping(self.n_classes, 0, np.pi, is_cyclic=True)

        self.mappings = [self.s_mapping, self.r_mapping, self.angle_mapping]

        self.label_processor_train = ShapePatchProcessor(
            mappings=self.mappings,
            class_perturbation_dict={0: 0.8, 1: 0.1, -1: 0.1},
            rng=self.rng,
            mask_params=self.loss_mask_params,
        )

        self.label_processor_val = ShapePatchProcessor(
            mappings=self.mappings,
            rng=self.rng,
            class_perturbation_dict=None,
            mask_params=self.loss_mask_params,
        )

        if train:
            self.__init_data__(reuse_data=reuse_data)

    def train_epoch(self, loader):
        self.model.train()
        all_metrics = None
        for x, y in loader:
            self.optimizer.zero_grad()
            results = self.model.forward(x.to(self.device))

            loss_dict = self.criterion(results, [t.to(self.device) for t in y['value_class_map']],
                                       loss_mask=y['loss_mask'].to(self.device))

            loss_dict['loss'].backward()
            self.optimizer.step()
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    def val_epoch(self, loader):
        self.model.eval()
        all_metrics = None
        for x, y in loader:
            results = self.model.forward(x.to(self.device))
            loss_dict = self.criterion(results, [t.to(self.device) for t in y['value_class_map']],
                                       loss_mask=y['loss_mask'].to(self.device))
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    @torch.no_grad()
    def infer_on_image(self, image, centers=None, params=None, apply_softmax=True, raw_output=False):
        self.model.eval()
        if centers is None or params is None:
            centers = np.array([])
            params = np.array([])
        img, label = self.label_processor_val.process(image, centers, params, idx=0)

        def infer(input_image):
            img, pad = pad_before_infer(input_image, depth=len(self.config["model"]["hidden_dims"]) - 1)
            img = torch.unsqueeze(img, 0)
            output = self.model.forward(img.to(self.device))
            if apply_softmax:
                output = [functional.softmax(t, dim=1) for t in output]  # output is List[BxCxHxW]

            output_numpy = [t.detach().cpu().numpy() for t in output]
            for i in range(len(output_numpy)):
                if pad[0] > 0:
                    output_numpy[i] = output_numpy[i][:, :, :-pad[0]]
                if pad[1] > 0:
                    output_numpy[i] = output_numpy[i][:, :, :, :-pad[1]]

            return output_numpy

        try:
            res = infer(img)
        except Exception as e:
            print(f"inference failed because {e}")
            print("falling back to infer in patches")
            shape = img.shape[1:]
            print(f"with image of shape {shape}")
            res = [np.empty((1, self.n_classes) + shape) for _ in range(3)]
            nx = shape[0] // PATCH_SIZE
            ny = shape[1] // PATCH_SIZE
            print(f"splitting into {nx} x {ny} patches")
            for i in range(nx + 1):
                for j in range(ny + 1):
                    s = np.s_[
                        min(i * PATCH_SIZE, shape[0]):min((i + 1) * PATCH_SIZE, shape[0]),
                        min(j * PATCH_SIZE, shape[1]):min((j + 1) * PATCH_SIZE, shape[1])
                        ]
                    img_crop = img[:, s[0], s[1]]
                    print(f"patch {img_crop.shape}")
                    if img_crop.shape[1] == 0 or img_crop.shape[2] == 0:
                        print("woops empty slice")
                        continue
                    res_crop = infer(img_crop)
                    for k in range(len(res_crop)):
                        res[k][:, :, s[0], s[1]] = res_crop[k]

        if raw_output:
            return res
        else:
            return [np.moveaxis(t[0], 0, -1) for t in res]

    def make_figures(self, epoch, logger: Logger = None, subset=None):
        self.model.eval()
        output = self.model.forward(self.images_figs.to(self.device))
        loss_dict = self.criterion(output, [t.to(self.device) for t in self.label_figs['value_class_map']],
                                   loss_mask=self.label_figs['loss_mask'].to(self.device),
                                   return_maps=True)

        thickness = self.images_figs.shape[2] // 64

        if logger is not None:
            metrics = np.array([logger.log['train_loss'], logger.log['val_loss']])
        else:
            metrics = None

        display_shape_inference(
            epoch=epoch,
            images=self.images_figs,
            output=output,
            labels=self.label_figs,
            loss=loss_dict,
            mappings=self.mappings,
            results_path=self.save_path,
            subset=subset,
            thickness=thickness,
            metrics=metrics,
            colors=['tab:orange', 'tab:blue']
        )

    def make_data_samples(self, loader: DataLoader, type: str, n_batch_samples=1):
        data_samples_dir = os.path.join(self.save_path, f'data_samples_{type}')
        if not os.path.exists(data_samples_dir):
            os.mkdir(data_samples_dir)
        loader_iter = loader.__iter__()
        for i in range(n_batch_samples):
            x, y = loader_iter.next()

            image_arrays = x.detach().cpu().permute((0, 2, 3, 1)).numpy()
            for j in range(len(image_arrays)):
                file_name = f'training_sample_b{i:02}_{j:04}'
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_raw.png'), image_arrays[j])

                centers = np.array(np.where(y['center_binary_map'][j].numpy())).T
                feature_value_classes = [y['value_class_map'][k][j].numpy() for k in range(3)]
                feature_values = class_id_to_value(feature_value_classes, mappings=self.mappings)

                label_image = _pred_to_image(pred=feature_values, centers=centers, image=image_arrays[j],
                                             color=np.array([0, 1, 0]), mix_factor=0.6)
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_label.png'), label_image)

    def data_preview(self):
        self.make_data_samples(self.train_loader, 'train')
        self.make_data_samples(self.val_loader, 'val')

    def train(self):
        self.data_preview()
        for epoch in range(self.last_epoch, self.n_epochs):
            train_metrics = self.train_epoch(self.train_loader)

            val_metrics = self.val_epoch(self.val_loader)

            print_metrics(epoch, train_metrics, val_metrics)

            self.logger.update_train_val(epoch, train_metrics, val_metrics)

            if epoch % self.figure_interval == 0 or epoch == self.n_epochs - 1:
                self.make_figures(epoch, logger=self.logger)

            if epoch % self.dataset_update_interval == 0 and epoch != 0:
                # error_maps = self.compute_errors(os.path.join(self.data_path, 'train'))
                make_patch_dataset(source_dataset=self.dataset,
                                   new_dataset=self.temp_dataset,
                                   config=self.config,
                                   make_val=False,
                                   rng=self.rng)
                self.data_train.update_files()

        self.make_figures(self.n_epochs - 1, subset=[0, 2, 3, 5])
        make_gif(self.save_path, 'res_*.png', 'res.gif')
        self.save()
        print("Saved model")
        self.clean()
        print("cleared temp files")

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pt'))

    def clean(self):
        shutil.rmtree(os.path.join(get_dataset_base_path(), self.temp_dataset))

    def infer(self, subset: str, min_confidence: float = 0.1, display_min_confidence: float = 0.5,
              overwrite: bool = True):

        if 'inference' in self.config:
            pos_model_name = self.config['inference']['pos_model']
            with open(get_model_config_by_name(pos_model_name), 'r') as f:
                pos_config = json.load(f)
            pos_model = PosNetModel(pos_config, overwrite=False, load=True, train=False)

            def pos_model_inference(img, centers, params, confidence):
                output_mask, output_vec, label = pos_model.infer_on_image(img, centers, params)
                detection_map = pos_model.vec2detection_map(output_vec, output_mask)
                detections_centers = np.array(np.where(detection_map >= confidence)).T
                detections_scores = detection_map[detections_centers[:, 0], detections_centers[:, 1]]
                nms_centers, nms_scores = nms_distance(detections_centers, detections_scores, threshold=6)
                return nms_centers, nms_scores
        else:
            logging.warning(f'no position inference model specified in config, falling back to using ground truth')
            pos_model_name = None

            def pos_model_inference(img, centers, params, confidence):
                return centers, np.ones(len(centers))

        id_re = re.compile(r'([0-9]+).*.png')

        results_dir = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=self.dataset, subset=subset)
        # coco_trlt = COCOTranslator(self.dataset, subset)
        dota_trlt = DOTAResultsTranslator(self.dataset, subset, results_dir, det_type='obb', all_classes=['vehicle'])
        make_if_not_exist(results_dir, recursive=True)
        paths_dict = fetch_data_paths(self.dataset, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']
        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {self.dataset}/{subset}'), annot_paths,
                              meta_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            if os.path.exists(os.path.join(results_dir, f'{patch_id:04}_results.pkl')) and not overwrite:
                logging.info(f"{patch_id:04}_results.pkl exists, skipping")
                continue
            img = plt.imread(pf)[..., :3]
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)
            centers, params = labels_dict['centers'], labels_dict['parameters']
            difficulty = labels_dict['difficult']
            with open(mf, 'r') as f:
                meta_dict = json.load(f)

            output = self.infer_on_image(img, raw_output=True)  # shape (marks,B,32,H,W)
            assert all([output[k][0].shape[1:] == img.shape[:2]] for k in range(3))
            pred_centers, pred_scores = pos_model_inference(img, centers, params, confidence=min_confidence)

            values_map = output_vector_to_value(output, self.mappings)

            pred_params = [
                sra_to_wla(values_map[0][0][c[0], c[1]], values_map[1][0][c[0], c[1]], values_map[2][0][c[0], c[1]])
                for c in
                pred_centers]

            image_w_pred = _pred_to_image2(
                centers=[c for c, s in zip(pred_centers, pred_scores) if s >= display_min_confidence],
                params=[p for p, s in zip(pred_params, pred_scores) if s >= display_min_confidence],
                scores=[s for s in pred_scores if s >= display_min_confidence],
                image=img,
                color='plasma')

            image_w_gt = _pred_to_image2(
                centers=centers,
                params=params,
                scores=None,
                image=img,
                color=(0, 1.0, 0))

            detection_as_poly = np.array(
                [rect_to_poly(c, p[0], p[1], p[2]) for c, p in zip(pred_centers, pred_params)])

            # a_map = plt.get_cmap('plasma')(values_map[0][0] / self.mappings[0].v_max)[:, :, :3]
            # b_map = plt.get_cmap('plasma')(values_map[1][0] / self.mappings[1].v_max)[:, :, :3]
            # angle_map = plt.get_cmap('twilight')(values_map[2][0] / self.mappings[2].v_max)[:, :, :3]

            results_dict = {
                'detection': detection_as_poly,
                'detection_type': 'poly',
                'detection_center': pred_centers,
                'detection_score': pred_scores,
                'detection_params': pred_params,
                'pos_model': pos_model_name,
                'mappings': self.mappings,
                'output': output
            }

            gt_as_poly = np.array(
                [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])
            # coco_trlt.add_image(image_id=patch_id, file_name=os.path.split(pf)[1], metadata=meta_dict)
            # coco_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly)
            # coco_trlt.add_detections(
            #     image_id=patch_id, image_shape=img.shape[:2], scores=pred_scores, polygons=detection_as_poly,
            #     flip_coor=False)
            dota_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly, difficulty=difficulty,categories=['vehicle' for _ in gt_as_poly])
            dota_trlt.add_detections(image_id=patch_id, scores=pred_scores, polygons=detection_as_poly,
                                     flip_coor=True, class_names=['vehicle' for _ in pred_scores])

            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_detection.png'), image_w_pred)
            # plt.imsave(os.path.join(results_dir, f'{patch_id:04}_map_a.png'), a_map)
            # plt.imsave(os.path.join(results_dir, f'{patch_id:04}_map_b.png'), b_map)
            # plt.imsave(os.path.join(results_dir, f'{patch_id:04}_map_angle.png'), angle_map)
            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_gt.png'), image_w_gt)
            with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)

        # coco_trlt.save(results_dir)
        dota_trlt.save()
        print('saved coco translation')

    def eval(self):
        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='obb'
        )
