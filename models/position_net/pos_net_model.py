import logging
import os
import pickle
import re
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import rescale
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_model import BaseModel, TorchModel
from data.patch_making import make_patch_dataset
from metrics.dota_eval import dota_eval
from metrics.dota_results_translator import DOTAResultsTranslator
from model_parts.losses.pos_loss import PointingVectorLoss
from model_parts.unet.unet import pad_before_infer
from utils.display.boxes import bboxes_over_image_cv2
from models.position_net.data_loaders import PosPatchProcessor
from models.position_net.display import display_pos_inference
from models.position_net.pos_net import PosNet
from models.position_net.torch_div import Divergence
from utils.data import get_dataset_base_path, fetch_data_paths, get_inference_path
from utils.files import make_gif, make_if_not_exist
from utils.logger import Logger
from utils.math_utils import divergence_map_from_vector_field
from utils.misc import timestamp
from utils.nms import nms_distance
from utils.training import update_metrics, print_metrics, startup_config, PatchBasedTrainer, Config

PATCH_SIZE = 512


class PosNetModel(BaseModel, PatchBasedTrainer, TorchModel):

    def __init__(self, config: Config, train: bool, load=False, reuse_data=False, overwrite=False, dataset: str = None):

        self.config, self.logger, self.save_path = startup_config(config, 'posnet', load_model=load,
                                                                  overwrite=overwrite)
        if not load:
            self.logger.clear()

        self.dataset = self.config['data_loader']["dataset"] if dataset is None else dataset
        self.error_update_interval = self.config['data_loader'].get("error_update_interval")
        self.error_densities = None
        self.temp_dataset = 'temp_' + self.config['model_name'] + '_' + timestamp()

        self.n_epochs = self.config['trainer']['n_epochs']
        self.last_epoch = 0
        self.batch_size = self.config['trainer']['batch_size']
        self.figure_interval = self.config['trainer']['figure_interval']

        self.max_distance = self.config['loss']['max_distance']
        self.target_mode = self.config['loss']['target_mode']
        self.n_classes = self.config['loss'].get('n_classes')

        self.learn_mask = self.config['loss']['learn_mask']
        self.out_channels = 3 if self.learn_mask else 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self.model = PosNet(
            in_channels=3,
            hidden_dims=self.config['model']['hidden_dims'],
            device=self.device,
            out_channels=self.out_channels
        ).to(self.device)
        self.div_clf = None
        if "div_clf_model" in self.config:
            self.div_clf = nn.Sequential(
                Divergence(div_channels=[0, 1], mask_channel=2),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
            ).to(self.device)

        self.logger.log_model(checkpoint_interval=4, model=self.model)

        if load:
            self._load()
            div_clf_path = os.path.join(self.save_path, 'model_div_clf.pt')
            if os.path.exists(div_clf_path):
                self.div_clf.load_state_dict(torch.load(div_clf_path))

        params = list(self.model.parameters())
        if self.div_clf is not None:
            params = params + list(self.div_clf.parameters())

        self.optimizer = Adam(params=params, lr=1e-3)
        self.criterion = PointingVectorLoss(
            learn_mask=self.learn_mask,
            compute_mask=self.config['loss']['compute_relevant'],
            balanced_mask_loss=self.config['loss']['balanced_mask_loss'],
            focal_loss=self.config['loss'].get('focal_loss'),
            vec_loss_on_prod=self.config['loss'].get('vec_loss_on_prod'),
        )

        self.rng = np.random.default_rng(42)

        label_processor = PosPatchProcessor(
            max_distance=self.max_distance,
            mode=self.target_mode,
            n_classes=self.config['loss'].get('n_classes'),
            sigma_dil = self.config['loss'].get('bin_map_dil')
        )
        self.label_processor_train = label_processor
        self.label_processor_val = label_processor

        if train:
            self.__init_data__(reuse_data=reuse_data)

    def train_epoch(self, loader):
        self.model.train()
        all_metrics = None
        for x, y in loader:
            self.optimizer.zero_grad()
            results = self.model.forward(x.to(self.device))
            if self.div_clf is not None:
                vec_and_mask = torch.concat([results[:, :2], torch.sigmoid(results[:, [2]])], dim=1)
                div_score = self.div_clf.forward(vec_and_mask)
            else:
                div_score = None

            loss_dict = self.criterion(results, y['pointing_map'].to(self.device),
                                       target_mask=None if not self.learn_mask else y['mask'].to(self.device),
                                       div_score=div_score,
                                       center_bin_map=y['center_binary_map_dil'].to(self.device))

            # loss_dict['loss'].backward()
            loss_dict['loss'].backward()
            self.optimizer.step()
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        all_metrics = None
        for x, y in loader:
            results = self.model.forward(x.to(self.device))
            loss_dict = self.criterion(results, y['pointing_map'].to(self.device),
                                       target_mask=None if not self.learn_mask else y['mask'].to(self.device))
            all_metrics = update_metrics(loss_dict, all_metrics)
        return all_metrics

    @torch.no_grad()
    def make_figures(self, epoch, logger: Logger = None, subset=None):
        self.model.eval()
        output = self.model.forward(self.images_figs.to(self.device))
        loss_dict = self.criterion(output, self.label_figs['pointing_map'].to(self.device),
                                   target_mask=None if not self.learn_mask else self.label_figs['mask'].to(self.device))

        if logger is not None:
            metrics = np.array([logger.log['train_loss'], logger.log['val_loss']])
        else:
            metrics = None

        display_pos_inference(
            model=self,
            epoch=epoch,
            images=self.images_figs,
            output=output,
            labels=self.label_figs,
            results_path=self.save_path,
            subset=subset,
            loss=loss_dict,
            learn_mask=self.learn_mask,
            thickness=1,
            metrics=metrics,
            colors=['tab:orange', 'tab:blue']
        )

    @torch.no_grad()
    def infer_on_image(self, image, centers=None, params=None):
        self.model.eval()
        if centers is None or params is None:
            centers = np.array([])
            params = np.array([])
        img, label = self.label_processor_train.process(image, centers, params, idx=0)

        @torch.no_grad()
        def infer(input_image):
            padded_image, pad = pad_before_infer(input_image, depth=len(self.config["model"]["hidden_dims"]) - 1)

            padded_image = torch.unsqueeze(padded_image, 0)
            output = self.model.forward(padded_image.to(self.device))
            mask = torch.sigmoid(output[:, 2]).detach().cpu().numpy().squeeze()
            vec = output[:, :2].permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()
            if pad[0] > 0:
                mask = mask[:-pad[0]]
                vec = vec[:-pad[0]]

            if pad[1] > 0:
                mask = mask[:, :-pad[1]]
                vec = vec[:, :-pad[1]]
            return mask, vec

        try:
            output_mask, output_vec = infer(img)
        except Exception as e:
            print(f"inference failed because {e}")
            print("falling back to infer in patches")
            torch.cuda.empty_cache()
            shape = img.shape[1:]
            print(f"with image of shape {shape}")
            output_vec = np.empty(shape + (2,))
            output_mask = np.empty(shape)
            nx = shape[0] // PATCH_SIZE
            ny = shape[1] // PATCH_SIZE
            print(f"splitting into {nx} x {ny} patches")
            pbar = tqdm(desc="processing patches", total=(nx+1)*(ny+1))
            for i in range(nx + 1):
                for j in range(ny + 1):
                    s = np.s_[
                        i * PATCH_SIZE:min((i + 1) * PATCH_SIZE, shape[0]),
                        j * PATCH_SIZE:min((j + 1) * PATCH_SIZE, shape[1])
                        ]
                    img_crop = img[:, s[0], s[1]]
                    if img_crop.shape[1] == 0 or img_crop.shape[2] == 0:
                        print("woops empty slice")
                        continue
                    output_mask_crop, output_vec_crop = infer(img_crop)
                    output_vec[s] = output_vec_crop
                    output_mask[s] = output_mask_crop
                    pbar.update(1)
                    torch.cuda.empty_cache()

        return output_mask, output_vec, label

    @torch.no_grad()
    def compute_errors(self, rescale_fac=1.0):
        model_name = os.path.split(self.save_path)[1]
        densities_dir = os.path.join(get_dataset_base_path(), 'error_maps', self.dataset, 'train', model_name)
        make_if_not_exist(densities_dir, recursive=True)
        self.model.eval()
        densities_files = []
        id_re = re.compile(r'[^0-9]*([0-9]+).*.png')
        paths_dict = fetch_data_paths(self.dataset, 'train')
        patch_files = paths_dict['images']
        label_files = paths_dict['annotations']
        patch_files.sort(), label_files.sort()
        for pf, lf in zip(tqdm(patch_files, desc='computing errors'), label_files):
            img = plt.imread(pf)[..., :3]
            with open(lf, 'rb') as f:
                labels_dict = pickle.load(f)
            centers, params = labels_dict['centers'], labels_dict['parameters']

            output_mask, output_vec, label = self.infer_on_image(img, centers, params)

            target_mask = label['mask'].cpu().numpy().squeeze()
            error = np.abs(target_mask - output_mask)
            assert np.all(error <= 1)

            if rescale_fac != 1:
                error = rescale(error, scale=rescale_fac)

            path_id = id_re.match(os.path.split(pf)[1]).group(1)
            file = os.path.join(densities_dir, f"{path_id}.png")
            plt.imsave(file, error, cmap='gray', vmin=0, vmax=1)
            densities_files.append(file)

            del img, target_mask, label, output_mask, error
            torch.cuda.empty_cache()

        return densities_files

    def make_data_samples(self, loader: DataLoader, type: str, n_batch_samples=1):
        data_samples_dir = os.path.join(self.save_path, f'data_samples_{type}')
        if not os.path.exists(data_samples_dir):
            os.mkdir(data_samples_dir)
        loader_iter = loader.__iter__()
        for i in range(n_batch_samples):
            x, y = loader_iter.next()
            bin_maps = np.squeeze(y['center_binary_map'].cpu().numpy())
            image_arrays = x.detach().cpu().permute((0, 2, 3, 1)).numpy()
            for j in range(len(image_arrays)):
                file_name = f'training_sample_b{i:02}_{j:04}'
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_raw.png'), image_arrays[j])
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_centers.png'), bin_maps[j], cmap='gray')

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

            rescale_fac = 1 / 8

            if epoch % self.dataset_update_interval == 0 and epoch != 0:
                if self.error_update_interval is not None and epoch % self.error_update_interval == 0:
                    print('computing errors')
                    self.error_densities = self.compute_errors(
                        rescale_fac=rescale_fac
                    )
                # error_maps = self.compute_errors(os.path.join(self.data_path, 'train'))
                print("remaking patch dataset")
                make_patch_dataset(new_dataset=self.temp_dataset,
                                   source_dataset=self.dataset,
                                   config=self.config,
                                   make_val=False,
                                   sampling_densities=self.error_densities,
                                   densities_rescale_fac=rescale_fac,
                                   d_sampler_weight=1 / 2,
                                   rng=self.rng)
                self.data_train.update_files()
                print("patch dataset done, resuming !")

        self.make_figures(self.n_epochs - 1, subset=[0, 2, 3, 5])
        make_gif(self.save_path, 'res_*.png', 'res.gif')
        self.save()
        print("Saved model")
        self.clean()
        print("cleared temp files")

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pt'))
        torch.save(self.div_clf.state_dict(), os.path.join(self.save_path, 'model_div_clf.pt'))

    def clean(self):
        shutil.rmtree(os.path.join(get_dataset_base_path(), self.temp_dataset))

    def vec2detection_map(self, vector_map: np.ndarray, mask: np.ndarray,skip_sigmoid=False):
        if self.div_clf is not None:
            vec_t = torch.tensor(vector_map).permute((2, 0, 1))
            mask_t = torch.unsqueeze(torch.tensor(mask), dim=0)
            t = torch.unsqueeze(torch.concat([vec_t, mask_t], dim=0), dim=0).to(self.device).float()
            score = self.div_clf.forward(t)
            if skip_sigmoid:
                return np.squeeze(score.detach().cpu().numpy())
            return np.squeeze(torch.sigmoid(score).detach().cpu().numpy())

        div = divergence_map_from_vector_field(vector_map, normalize=True)
        return np.clip(-div / 2, 0, 1) * mask

    def infer(self, subset: str, min_confidence: float = 0.1, display_min_confidence: float = 0.5, overwrite=True):
        id_re = re.compile(r'([0-9]+).*.png')
        results_dir = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=self.dataset, subset=subset)
        dota_trlt = DOTAResultsTranslator(self.dataset, subset, results_dir, 'hbb', all_classes=['vehicle'])
        make_if_not_exist(results_dir, recursive=True)
        paths_dict = fetch_data_paths(self.dataset, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']

        for pf, af in zip(tqdm(image_paths, desc=f'inferring on {self.dataset}/{subset}'), annot_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))

            if os.path.exists(os.path.join(results_dir, f'{patch_id:04}_results.pkl')) and not overwrite:
                logging.info(f"{patch_id:04}_results.pkl exists, skipping")
                continue

            img = plt.imread(pf)[..., :3]
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)
            centers, params = labels_dict['centers'], labels_dict['parameters']

            output_mask, output_vec, _ = self.infer_on_image(img, centers, params)

            detection_map = self.vec2detection_map(output_vec, output_mask)

            detections_centers = np.array(np.where(detection_map > min_confidence)).T
            detections_scores = detection_map[detections_centers[:, 0], detections_centers[:, 1]]

            start = time.perf_counter()
            nms_centers, nms_scores = nms_distance(detections_centers, detections_scores, threshold=6)
            end = time.perf_counter()
            delta = end - start
            if delta > 20:
                logging.warning(f"nms took {delta}s maybe the min_confidence is too low")

            s = 12
            s1 = s // 2
            s2 = s - s1
            nms_boxes = np.array(
                [[c[1] - s1, c[0] - s1, c[1] + s2, c[0] + s2] for c in nms_centers])  # x1,y1,x2,y2

            gt_as_boxes = np.array(
                [[c[1] - s1, c[0] - s1, c[1] + s2, c[0] + s2] for c in centers])

            image_w_bboxes = bboxes_over_image_cv2(
                images=img,
                boxes=[b for b, s in zip(nms_boxes, nms_scores) if s >= display_min_confidence],
                scores=[s for s in nms_scores if s >= display_min_confidence],
                color='plasma')

            image_w_gt = bboxes_over_image_cv2(
                images=img,
                boxes=gt_as_boxes,
                color=(0, 1.0, 0))

            results_dict = {
                'detection': detections_centers,
                'detection_score': detections_scores,
                'detection_type': 'center',
                'detection_map': detection_map
            }

            gt_poly = np.array([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] for b in gt_as_boxes])

            dota_trlt.add_gt(image_id=patch_id, polygons=gt_poly, difficulty=labels_dict['difficult'], flip_coor=False,
                             categories=['vehicle' for _ in gt_poly])
            dota_trlt.add_detections(image_id=patch_id, scores=nms_scores, bbox=nms_boxes, flip_coor=False,
                                     class_names=['vehicle' for _ in nms_scores])

            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_nms_detection.png'), image_w_bboxes)
            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_gt.png'), image_w_gt)
            with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)
        dota_trlt.save()
        print('saved translations')

    def eval(self):
        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='hbb'
        )
