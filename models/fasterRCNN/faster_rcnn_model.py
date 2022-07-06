import json
import logging
import math
import os
import pickle
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from base.base_model import BaseModel, TorchModel
from data.patch_making import make_patch_dataset
from metrics.dota_eval import dota_eval
from metrics.dota_results_translator import DOTAResultsTranslator
from models.fasterRCNN.display import display_rcnn_inference
from utils.display.boxes import bboxes_over_image_cv2
from models.fasterRCNN.patch_processor import FasterRCNNPatchProcessor
from base.shapes.rectangle import rect_to_poly
from utils.data import get_dataset_base_path, fetch_data_paths, split_image, get_inference_path
from utils.files import make_gif, make_if_not_exist
from utils.logger import Logger
from utils.misc import timestamp, append_lists_in_dict
from utils.nms import nms
from utils.training import PatchBasedTrainer, Config, startup_config, print_metrics


def collate_fn(batch):
    return tuple(zip(*batch))


def build_faster_rcnn_model(num_classes):
    # From Camilo Aguilar
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.bba_models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Stop here if you are fine-tunning Faster-RCNN

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,  hidden_layer, num_classes)
    return model


class FasterRCNNModel(BaseModel, PatchBasedTrainer, TorchModel):

    def __init__(self, config: Config, train: bool, load=False, reuse_data=False, overwrite=False, dataset: str = None):
        self.config, self.logger, self.save_path = startup_config(config, 'fasterrcnn', load_model=load,
                                                                  overwrite=overwrite)

        self.dataset = self.config['data_loader']["dataset"] if dataset is None else dataset
        self.temp_dataset = 'temp_' + self.config['model_name'] + '_' + timestamp()

        self.n_epochs = self.config['trainer']['n_epochs']
        self.last_epoch = 0
        self.batch_size = self.config['trainer']['batch_size']
        self.figure_interval = self.config['trainer']['figure_interval']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device is {self.device}")

        self.model = build_faster_rcnn_model(2).to(self.device)
        self.logger.log_model(checkpoint_interval=4, model=self.model)

        if load:
            self._load()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        self.rng = np.random.default_rng(42)

        label_processor = FasterRCNNPatchProcessor(

        )
        self.label_processor_train = label_processor
        self.label_processor_val = label_processor

        if train:
            self.__init_data__(reuse_data=reuse_data, collate_fn=collate_fn)

    def infer_on_images(self, images, confidence=0.25):
        self.model.eval()
        images = list(image.to(self.device) for image in images)
        pred = self.model(images)

        all_boxes = []
        all_scores = []

        for p in pred:
            boxes = p['boxes']
            scores = p['scores']
            # labels = p['labels']

            # NMS
            nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.config["model"]["iou_threshold"])
            nms_boxes = boxes[nms_indices]
            nms_scores = scores[nms_indices]

            pred_scores = list(nms_scores.detach().cpu().numpy())
            pred_boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in
                          list(nms_boxes.detach().cpu().numpy())]

            pred_scores_t = [p for p in pred_scores if p >= confidence]
            pred_boxes_t = [b for i, b in enumerate(pred_boxes) if pred_scores[i] >= confidence]

            all_boxes.append(pred_boxes_t)
            all_scores.append(pred_scores_t)

        return all_boxes, all_scores

    def make_data_samples(self, loader: DataLoader, type: str, n_batch_samples=1):
        data_samples_dir = os.path.join(self.save_path, f'data_samples_{type}')
        if not os.path.exists(data_samples_dir):
            os.mkdir(data_samples_dir)
        loader_iter = loader.__iter__()
        for i in range(n_batch_samples):
            x, y = loader_iter.next()
            for j in range(len(x)):
                image = x[j].detach().cpu().permute((1, 2, 0)).numpy()
                boxes = y[j]['boxes'].cpu().numpy()
                file_name = f'training_sample_b{i:02}_{j:04}'
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_raw.png'), image)

                box_image = bboxes_over_image_cv2([image], [boxes])[0]
                plt.imsave(os.path.join(data_samples_dir, f'{file_name}_label.png'), box_image)

    def train_epoch(self, epoch: int, data_loader: DataLoader, scaler=None):
        self.model.train()
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        # header = f"Epoch: [{epoch}]"

        lr_scheduler = None
        metrics = {}
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        for images, targets in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # todo (maybe) reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                losses.backward()
                self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            append_lists_in_dict(metrics, {
                "loss": losses_reduced.detach().cpu().numpy(),
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: v.detach().cpu().numpy() for k, v in loss_dict_reduced.items()}
            })

        return metrics

    @torch.no_grad()
    def val_epoch(self, data_loader: DataLoader):
        self.model.train()
        metrics = {}

        for images, targets in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=False):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # todo (maybe) reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            append_lists_in_dict(metrics, {
                "loss": losses_reduced.detach().cpu().numpy(),
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: v.detach().cpu().numpy() for k, v in loss_dict_reduced.items()}
            })

        return metrics

    def data_preview(self):
        self.make_data_samples(self.train_loader, 'train')
        self.make_data_samples(self.val_loader, 'val')

    @torch.no_grad()
    def make_figures(self, epoch, logger: Logger = None):
        self.model.eval()
        all_boxes, all_scores = self.infer_on_images([i.to(self.device) for i in self.images_figs])
        gt_boxes = [l['boxes'] for l in self.label_figs]
        train_loss = logger.log['train_loss']
        val_loss = logger.log['val_loss']

        display_rcnn_inference(
            epoch=epoch, images=self.images_figs, gt_boxes=gt_boxes, boxes=all_boxes, scores=all_scores,
            train_loss=train_loss,
            val_loss=val_loss, colors=['tab:orange', 'tab:blue'], results_path=self.save_path)

    def train(self):
        self.data_preview()
        for epoch in range(self.last_epoch, self.n_epochs):

            # train for one epoch, printing every 10 iterations
            train_metrics = self.train_epoch(epoch, self.train_loader)

            val_metrics = self.val_epoch(self.val_loader)

            print_metrics(epoch, train_metrics, val_metrics)

            self.logger.update_train_val(epoch, train_metrics, val_metrics)

            if epoch % self.figure_interval == 0 or epoch == self.n_epochs - 1:
                self.make_figures(epoch, logger=self.logger)

            if epoch % self.dataset_update_interval == 0 and epoch != 0:
                # error_maps = self.compute_errors(os.path.join(self.data_path, 'train'))
                make_patch_dataset(new_dataset=self.temp_dataset,
                                   source_dataset=self.dataset,
                                   config=self.config,
                                   make_val=False,
                                   rng=self.rng)
                self.data_train.update_files()

        self.make_figures(self.n_epochs)
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
        processor = FasterRCNNPatchProcessor()
        id_re = re.compile(r'([0-9]+).*.png')

        results_dir = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=self.dataset, subset=subset)
        # coco_trlt = COCOTranslator(self.dataset, subset)
        dota_trlt = DOTAResultsTranslator(self.dataset, subset, results_dir, 'hbb', all_classes=['vehicle'])
        make_if_not_exist(results_dir, recursive=True)
        paths_dict = fetch_data_paths(self.dataset, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']
        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {self.dataset}/{subset}'), annot_paths,
                              meta_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            if os.path.exists(os.path.join(results_dir, f'{patch_id:04}_results.pkl')) and not overwrite:
                print(f"{patch_id:04}_results.pkl exists, skipping")
                continue

            img = plt.imread(pf)[..., :3]
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)
            centers, params = labels_dict['centers'], labels_dict['parameters']
            difficulty = labels_dict['difficult']
            with open(mf, 'r') as f:
                meta_dict = json.load(f)

            image_patches = split_image(img, centers, params,
                                        target_size=self.config["data_loader"]['patch_maker_params']['patch_size'])
            all_boxes = []
            all_scores = []

            for i in range(len(image_patches)):
                sub_image = image_patches[i]['image']
                sub_centers = image_patches[i]['centers']
                sub_params = image_patches[i]['parameters']
                anchor = image_patches[i]['anchor']

                tensor_image, target = processor.process(sub_image, sub_centers, sub_params, idx=patch_id)

                nms_boxes, nms_scores = self.infer_on_images([tensor_image], confidence=min_confidence)
                nms_boxes = nms_boxes[0]
                nms_scores = nms_scores[0]

                all_boxes = all_boxes + [[b[0] + anchor[1], b[1] + anchor[0], b[2] + anchor[1], b[3] + anchor[0]]
                                         for b in nms_boxes]
                all_scores = all_scores + nms_scores

            nms_boxes, nms_scores = nms(all_boxes, all_scores, threshold=0.5)

            _, target = processor.process(img, centers, params, idx=patch_id)

            image_w_bboxes = bboxes_over_image_cv2(
                images=img,
                boxes=[b for b, s in zip(nms_boxes, nms_scores) if s >= display_min_confidence],
                scores=[s for s in nms_scores if s >= display_min_confidence],
                color='plasma')

            image_w_gt = bboxes_over_image_cv2(
                images=img,
                boxes=target['boxes'],
                color=(0, 1.0, 0))

            results_dict = {
                'detection': nms_boxes,
                'detection_score': nms_scores,
                'detection_type': 'bbox',
                'gt': centers,
                'gt_bbox': target['boxes']
            }
            gt_as_poly = np.array(
                [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])
            # coco_trlt.add_image(image_id=patch_id, file_name=os.path.split(pf)[1], metadata=meta_dict)
            # coco_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly)
            # coco_trlt.add_detections(
            #     image_id=patch_id, image_shape=img.shape[:2], scores=nms_scores, bbox=nms_boxes, flip_coor=False)

            dota_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly, difficulty=difficulty, flip_coor=True)
            dota_trlt.add_detections(image_id=patch_id, scores=nms_scores, bbox=nms_boxes, flip_coor=False,
                                     class_names=['vehicle' for _ in nms_scores])

            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_nms_detection.png'), image_w_bboxes)
            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_gt.png'), image_w_gt)
            with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)

        # coco_trlt.save(results_dir)
        dota_trlt.save()
        print('saved translations')

    def eval(self):
        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='hbb'
        )
