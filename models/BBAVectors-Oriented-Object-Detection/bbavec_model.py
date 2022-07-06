import os
import pickle
import re
import shutil
from collections import namedtuple

import cv2
import numpy as np
import pandas
import torch
from tqdm import tqdm

import eval
import train
import decoder
from base.shapes.rectangle import polygon_to_abw, rect_to_poly
from datasets.dataset_dota import DOTA
from datasets.dataset_dota_gsd50 import CustomDOTA
from datasets.dataset_hrsc import HRSC
from base.base_model import BaseModel
from bbav_models import ctrbox_net
from func_utils import decode_prediction, non_maximum_suppression
from metrics.dota_eval import dota_eval
from metrics.dota_results_translator import DOTAResultsTranslator
from models.shape_net.display import _pred_to_image2
from utils.data import get_dataset_base_path, get_inference_path, fetch_data_paths
from utils.files import make_if_not_exist
from utils.training import startup_config
import matplotlib.pyplot as plt


class BBAVec(BaseModel):
    def __init__(self, config: dict, load=False, overwrite=False,dataset=None):
        self.config, self.logger, self.save_path = startup_config(config, 'bbavec', load_model=load,
                                                                  overwrite=overwrite)

        if dataset is not None:
            config["data_dir"] = dataset

        ArgsConfig = namedtuple('args', config)
        self.args = ArgsConfig(**config)

        self.dataset = {'dota': DOTA, 'hrsc': HRSC, 'dota_gsd50': CustomDOTA}

        self.num_classes = {'dota': 15, 'hrsc': 1, 'dota_gsd50': 1}
        self.heads = {'hm': self.num_classes[self.args.dataset],
                      'wh': 10,
                      'reg': 2,
                      'cls_theta': 1
                      }
        self.down_ratio = 2
        self.model = ctrbox_net.CTRBOX(heads=self.heads,
                                       pretrained=True,
                                       down_ratio=self.down_ratio,
                                       final_kernel=1,
                                       head_conv=3)
        self.decoder = decoder.DecDecoder(K=self.args.K,
                                          conf_thresh=self.args.conf_thresh,
                                          num_classes=self.num_classes[self.args.dataset])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        ctrbox_obj = train.TrainModule(dataset=self.dataset,
                                       num_classes=self.num_classes,
                                       model=self.model,
                                       decoder=self.decoder,
                                       down_ratio=self.down_ratio)

        ctrbox_obj.train_network(self.args, save_path=self.save_path)

        self.save()

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pt'))

    def eval(self):
        dota_eval(
            model_dir=self.save_path,
            dataset=self.args.data_dir,
            subset='val',
            det_type='obb'
        )

        dota_eval(
            model_dir=self.save_path,
            dataset=self.args.data_dir,
            subset='val',
            det_type='obb',
            postfix='-SV'
        )

    def infer(self, subset: str = 'val', min_confidence: float = None, display_min_confidence: float = None,
              overwrite: bool = False):

        ctrbox_obj = eval.EvalModule(dataset=self.dataset, num_classes=self.num_classes, model=self.model,
                                     decoder=decoder)
        save_path = os.path.join(self.save_path, 'weights_' + self.args.dataset)
        resume = 'model_last.pth'
        self.model = ctrbox_obj.load_model(self.model, os.path.join(save_path, resume))
        self.model = self.model.to(self.device)
        self.model.eval()
        print('loaded model !')

        inference_path = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=self.args.data_dir, subset='val')
        make_if_not_exist(inference_path, recursive=True)
        print(f'saving to {inference_path}')

        dataset_name = self.args.data_dir

        dota_trlt = DOTAResultsTranslator(dataset_name, subset, inference_path, det_type='obb', all_classes=['vehicle'])
        dota_trlt_2 = DOTAResultsTranslator(dataset_name, subset, inference_path, det_type='obb',
                                            all_classes=['vehicle'], postfix='-SV')

        id_re = re.compile(r'([0-9]+).*.png')
        paths_dict = fetch_data_paths(dataset_name, subset=subset)
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']
        category = ['vehicle']

        img_ids = [int(id_re.match(os.path.split(pf)[1]).group(1)) for pf in image_paths]
        results = {cat: {f'{img_id:04}': [] for img_id in img_ids} for cat in category}
        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {dataset_name}/{subset}'), annot_paths,
                              meta_paths):
            img_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            img = cv2.imread(pf)
            # cv2.resize(img, dsize=(img.shape[0] * 2, img.shape[1] * 2))
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)
            difficulty = labels_dict['difficult']
            centers, params = labels_dict['centers'], labels_dict['parameters']

            gt_as_poly = np.array(
                [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])

            shape = img.shape

            PATCH_SIZE = 608

            nx = int(np.ceil(shape[0] / PATCH_SIZE))
            ny = int(np.ceil(shape[1] / PATCH_SIZE))
            print(f"splitting into {nx} x {ny} patches")

            anchors_x = np.linspace(0, shape[0] - PATCH_SIZE, nx, dtype=int)
            anchors_y = np.linspace(0, shape[1] - PATCH_SIZE, ny, dtype=int)

            decoded_pts = []
            decoded_scores = []

            for ax in anchors_x:
                for ay in anchors_y:
                    tl_anchor = np.array([ax, ay])
                    s = np.s_[
                        ax:ax + PATCH_SIZE,
                        ay:ay + PATCH_SIZE
                        ]
                    img_crop = img[s[0], s[1], :]

                    bot_pad = PATCH_SIZE - img_crop.shape[0]
                    right_pad = PATCH_SIZE - img_crop.shape[1]
                    if bot_pad > 0 or right_pad > 0:
                        img_crop = cv2.copyMakeBorder(img_crop, 0, bot_pad, 0, right_pad,
                                                      borderType=cv2.BORDER_CONSTANT)

                    if img_crop.shape[1] == 0 or img_crop.shape[2] == 0:
                        print("woops empty slice")
                        continue

                    h, w, c = img_crop.shape

                    input_w = PATCH_SIZE
                    input_h = PATCH_SIZE

                    image = cv2.resize(img_crop, (input_w, input_h))
                    out_image = image.astype(np.float32) / 255.
                    out_image = out_image - 0.5
                    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
                    # print(out_image.shape,out_image.max(),out_image.min())
                    out_image = torch.from_numpy(out_image)

                    with torch.no_grad():
                        pr_decs = self.model(out_image.to(self.device))

                    torch.cuda.synchronize(self.device)
                    predictions = self.decoder.ctdet_decode(pr_decs)

                    predictions = predictions[0, :, :]

                    print(f"image:{img_id:04} "
                          f"anchor:({tl_anchor[0]:4}, {tl_anchor[1]:4}) "
                          f"shape:{img_crop.shape} "
                          f"padded:({bot_pad:3}, {right_pad:3}) "
                          f"predictions:{len(predictions):4} ")

                    pts0 = {cat: [] for cat in category}
                    scores0 = {cat: [] for cat in category}
                    for pred in predictions:
                        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
                        tt = np.asarray([pred[2], pred[3]], np.float32)
                        rr = np.asarray([pred[4], pred[5]], np.float32)
                        bb = np.asarray([pred[6], pred[7]], np.float32)
                        ll = np.asarray([pred[8], pred[9]], np.float32)
                        tl = tt + ll - cen_pt
                        bl = bb + ll - cen_pt
                        tr = tt + rr - cen_pt
                        br = bb + rr - cen_pt
                        score = pred[10]
                        clse = pred[11]
                        pts = np.asarray([tr, br, bl, tl], np.float32)
                        pts[:, 0] = (pts[:, 0] * self.down_ratio / input_w * w) + tl_anchor[1]
                        pts[:, 1] = (pts[:, 1] * self.down_ratio / input_h * h) + tl_anchor[0]
                        pts0[category[int(clse)]].append(pts)
                        scores0[category[int(clse)]].append(score)

                    decoded_pts.append(pts0)
                    decoded_scores.append(scores0)

            # nms
            for cat in category:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = non_maximum_suppression(pts_cat, scores_cat)
                    results[cat][f'{img_id:04}'].extend(nms_results)

            dota_trlt.add_gt(image_id=img_id, polygons=gt_as_poly, difficulty=difficulty,
                             categories=['vehicle' for _ in gt_as_poly])

            dota_trlt_2.add_gt(
                image_id=img_id, polygons=gt_as_poly,
                difficulty=[d or c == 'large-vehicle' for d, c in
                            zip(labels_dict['difficult'], labels_dict['categories'])],
                categories=['vehicle' for _ in gt_as_poly])

        for cat in category:
            if cat == 'background':
                continue
            with open(os.path.join(inference_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
                for img_id in results[cat]:
                    for pt in results[cat][img_id]:
                        f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            f'{img_id}', pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))

                        dota_trlt.det_lines_per_cat[cat].append(
                            f'{img_id} {pt[8]:.12f} {pt[0]:.1f} {pt[1]:.1f} {pt[2]:.1f} {pt[3]:.1f} {pt[4]:.1f} '
                            f'{pt[5]:.1f} {pt[6]:.1f} {pt[7]:.1f}'
                        )

                        dota_trlt_2.det_lines_per_cat[cat].append(
                            f'{img_id} {pt[8]:.12f} {pt[0]:.1f} {pt[1]:.1f} {pt[2]:.1f} {pt[3]:.1f} {pt[4]:.1f} '
                            f'{pt[5]:.1f} {pt[6]:.1f} {pt[7]:.1f}'
                        )

        dota_trlt.save()
        dota_trlt_2.save()
        print('saved coco translation')
        self.display_inference()

    def data_preview(self):
        pass

    def display_inference(self):
        inference_path = get_inference_path(
            model_name=os.path.split(self.save_path)[1], dataset=self.args.data_dir, subset='val')
        inference_file = os.path.join(inference_path, 'Task1_vehicle.txt')

        df = pandas.read_csv(inference_file, sep=' ', names=[
            'patch_id', 'score', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'
        ])
        id_re = re.compile(r'([0-9]+).*.png')
        paths_dict = fetch_data_paths(self.args.data_dir, subset='val')
        image_paths = paths_dict['images']
        annot_paths = paths_dict['annotations']
        meta_paths = paths_dict['metadata']

        assert len(image_paths) > 0

        display_min_confidence = 0.1

        for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {self.args.data_dir}/val)'), annot_paths,
                              meta_paths):
            patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
            detections = df[df['patch_id'] == patch_id]

            img = plt.imread(pf)[..., :3]

            polygons_x = detections[['x1', 'x2', 'x3', 'x4']].to_numpy()
            polygons_y = detections[['y1', 'y2', 'y3', 'y4']].to_numpy()
            polygons = np.stack([polygons_x, polygons_y], axis=-1)
            polygons = np.flip(polygons, axis=-1)
            print(f"{patch_id:04}: poly {polygons.shape}")

            pred_centers = np.mean(polygons, axis=1)
            pred_scores = detections['score'].to_numpy()

            pred_params = np.array(list(map(polygon_to_abw, polygons)))

            image_w_pred = _pred_to_image2(
                centers=[c for c, s in zip(pred_centers, pred_scores) if s >= display_min_confidence],
                params=[p for p, s in zip(pred_params, pred_scores) if s >= display_min_confidence],
                scores=[s for s in pred_scores if s >= display_min_confidence],
                image=img,
                color='plasma')

            plt.imsave(os.path.join(inference_path, f'{patch_id:04}_detection.png'), image_w_pred)
