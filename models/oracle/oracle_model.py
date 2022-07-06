import os
import pickle
import re

from tqdm import tqdm

from base.base_model import BaseModel
from metrics.dota_eval import dota_eval
from metrics.dota_results_translator import DOTAResultsTranslator
from models.shape_net.display import _pred_to_image2
from base.shapes.rectangle import rect_to_poly
from utils.data import fetch_data_paths, get_inference_path
from utils.files import make_if_not_exist
from utils.training import startup_config
import matplotlib.pyplot as plt
import numpy as np


class OracleModel(BaseModel):
    def __init__(self, config: dict, dataset: str):
        self.config, self.logger, self.save_path = startup_config(config, 'oracle', load_model=False,
                                                                  overwrite=True)
        print("This is an Oracle model, for testing purposes only")
        self.dataset = dataset

    def train(self):
        print("The oracle model won't train")

    def infer(self, subset: str, min_confidence: float, display_min_confidence: float, overwrite: bool):
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

            img = plt.imread(pf)[..., :3]
            with open(af, 'rb') as f:
                labels_dict = pickle.load(f)
            centers, params = labels_dict['centers'], labels_dict['parameters']
            difficulty = labels_dict['difficult']

            gt_as_poly = np.array(
                [rect_to_poly(c, short=p[0], long=p[1], angle=p[2]) for c, p in zip(centers, params)])

            pred_scores = [1.0 for _ in gt_as_poly]

            results_dict = {
                'detection': gt_as_poly,
                'detection_type': 'poly',
                'detection_center': centers,
                'detection_score': pred_scores,
                'detection_params': params
            }

            image_w_gt = _pred_to_image2(
                centers=centers,
                params=params,
                scores=None,
                image=img,
                color=(0, 1.0, 0))

            dota_trlt.add_gt(image_id=patch_id, polygons=gt_as_poly, difficulty=difficulty)

            dota_trlt.add_detections(image_id=patch_id, scores=pred_scores, polygons=gt_as_poly,
                                     flip_coor=True, class_names=['vehicle' for _ in pred_scores])

            plt.imsave(os.path.join(results_dir, f'{patch_id:04}_gt.png'), image_w_gt)
            with open(os.path.join(results_dir, f'{patch_id:04}_results.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)

        dota_trlt.save()
        print('saved coco translation')

    def eval(self):
        dota_eval(
            model_dir=self.save_path,
            dataset=self.dataset,
            subset='val',
            det_type='obb'
        )

    def data_preview(self):
        pass
