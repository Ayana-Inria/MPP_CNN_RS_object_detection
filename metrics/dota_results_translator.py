from typing import List, Union

import numpy as np
import os

from data.DOTA_devkit.dota_utils import dots4ToRec4
from utils.files import make_if_not_exist


class DOTAResultsTranslator:
    def __init__(self, dataset: str, subset: str, results_dir: str, det_type: str, all_classes: List[str],
                 postfix: str = ''):
        assert det_type in ['obb', 'hbb']
        self.det_type = det_type
        self.det_dir = os.path.join(results_dir, 'dota' + postfix, 'det')
        self.annot_dir = os.path.join(results_dir, 'dota' + postfix, 'gt')
        self.image_set = []
        self.image_set_file = os.path.join(results_dir, 'dota' + postfix, 'imageSet.txt')
        self.det_lines_per_cat = {k: [] for k in all_classes}

        make_if_not_exist([self.det_dir, self.annot_dir], recursive=True)

    def add_gt(self, image_id: int, difficulty: Union[List, np.ndarray], polygons: np.ndarray, categories,
               flip_coor=True):
        self.image_set.append(f"{image_id:04}")
        lines = []
        n_gt = len(polygons)
        for i in range(n_gt):
            p = polygons[i]

            if flip_coor:
                p = np.flip(p, axis=-1)

            if self.det_type == 'hbb':
                xmin, ymin, xmax, ymax = dots4ToRec4(p)
                p = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

            p = p.astype(int)

            coor_str = ' '.join([str(a) for a in p.ravel()])

            lines.append(' '.join([coor_str, categories[i], str(int(difficulty[i]))]))

        with open(os.path.join(self.annot_dir, f'{image_id:04}.txt'), 'w') as f:
            f.write('\n'.join(lines))

    def add_detections(self, image_id: int, scores, class_names, polygons: np.ndarray = None, bbox=None,
                       flip_coor=True):

        if polygons is not None:
            n_det = len(polygons)
        else:
            n_det = len(bbox)
        for i in range(n_det):
            if polygons is not None:
                p = polygons[i]
                if flip_coor:
                    p = np.flip(p, axis=-1)

                coor_str = ' '.join([f"{a:.1f}" for a in p.ravel()])
            elif bbox is not None:
                if flip_coor:
                    p = [bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]]
                else:
                    p = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
                coor_str = ' '.join([f"{a:.1f}" for a in p])
            else:
                raise ValueError

            str_line = ' '.join([f'{image_id:04}', str(scores[i]), coor_str])
            self.det_lines_per_cat[class_names[i]].append(str_line)

    def save(self):

        for class_name, det_class in self.det_lines_per_cat.items():
            with open(os.path.join(self.det_dir, f'{class_name}.txt'), 'w') as f:
                f.write('\n'.join(det_class))

        with open(self.image_set_file, 'w') as f:
            f.write('\n'.join(self.image_set))
