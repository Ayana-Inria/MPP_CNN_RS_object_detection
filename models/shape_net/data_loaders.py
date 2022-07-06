import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage import draw
from skimage.segmentation import watershed
from torch import Tensor

from data.image_dataset import LabelProcessor
from models.shape_net.mappings import ValueMapping, values_to_class_id
from base.shapes.rectangle import rect_to_poly, wla_to_sra


@dataclass
class LossMaskParams:
    mode: str
    mask_sigma: Union[float, str] = None
    mask_cutoff_dist: float = None

    def __post_init__(self):
        assert self.mode in ['gaussian', 'shapes']
        if self.mode == 'shapes':
            if self.mask_sigma is not None:
                logging.info('mask_sigma is ignored since mode == shapes')
                # print('[LossMaskParams] warning, mask_sigma is ignored since mode == shapes')
            if self.mask_cutoff_dist is not None:
                logging.info('mask_cutoff_dist is ignored since mode == shapes')
                print('[LossMaskParams] warning, mask_cutoff_dist is ignored since mode == shapes')


@dataclass
class ShapePatchProcessor(LabelProcessor):
    mappings: List[ValueMapping]
    class_perturbation_dict: Optional[Dict[int, float]]
    rng: np.random.Generator
    mask_params: LossMaskParams

    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx: int) \
            -> Tuple[Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        n_points = len(centers)

        params_srw = [wla_to_sra(a, b, w) for a, b, w in params]
        n_feat = len(self.mappings)
        classes = values_to_class_id(params_srw, self.mappings, as_tensor=False)
        if self.class_perturbation_dict is not None:
            for k in range(n_points):
                for i in range(n_feat):  # iterate over shape features
                    pert = self.rng.choice(list(self.class_perturbation_dict.keys()),
                                           p=list(self.class_perturbation_dict.values()))

                    c = classes[i][k]
                    if self.mappings[i].is_cyclic:
                        classes[i][k] = (c + pert) % self.mappings[i].n_classes
                    else:
                        classes[i][k] = int(np.clip((c + pert), 0, self.mappings[i].n_classes - 1))

        center_bin_map = np.zeros(patch.shape[:2], dtype=bool)
        for k, c in enumerate(centers):
            try:
                center_bin_map[c[0], c[1]] = 1
            except IndexError as e:
                logging.info(f"point ({c}) out of bounds in patch of shape {patch.shape}: {e}")

        distance = distance_transform_edt(1 - center_bin_map)

        if self.mask_params.mode == 'gaussian':

            value_class_map = []
            for i in range(n_feat):  # iterate over shape features

                param_seed = np.zeros(patch.shape[:2], dtype=int)
                for k, c in enumerate(centers):
                    param_seed[c[0], c[1]] = classes[i][k]

                param_map = watershed(distance, param_seed)

                value_class_map.append(param_map)
            size_map = self.mappings[0].class_to_value(value_class_map[0])
            if self.mask_params.mask_sigma == 'auto':
                assert size_map is not None
                sigma = size_map / 4
                loss_mask = np.exp(-0.5 * np.square(distance / sigma))
                loss_mask[loss_mask < 1e-3] = 0
            else:
                loss_mask = np.exp(-0.5 * np.square(distance / self.mask_params.mask_sigma))
                loss_mask[distance >= self.mask_params.mask_cutoff_dist] = 0
            if len(centers) == 0:
                loss_mask = np.zeros(loss_mask.shape)
            else:
                loss_mask = loss_mask / np.sum(loss_mask)

        else:
            value_class_map = [np.zeros(patch.shape[:2], dtype=int) for _ in range(3)]
            loss_mask = np.zeros(patch.shape[:2], dtype=bool)
            for k, (c, p) in enumerate(zip(centers, params)):
                a, b, w = p
                object_mask = draw.polygon2mask(
                    patch.shape[:2],
                    rect_to_poly(c, a, b, w))
                loss_mask += object_mask
                for i in range(n_feat):
                    value_class_map[i][object_mask] = classes[i][k]
            if len(centers) == 0:
                loss_mask = np.zeros(loss_mask.shape)
            else:
                loss_mask = loss_mask / np.sum(loss_mask)

        patch = torch.tensor(patch).permute((2, 0, 1))
        label_dict = {
            'value_class_map': [torch.tensor(v, dtype=torch.long) for v in value_class_map],
            'center_binary_map': torch.tensor(center_bin_map),
            'distance_map': torch.tensor(distance),
            'loss_mask': torch.tensor(loss_mask)
        }
        return patch, label_dict


def gaussian_filter(shape: Tuple[int, int], sigma: float):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
