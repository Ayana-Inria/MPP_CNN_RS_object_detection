import logging
from dataclasses import dataclass
from typing import Union, Tuple, Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from torch import Tensor

from data.image_dataset import LabelProcessor

MODES = ['vec', 'uvec', 'dist']


@dataclass
class PosPatchProcessor(LabelProcessor):
    max_distance: Union[str, float]
    mode: str
    n_classes: Optional[int] = None
    sigma_dil: float = None

    def process(self, patch: np.ndarray, centers: np.ndarray, params: np.ndarray, idx: int) -> Tuple[
        Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        center_bin_map = np.zeros(patch.shape[:2], dtype=bool)
        for k, c in enumerate(centers):
            try:
                center_bin_map[c[0], c[1]] = 1
            except IndexError as e:
                logging.info(f"point ({c}) out of bounds in patch of shape {patch.shape}: {e}")
                # raise e

        distance = distance_transform_edt(1 - center_bin_map)

        if self.sigma_dil is None:
            sigma_dil = 0.6
        else:
            sigma_dil = self.sigma_dil
        center_bin_map_dil = np.exp(-0.5 * np.square(distance / sigma_dil))
        center_bin_map_dil[center_bin_map_dil < 1e-5] = 0

        compute_size_map = self.max_distance == 'auto'

        if compute_size_map:
            n_feat = len(params[0])
            params_map = []
            for i in range(n_feat):  # iterate over shape features

                param_seed = np.zeros(patch.shape[:2], dtype=int)
                for k, c in enumerate(centers):
                    param_seed[c[0], c[1]] = params[k][i]

                param_map = watershed(distance, param_seed)

                params_map.append(param_map)

            size_map = (params_map[0] + params_map[1]) / 2
        else:
            size_map = None

        if self.mode in ['vec', 'uvec']:

            vector_seed = np.zeros(patch.shape[:2], dtype=int)
            for k, c in enumerate(centers):
                try:
                    vector_seed[c[0], c[1]] = k + 1
                except IndexError as e:
                    pass

            all_centers = np.array(centers)

            vector_map = watershed(distance, vector_seed) - 1
            # vector_map = all_centers[vector_map.ravel()].reshape(patch.shape[:2]+(2,))
            if len(centers) == 0:
                pointy_map = np.zeros(patch.shape[:2] + (2,))
            else:
                vector_map = all_centers[vector_map]

                coor_map = np.stack(np.mgrid[:patch.shape[0], :patch.shape[1]], axis=-1)

                pointy_map = vector_map - coor_map
            norm = np.linalg.norm(pointy_map, axis=-1) + 1e-8
            if self.mode == 'uvec':
                pointy_map = pointy_map / np.stack((norm, norm), axis=-1)
                pointy_map[np.isnan(pointy_map)] = 0
            # pointy_map = pointy_map / np.stack((norm, norm), axis=-1)
            # pointy_map[np.isnan(pointy_map)] = 0
            if compute_size_map:
                mask = norm > size_map
            else:
                mask = norm > self.max_distance
            if len(centers) == 0:
                mask = np.ones(mask.shape, dtype=bool)
            pointy_map[mask] = 0

            label_dict = {
                'pointing_map': torch.tensor(pointy_map, dtype=torch.float).permute((2, 0, 1)),
                'mask': torch.tensor(~mask, dtype=torch.float32),
                'center_binary_map': torch.tensor(center_bin_map),
                'center_binary_map_dil': torch.tensor(center_bin_map_dil, dtype=torch.float32),
                'distance_map': torch.tensor(distance),
            }
        else:
            if compute_size_map:
                sigma = size_map / 4
            else:
                sigma = self.max_distance / 2
            blob_map = torch.exp(-0.5 * torch.square(torch.tensor(distance) / sigma))
            blob_map[blob_map < 1e-3] = 0

            blob_map_class = (blob_map * (self.n_classes - 1)).type(torch.long)

            label_dict = {
                'blob_map': blob_map,
                'blob_map_class': blob_map_class,
                'center_binary_map': torch.tensor(center_bin_map),
                'center_binary_map_dil': torch.tensor(center_bin_map_dil, dtype=torch.float32),
                'distance_map': torch.tensor(distance),
            }
        if compute_size_map:
            label_dict['size_map']: torch.tensor(size_map)

        patch = torch.tensor(patch).permute((2, 0, 1))
        return patch, label_dict
