from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np

from base.shapes.rectangle import Rectangle
from models.mpp.point_set.energy_point_set import EPointsSet
from models.shape_net.mappings import ValueMapping


@dataclass
class ImageWMaps:
    name: str
    shape: Tuple[int, int]
    image: np.ndarray
    detection_map: np.ndarray
    param_dist_maps: List[np.ndarray]
    mappings: List[ValueMapping]
    param_names: List[str]
    labels: Dict[str, Any] = None
    gt_config: List[Rectangle] = None
    gt_config_set: EPointsSet = None
    crop_data: Dict = None