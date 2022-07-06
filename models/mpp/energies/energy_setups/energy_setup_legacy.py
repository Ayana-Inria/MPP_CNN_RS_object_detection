import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Any, Dict

import numpy as np
from numpy.random import Generator

from base.shapes.rectangle import Rectangle
from models.mpp.calibration.energy_calibration import calibrate_detection_threshold, \
    calibrate_param_dists, calibrate_min_area, calibrate_alignment
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.energies.data_energies import PositionEnergy, ShapeEnergy
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.energies.prior_energies import RectangleOverlapEnergy, ShapeAlignmentEnergy, AreaPriorEnergy
from models.shape_net.mappings import ValueMapping
from utils.files import NumpyEncoder
from utils.math_utils import sigmoid


@dataclass
class EnergiesCalibration:
    detection_threshold: float
    param_dist_remap_coefs: List[float]
    param_dist_remap_intercepts: List[float]
    min_area: float
    max_area: float

    def apply_remap_param_dist(self, param_dist_maps: List[np.ndarray]):
        return apply_remap_param_dist(param_dist_maps, self.param_dist_remap_coefs, self.param_dist_remap_intercepts)


@dataclass
class LegacyEnergySetup(EnergySetup):
    calibration_params: Dict[str, Any]
    rewarding_priors: bool = True
    energy_calibration: EnergiesCalibration = None

    NAMES = [
        "PositionEnergy",
        "ShapeEnergy",
        "RectangleOverlapEnergy",
        "ShapeAlignmentEnergy",
        "AreaPriorEnergy"
    ]

    @property
    def energy_names(self) -> List[str]:
        return self.NAMES.copy()

    def make_energies(self, image_data: ImageWMaps) -> Tuple[List[UnitEnergyConstructor], List[PairEnergyConstructor]]:
        detection_energy = PositionEnergy(
            name=self.NAMES[0],
            detection_map=image_data.detection_map,
            threshold=self.energy_calibration.detection_threshold
        )
        # --- build shape energy (use interpolation)
        shape_energy = ShapeEnergy(
            name=self.NAMES[1],
            parameter_energy_map=self.energy_calibration.apply_remap_param_dist(image_data.param_dist_maps),
            mappings=image_data.mappings,
            param_names=image_data.param_names
        )
        # --- build priors
        overlap_energy = RectangleOverlapEnergy(
            name=self.NAMES[2],
            max_dist=32  # todo calibrate ?
        )

        angle_energy = ShapeAlignmentEnergy(
            name=self.NAMES[3],
            max_dist=16,  # todo calibrate ?
            rewarding=self.rewarding_priors
        )

        min_area_energy = AreaPriorEnergy(
            name=self.NAMES[4],
            min_area=self.energy_calibration.min_area,
            max_area=self.energy_calibration.max_area
        )

        unit_energies = [detection_energy, shape_energy, min_area_energy]
        pair_energies = [overlap_energy, angle_energy]

        return unit_energies, pair_energies

    def calibrate(self, image_configs: List[ImageWMaps], rng: Generator, save_path: str = None):
        detection_threshold = calibrate_detection_threshold(
            detection_maps=[c.detection_map for c in image_configs],
            labels=[c.labels for c in image_configs],
            target=self.calibration_params.get("threshold_target"),
            plot_dir=save_path)

        coefs, intercepts = calibrate_param_dists(
            param_dist_maps=[c.param_dist_maps for c in image_configs],
            gt_rectangles=[c.gt_config for c in image_configs],
            mappings=image_configs[0].mappings,  # we assume all mappings are the same ?
            param_names=Rectangle.PARAMETERS,
            rng=rng,
            plot_dir=save_path)

        min_area, max_area = calibrate_min_area(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path)

        calibrate_alignment(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path,
            rewarding_priors=self.rewarding_priors)

        calibration = EnergiesCalibration(
            detection_threshold=detection_threshold,
            param_dist_remap_coefs=coefs,
            param_dist_remap_intercepts=intercepts,
            min_area=min_area,
            max_area=max_area
        )

        self.energy_calibration = calibration

        with open(os.path.join(save_path, f'calibration.json'), 'w') as f:
            json.dump(asdict(calibration), f, cls=NumpyEncoder, indent=1)

    def load_calibration(self,save_dir:str):
        with open(os.path.join(save_dir, f'calibration.json'), 'r') as f:
            energy_calibration_dict = json.load(f)

        self.energy_calibration = EnergiesCalibration(
            detection_threshold=energy_calibration_dict['detection_threshold'],
            param_dist_remap_coefs=energy_calibration_dict['param_dist_remap_coefs'],
            param_dist_remap_intercepts=energy_calibration_dict['param_dist_remap_intercepts'],
            max_area=energy_calibration_dict['max_area'],
            min_area=energy_calibration_dict['min_area'],
        )

    @property
    def detection_threshold(self) -> float:
        return self.energy_calibration.detection_threshold


def apply_remap_param_dist(param_dist_maps: List[np.ndarray], coefs: List[float], intercepts: List[float]):
    remapped_list = []
    for i, (coef, intercept) in enumerate(zip(coefs, intercepts)):
        remapped = -2 * sigmoid(param_dist_maps[i] * coef + intercept) + 1
        remapped_list.append(remapped)
    return remapped_list


def f_beta(p, r, beta):
    div = ((beta ** 2 * p) + r)
    return (1 + beta ** 2) * p * r / div if div > 0 else 0


def generate_wrong_value(gt_class_value: int, mapping: ValueMapping, min_offset: int, rng: np.random.Generator):
    possible_classes = set(range(0, mapping.n_classes))
    possible_classes = possible_classes - {gt_class_value}
    for v in range(1, min_offset):
        for o in [v, -v]:
            c = gt_class_value + o
            if mapping.is_cyclic:
                c = c % mapping.n_classes
            if 0 <= 0 < mapping.n_classes:
                possible_classes -= {c}
            else:
                pass
    wrong_value_class = rng.choice(list(possible_classes))
    return wrong_value_class
