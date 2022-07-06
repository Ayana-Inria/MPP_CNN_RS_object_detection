import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, List
import numpy as np
from numpy.random import Generator

from base.shapes.rectangle import Rectangle
from models.mpp.calibration.energy_calibration import calibrate_min_area, calibrate_alignment, calibrate_param_dists, \
    apply_remap_param_dist
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.energies.data_energies import PositionEnergy, SingleMarkEnergy
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.energies.prior_energies import RectangleOverlapEnergy, ShapeAlignmentEnergy, AreaPriorEnergy, \
    RatioPriorEnergy
from utils.files import NumpyEncoder


@dataclass
class EnergiesCalibration:
    min_area: float
    max_area: float
    param_dist_remap_coefs: List[float]
    param_dist_remap_intercepts: List[float]

    def apply_remap_param_dist(self, param_dist_maps: List[np.ndarray]):
        return apply_remap_param_dist(param_dist_maps, self.param_dist_remap_coefs, self.param_dist_remap_intercepts)


class NoCalibrationEnergySetup(EnergySetup):

    def __init__(self, rewarding_priors: bool = True, ratio_prior: bool = False, calib_marks: bool = False):
        self.energy_calibration: EnergiesCalibration = None
        self.rewarding_priors = rewarding_priors
        self.ratio_prior = ratio_prior
        self.calib_marks = calib_marks

        self.NAMES = [
            "PositionEnergy",
            "SizeEnergy",
            "RatioEnergy",
            "AngleEnergy",
            "OverlapPriorEnergy",
            "AlignmentPriorEnergy",
            "AreaPriorEnergy"
        ]

        if self.ratio_prior:
            self.NAMES.append('RatioPriorEnergy')

    @property
    def energy_names(self) -> List[str]:
        return self.NAMES.copy()

    def make_energies(self, image_data: ImageWMaps
                      ) -> Tuple[List[UnitEnergyConstructor], List[PairEnergyConstructor]]:
        # todo use calibration optionally

        detection_energy = PositionEnergy(
            name=self.NAMES[0],
            detection_map=image_data.detection_map,
            threshold=0
        )

        mark_energies = []

        if self.calib_marks:
            param_e_maps = self.energy_calibration.apply_remap_param_dist(image_data.param_dist_maps)
        else:
            param_e_maps = [-m for m in image_data.param_dist_maps]

        for i, p_name in enumerate(['size', 'ratio', 'angle']):
            mark_energies.append(SingleMarkEnergy(
                name=self.NAMES[i + 1],
                parameter_energy_map=param_e_maps[i],
                mapping=image_data.mappings[i],
                param_name=p_name
            ))

        size_energy, ratio_energy, angle_energy = mark_energies

        overlap_energy = RectangleOverlapEnergy(
            name=self.NAMES[4],
            max_dist=32  # todo calibrate ?
        )

        align_energy = ShapeAlignmentEnergy(
            name=self.NAMES[5],
            max_dist=16,  # todo calibrate ?
            rewarding=self.rewarding_priors
        )

        min_area_energy = AreaPriorEnergy(
            name=self.NAMES[6],
            min_area=self.energy_calibration.min_area,
            max_area=self.energy_calibration.max_area
        )

        unit_energies = [detection_energy, size_energy, ratio_energy, angle_energy, min_area_energy]
        pair_energies = [overlap_energy, align_energy]

        if self.ratio_prior:
            ratio_energy = RatioPriorEnergy(
                name=self.NAMES[7],
                target_ratio=0.5
            )
            unit_energies.append(ratio_energy)

        return unit_energies, pair_energies

    def calibrate(self, image_configs: List[ImageWMaps], rng: Generator, save_path: str = None):
        min_area, max_area = calibrate_min_area(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path)

        calibrate_alignment(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path,
            rewarding_priors=self.rewarding_priors)

        if self.calib_marks:
            coefs, intercepts = calibrate_param_dists(
                param_dist_maps=[c.param_dist_maps for c in image_configs],
                gt_rectangles=[c.gt_config for c in image_configs],
                mappings=image_configs[0].mappings,  # we assume all mappings are the same ?
                param_names=Rectangle.PARAMETERS,
                rng=rng,
                plot_dir=save_path)
            print(f"marks calibration :\n{coefs}\n{intercepts}")
        else:
            coefs, intercepts = None, None

        calibration = EnergiesCalibration(
            min_area=min_area,
            max_area=max_area,
            param_dist_remap_coefs=coefs,
            param_dist_remap_intercepts=intercepts,
        )

        self.energy_calibration = calibration

        with open(os.path.join(save_path, f'calibration.json'), 'w') as f:
            json.dump(asdict(calibration), f, cls=NumpyEncoder, indent=1)

    def load_calibration(self, save_dir: str):
        with open(os.path.join(save_dir, f'calibration.json'), 'r') as f:
            energy_calibration_dict = json.load(f)

        self.energy_calibration = EnergiesCalibration(
            max_area=energy_calibration_dict['max_area'],
            min_area=energy_calibration_dict['min_area'],
            param_dist_remap_coefs=energy_calibration_dict.get('param_dist_remap_coefs'),
            param_dist_remap_intercepts=energy_calibration_dict.get('param_dist_remap_intercepts'),
        )

    @property
    def detection_threshold(self) -> float:
        return 0.5
