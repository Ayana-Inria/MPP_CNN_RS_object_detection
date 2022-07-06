import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Union

import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
from base.shapes.rectangle import Rectangle
from metrics.detection import compute_precision_recall
from models.mpp.calibration.energy_calibration import calibrate_min_area, calibrate_alignment
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.energies.classics import ContrastEnergy, GradientEnergy
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.energies.prior_energies import RectangleOverlapEnergy, ShapeAlignmentEnergy, AreaPriorEnergy, \
    RatioPriorEnergy
from utils.files import NumpyEncoder
from utils.math_utils import f_beta


@dataclass
class EnergiesCalibration:
    detection_thresh: float
    min_area: float
    max_area: float


class ContrastMeasureEnergySetup(EnergySetup):
    NAMES = [
        "ContrastEnergy",
        "OverlapPriorEnergy",
        "AlignmentPriorEnergy",
        "AreaPriorEnergy",
        "RatioPriorEnergy"
    ]

    def __init__(self, contrast_type: str, learn_threshold: bool = False, rewarding_priors: bool = True,
                 manual_threshold=None):
        self.energy_cal = None
        self.contrast_type = contrast_type
        self.rewarding_priors: bool = rewarding_priors
        self.learn_threshold = learn_threshold
        self.manual_threshold = manual_threshold

    @property
    def energy_names(self) -> List[str]:
        return self.NAMES.copy()

    def _make_contrast_energy(self, image_data, detection_thresh: float):
        thresh = detection_thresh if detection_thresh is not None else 0.0
        print(f"ContrastEnergy using thresh {thresh}")
        if self.contrast_type == 'gradient':
            detection_energy = GradientEnergy(
                name=self.NAMES[0],
                image=image_data.image,
                dilation=1,
                rgb=True,
                thresh=thresh
            )
        else:

            noisy_image = image_data.image + np.random.normal(0, 0.05, size=image_data.image.shape)
            noisy_image = np.clip(noisy_image, 0, 1)

            detection_energy = ContrastEnergy(
                name=self.NAMES[0],
                image=image_data.image if self.contrast_type != 't-test' else noisy_image,
                dilation=2,
                gap=1 if self.contrast_type != 'craciun' else 0,
                erode=1 if self.contrast_type != 'craciun' else 0,
                contrast_measure_type=self.contrast_type,
                rgb=self.contrast_type != 't-test',
                thresh=thresh,
                normalize=self.contrast_type == 't-test'

            )
        return detection_energy

    def make_energies(self, image_data: ImageWMaps) -> Tuple[List[UnitEnergyConstructor], List[PairEnergyConstructor]]:

        detection_energy = self._make_contrast_energy(image_data, detection_thresh=self.energy_cal.detection_thresh)

        overlap_energy = RectangleOverlapEnergy(
            name=self.NAMES[1],
            max_dist=32  # todo calibrate ?
        )

        align_energy = ShapeAlignmentEnergy(
            name=self.NAMES[2],
            max_dist=16,  # todo calibrate ?
            rewarding=self.rewarding_priors
        )

        min_area_energy = AreaPriorEnergy(
            name=self.NAMES[3],
            min_area=self.energy_cal.min_area,
            max_area=self.energy_cal.max_area
        )

        ratio_energy = RatioPriorEnergy(
            name=self.NAMES[4],
            target_ratio=0.5
        )

        unit_energies = [detection_energy, min_area_energy, ratio_energy]
        pair_energies = [overlap_energy, align_energy]

        return unit_energies, pair_energies

    def calibrate(self, image_configs: List[ImageWMaps], rng: Generator, save_path: str = None):
        detection_threshold = None
        if self.learn_threshold:
            detection_threshold = calibrate_detection_threshold(
                contrast_energy_maker=self._make_contrast_energy,
                image_configs=image_configs,
                rng=rng,
                plot_dir=save_path
            )
        elif self.manual_threshold is not None:
            detection_threshold = self.manual_threshold

        min_area, max_area = calibrate_min_area(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path)

        calibrate_alignment(
            gt_configs=[c.gt_config for c in image_configs],
            plot_dir=save_path,
            rewarding_priors=self.rewarding_priors)

        calibration = EnergiesCalibration(
            min_area=min_area,
            max_area=max_area,
            detection_thresh=detection_threshold
        )

        self.energy_cal = calibration

        with open(os.path.join(save_path, f'calibration.json'), 'w') as f:
            json.dump(asdict(calibration), f, cls=NumpyEncoder, indent=1)

    def load_calibration(self, save_dir: str):
        with open(os.path.join(save_dir, f'calibration.json'), 'r') as f:
            energy_calibration_dict = json.load(f)

        self.energy_cal = EnergiesCalibration(
            detection_thresh=energy_calibration_dict['detection_thresh'],
            max_area=energy_calibration_dict['max_area'],
            min_area=energy_calibration_dict['min_area'],
        )

    @property
    def detection_threshold(self) -> float:
        return 0.5


CAL_COLOR = 'tab:red'


def calibrate_detection_threshold(contrast_energy_maker, image_configs: List[ImageWMaps], rng: Generator,
                                  target: str = 'f1', plot_dir: str = None):
    x = []
    y = []

    for image_data in image_configs:
        contrast_energy: Union[ContrastEnergy, GradientEnergy] = contrast_energy_maker(image_data, detection_thresh=0.0)

        gt_values = [-contrast_energy.compute(u) for u in image_data.gt_config]
        n_false = 4 * len(gt_values)
        rd_values = []
        for _ in range(n_false):
            rd_point = Rectangle(
                x=rng.integers(0, image_data.image.shape[0]),
                y=rng.integers(0, image_data.image.shape[1]),
                size=rng.normal(8, 1.0),
                ratio=float(np.clip(rng.normal(0.5, 0.1), 0.1, 1)),
                angle=rng.uniform(0, np.pi)
            )

            rd_values.append(
                -contrast_energy.compute(rd_point)
            )
        x.append(gt_values + rd_values)
        y.append([True] * len(gt_values) + [False] * len(rd_values))

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0).astype(bool)

    thresholds = np.linspace(np.min(x), np.max(x), 100)

    precision, recall = compute_precision_recall(x, y, thresholds)
    precision = np.array(precision)
    recall = np.array(recall)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': (precision * recall) / (precision + recall)
    }

    metrics['f1'] = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(metrics['precision'], metrics['recall'])]
    metrics['f2'] = [f_beta(p, r, 2.0) for p, r in zip(metrics['precision'], metrics['recall'])]
    metrics['f0.5'] = [f_beta(p, r, 0.5) for p, r in zip(metrics['precision'], metrics['recall'])]

    argmax = np.argmax(metrics[target])
    if plot_dir is not None:
        plt.ioff()
        fig, axs = plt.subplots(1, 3, figsize=(3 * 3, 3))
        axs[0].plot(metrics['recall'], metrics['precision'])
        axs[0].scatter(metrics['recall'][argmax], metrics['precision'][argmax], color=CAL_COLOR)
        axs[0].set_xlabel('recall')
        axs[0].set_ylabel('precision')
        for t in ['f1', 'f2', 'f0.5']:
            axs[1].plot(thresholds, metrics[t], label=t)
        axs[1].set_xlabel('threshold')
        axs[1].set_ylabel(target)
        axs[1].vlines(thresholds[argmax], 0, 1, colors=CAL_COLOR)
        axs[1].set_title(f"max {target} at {thresholds[argmax]:.2f}")
        axs[1].legend()

        pos_values = x[y == 1]
        neg_values = x[y == 0]
        max_value = max(np.max(pos_values), np.max(neg_values))
        min_value = min(np.min(pos_values), np.min(neg_values))
        bins = np.linspace(min_value, max_value, 20)
        axs[2].hist(neg_values, bins=bins, alpha=0.5, label=f'non valid', density=True)
        axs[2].hist(pos_values, bins=bins, alpha=0.5, label=f'valid', density=True)
        axs[2].legend()
        axs[2].set_xlabel('values')
        axs[2].set_ylabel('occurences')

        ax2 = axs[2].twinx()
        xx = np.linspace(min_value, max_value, 100)
        ax2.plot(xx, -1 * (xx - thresholds[argmax]), color=CAL_COLOR)
        ax2.set_ylabel('energy', color=CAL_COLOR)

        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'calibrate_detection_threshold.png'))
        plt.close('all')

    return -thresholds[argmax]
