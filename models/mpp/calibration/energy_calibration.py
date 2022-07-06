import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator
from scipy.ndimage import binary_dilation
from sklearn.linear_model import LogisticRegression

from base.shapes.rectangle import Rectangle
from metrics.detection import precision_recall_curve_on_detection_map
from models.mpp.energies.prior_energies import AreaPriorEnergy, ShapeAlignmentEnergy
from models.shape_net.mappings import ValueMapping
from utils.math_utils import sigmoid, f_beta

CAL_COLOR = 'tab:red'


def calibrate_detection_threshold(detection_maps: List[np.ndarray], labels: List[Dict],
                                  target: str = 'f1', plot_dir: str = None):
    thresh, metrics = precision_recall_curve_on_detection_map(
        detection_map=detection_maps, labels=labels, num_thresholds=100, dilation=2)
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
        for t in ['f1','f2','f0.5']:
            axs[1].plot(thresh, metrics[t],label=t)
        axs[1].set_xlabel('threshold')
        axs[1].set_ylabel(target)
        axs[1].vlines(thresh[argmax], 0, 1, colors=CAL_COLOR)
        axs[1].set_title(f"max {target} at {thresh[argmax]:.2f}")
        axs[1].legend()

        pos_values = []
        neg_values = []
        for d, l in zip(detection_maps, labels):
            label_map = np.zeros(d.shape[:2], dtype=bool)
            centers = l['centers']
            if len(centers) > 0:
                label_map[centers[:, 0], centers[:, 1]] = True
                label_map = binary_dilation(label_map, iterations=2)
                pos_values.append(d[label_map].ravel())
                neg_values.append(d[~label_map].ravel())

        pos_values = np.concatenate(pos_values)
        neg_values = np.concatenate(neg_values)
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
        ax2.plot(xx, -1 * (xx - thresh[argmax]), color=CAL_COLOR)
        ax2.set_ylabel('energy', color=CAL_COLOR)

        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'calibrate_detection_threshold.png'))
        plt.close('all')

    return thresh[argmax]


def calibrate_param_dists(param_dist_maps: List[List[np.ndarray]], gt_rectangles: List[List[Rectangle]],
                          mappings: List[ValueMapping], param_names: List[str], rng: np.random.Generator,
                          plot_dir: str = None):
    nv_samples_per_valid = 1
    intercepts = []
    coefs = []
    all_sm_values = []
    all_labels = []
    for i_p, (mapping, p_name) in enumerate(zip(mappings, param_names)):
        sm_values = []
        labels = []

        for k in range(len(param_dist_maps)):

            for i, gt_point in enumerate(gt_rectangles[k]):
                gt_value = gt_point.__getattribute__(p_name)
                local_dist = param_dist_maps[k][i_p][gt_point.x, gt_point.y]
                gt_class_value = mapping.value_to_class(gt_value)
                sm_gt_value = local_dist[gt_class_value]
                sm_values.append(sm_gt_value)
                labels.append(1)

                for _ in range(nv_samples_per_valid):
                    wrong_value_class = generate_wrong_value(gt_class_value, mapping, 2, rng)
                    wrong_value_sm = local_dist[wrong_value_class]
                    sm_values.append(wrong_value_sm)
                    labels.append(0)
        sm_values = np.array(sm_values)
        labels = np.array(labels)
        clf = LogisticRegression(penalty='none', class_weight='balanced').fit(sm_values.reshape((-1, 1)), labels)
        all_sm_values.append(sm_values)
        all_labels.append(labels)
        coefs.append(clf.coef_[0, 0])
        intercepts.append(clf.intercept_[0])

    if plot_dir is not None:
        n_param = len(mappings)
        plt.ioff()
        fig, axs = plt.subplots(1, n_param, figsize=(3 * n_param, 3))
        for i_p in range(n_param):
            x_0 = all_sm_values[i_p][all_labels[i_p] == 0]
            x_1 = all_sm_values[i_p][all_labels[i_p] == 1]
            max_value = max(np.max(x_0), np.max(x_1))
            bins = np.linspace(0, max_value, 20)
            axs[i_p].hist(x_0, bins=bins, alpha=0.5, label=f'non valid {param_names[i_p]}', density=True)
            axs[i_p].hist(x_1, bins=bins, alpha=0.5, label=f'valid {param_names[i_p]}', density=True)
            axs[i_p].legend()
            axs[i_p].set_xlabel('softmax values')
            axs[i_p].set_ylabel('occurences')

            ax2 = axs[i_p].twinx()
            xx = np.linspace(0, max_value, 100)
            ax2.plot(xx, -2 * sigmoid(xx * coefs[i_p] + intercepts[i_p]) + 1, color=CAL_COLOR)
            ax2.set_ylabel('energy', color=CAL_COLOR)
        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'calibrate_shape_reparam.png'))
        plt.close('all')

    return coefs, intercepts


def apply_remap_param_dist(param_dist_maps: List[np.ndarray], coefs: List[float], intercepts: List[float]):
    remapped_list = []
    for i, (coef, intercept) in enumerate(zip(coefs, intercepts)):
        remapped = -2 * sigmoid(param_dist_maps[i] * coef + intercept) + 1
        remapped_list.append(remapped)
    return remapped_list


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


def calibrate_min_area(gt_configs: List[List[Rectangle]], quantile: float = 0.01, plot_dir: str = None):
    areas = []
    for conf in gt_configs:
        areas.append([p.poly.area for p in conf])
    areas = np.concatenate(areas, axis=0)
    min_area = np.quantile(areas, quantile)
    max_area = np.quantile(areas, 1 - quantile)

    if plot_dir is not None:
        fig, axs = plt.subplots(1, 1, figsize=(3, 3))
        axs.hist(areas, bins=20, density=True)
        ax2 = axs.twinx()
        max_v = np.max(areas)
        xx = np.linspace(0, max_v, 100)
        ax2.plot(xx, AreaPriorEnergy.response_function(xx, min_area, max_area), color=CAL_COLOR)
        axs.set_title('area values distribution')
        axs.set_xlabel('area')
        axs.set_ylabel('occurences')
        ax2.set_ylabel('energy', color=CAL_COLOR)
        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'calibrate_min_area.png'))
        plt.close('all')

    return min_area, max_area


def calibrate_alignment(gt_configs: List[List[Rectangle]], rewarding_priors: bool, max_dist: int = 16,
                        plot_dir: str = None):
    angles = []
    for conf in gt_configs:
        for u in conf:
            angles.append([v.angle - u.angle
                           for v in conf if (v is not u and np.linalg.norm(u.get_coord() - v.get_coord()) <= max_dist)])
    angles = np.concatenate(angles, axis=0)

    if plot_dir is not None:
        fig, axs = plt.subplots(1, 1, figsize=(3, 3))
        axs.hist(angles, bins=20, density=True)
        ax2 = axs.twinx()
        min_v = min(np.min(angles), -np.pi / 2)
        max_v = max(np.max(angles), np.pi / 2)
        xx = np.linspace(min_v, max_v, 100)
        yy = ShapeAlignmentEnergy.response_function(xx, rewarding=rewarding_priors)
        ax2.plot(xx, yy, color=CAL_COLOR)
        axs.set_title('angle values distribution')
        axs.set_xlabel('angle delta')
        axs.set_ylabel('occurences')
        ax2.set_ylabel('energy', color=CAL_COLOR)
        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'calibrate_alignment.png'))
        plt.close('all')
