import json
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, cm
from matplotlib.colors import Normalize
from tqdm import tqdm

from base.shapes.rectangle import show_rectangles
from models.mpp.data_loaders import load_image_w_maps, crop_image_w_maps
from models.mpp.energies.combination.hierarchical import HierarchicalEnergyCombinator
from models.mpp.energies.energy_setups.energy_setup_contrast import ContrastMeasureEnergySetup
from models.mpp.mpp_model import MPPModel
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.sample_rjmcmc import sample_rjmcmc
from utils.data import get_model_config_by_name, fetch_data_paths, get_config_from_model_configs
from utils.math_utils import normalize

PLOTS = []


def show_interactions(ax: plt.Axes, points: EPointsSet, key: str, min_value: float, max_value: float, fill=False,
                      cmap='plasma', lw_max: float = 4, **polykwargs):
    colors = plt.get_cmap(cmap)
    placed_e = []
    for p in points:

        poly = p.poly_coord[:, [1, 0]]
        patch = patches.Polygon(poly, fill=fill, **polykwargs)
        ax.add_patch(patch)
        for e in points.energy_graph.pe_per_point[p]:
            en_name = e.constructor.__class__.__name__
            if en_name == key and e not in placed_e:
                score = np.clip((e.value - min_value) / (max_value - min_value), 0, 1)
                norm_score = np.clip(e.value / max(-min_value, max_value), 0, 1)
                col = colors(score)
                p1 = e.point_1
                p2 = e.point_2
                ax.plot([p1.y, p2.y], [p1.x, p2.x], lw=1 + norm_score * (lw_max - 1), c=col)
                placed_e.append(e)


def display_method(idx, data, energy_names: List[str]):
    n_rows, n_cols = 3, max(4, len(energy_names))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharey='all', sharex='all')

    res = data[idx]

    image_data = res['image_data']
    scores = res['scores']
    scores_gt = res['scores_gt']
    gt_points = res['gt_points']
    points = res['points']
    energies = res['energies']
    deltas = res['deltas']
    energy_vectors = res['energy_vectors']

    [ax.imshow(image_data.image) for ax in axs.ravel()]

    # max_score = 1.0 if len(scores) == 0 else np.max(np.abs(scores))
    max_score = 1.0 if len(scores) == 0 else np.quantile(np.abs(scores), 0.9)
    axs[0, 0].set_title("papangelou intensity")
    show_rectangles(axs[0, 0], points, cmap='viridis', scores=scores, min_score=0, max_score=max_score)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_score), cmap='viridis'), ax=axs[0, 0])

    max_score = 1.0 if len(scores_gt) == 0 else np.quantile(np.abs(scores_gt), 0.9)
    axs[0, 3].set_title("papangelou intensity GT")
    show_rectangles(axs[0, 3], gt_points, cmap='viridis', scores=scores_gt, min_score=0, max_score=max_score)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_score), cmap='viridis'), ax=axs[0, 3])

    axs[0, 1].set_title("energy delta on addition")
    max_score = 1.0 if len(scores) == 0 else np.max(np.abs(deltas))
    show_rectangles(axs[0, 1], points, cmap='coolwarm', scores=deltas, min_score=-max_score, max_score=max_score)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-max_score, vmax=max_score), cmap='coolwarm'), ax=axs[0, 1])

    axs[0, 2].set_title("energy per point")
    max_score = 1.0 if len(scores) == 0 else np.max(np.abs(energies))
    show_rectangles(axs[0, 2], points, cmap='coolwarm', scores=energies, min_score=-max_score,
                    max_score=max_score)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-max_score, vmax=max_score), cmap='coolwarm'), ax=axs[0, 2])

    [ax.clear() for ax in axs[0, 4:]]

    for i in range(min(n_cols, len(energy_names))):
        sub_e = energy_vectors[:, i]
        max_value = 1.0 if len(energy_vectors) == 0 else np.max(np.abs(sub_e))
        if max_value == 0.0:
            max_value = 1.0
        axs[1, i].set_title(energy_names[i])
        show_rectangles(axs[1, i], points, cmap='coolwarm', scores=sub_e, min_score=-max_value,
                        max_score=max_value)
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-max_value, vmax=max_value), cmap='coolwarm'),
                     ax=axs[1, i])

    max_value = 1.0 if len(energy_vectors) == 0 else np.max(np.abs(energy_vectors[:, 2]))
    show_interactions(axs[2, 2], points, key='RectangleOverlapEnergy', min_value=-max_value, max_value=max_value,
                      cmap='coolwarm')

    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-max_value, vmax=max_value), cmap='coolwarm'), ax=axs[2, 2])

    axs[2, 2].set_title("RectangleOverlapEnergy interactions")
    max_value = 1.0 if len(energy_vectors) == 0 else np.max(np.abs(energy_vectors[:, 3]))
    show_interactions(axs[2, 3], points, key='ShapeAlignmentEnergy', min_value=-max_value, max_value=max_value,
                      cmap='coolwarm')

    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-max_value, vmax=max_value), cmap='coolwarm'), ax=axs[2, 3])

    axs[2, 3].set_title("ShapeAlignmentEnergy interactions")

    [ax.axis('off') for ax in axs.ravel()]
    fig.show()


def check_delta_e():
    # models = ['mpp_hrc-001b', 'mpp_mlp-001b', 'mpp_mlpRE-001b', 'mpp_mlpRE-001b']

    INFER = True

    model = 'config_mpp_log-1004b.json'

    # config_file = get_model_config_by_name(model)
    config_file = get_config_from_model_configs(model)
    with open(config_file, 'r') as f:
        config = json.load(f)

    rng = np.random.default_rng(0)
    model = MPPModel(config, phase='train', overwrite=True, load=False, dataset="DOTA_gsd50_miniplus")
    model.train()

    dataset = 'DOTA_gsd50_miniplus'
    subset = 'val'
    patch_size = 256

    energy_model = model.energy_model

    patch_ids = [683]

    results = []

    for patch_id in patch_ids:

        image_data = load_image_w_maps(patch_id, dataset, subset, position_model='posvec_dota_24',
                                       shape_model='shape_dota_22')

        # center_anchor = rng.choice(image_data.labels['centers'], axis=0)
        gt_centers = np.array([r.get_coord() for r in image_data.gt_config])
        center_of_interest = np.median(gt_centers, axis=0)
        anchor = (center_of_interest - patch_size / 2).astype(int)
        h, w = image_data.image.shape[:2]
        anchor = np.clip(anchor, (0, 0), (h - patch_size - 1, w - patch_size - 1))

        image_data = crop_image_w_maps(image_data=image_data, tl_anchor=anchor, patch_size=patch_size)

        unit_energies, pair_energies = model.energy_setup.make_energies(image_data)

        gt_points = EPointsSet(
            points=image_data.gt_config,
            support_shape=image_data.shape,
            unit_energies_constructors=unit_energies,
            pair_energies_constructors=pair_energies,
        )
        if INFER:
            target_temperature = 1e-4
            init_temperature = 10
            n_iter = 10000
            alpha = np.power(target_temperature / init_temperature, 1 / n_iter)
            print(f"{alpha=}")
            infer_points = sample_rjmcmc(
                image_data=image_data, rng=rng, num_samples=1, energy_setup=model.energy_setup,
                energy_combinator=energy_model, init_config=None, init_temperature=init_temperature, alpha_t=alpha,
                burn_in=n_iter,
                samples_interval=1, target_temperature=0.0, verbose=1
            )

            infer_points = EPointsSet(
                points=infer_points[-1],
                support_shape=image_data.shape,
                unit_energies_constructors=unit_energies,
                pair_energies_constructors=pair_energies,
            )

            points = infer_points
        else:
            points = gt_points

        scores = [points.papangelou(u, energy_combinator=energy_model, remove_u_from_point_set=True,
                                    return_energy_delta=False)
                  for u in points]

        if INFER:
            scores_gt = [points.papangelou(u, energy_combinator=energy_model, remove_u_from_point_set=False,
                                           return_energy_delta=False)
                         for u in gt_points]
        else:
            scores_gt = scores

        deltas = [points.papangelou(u, energy_combinator=energy_model, remove_u_from_point_set=True,
                                    return_energy_delta=True)
                  for u in points]

        energies = [points.energy_graph.compute_subset(subset=[u], energy_combinator=energy_model)
                    for u in points]

        energy_vectors = points.energy_graph.compute_subset(subset=points, energy_combinator=energy_model,
                                                            return_vector=True)
        energy_vectors = np.stack([energy_vectors[k] for k in model.energy_setup.energy_names], axis=-1)

        results.append({
            'image_data': image_data,
            'scores': scores,
            'scores_gt': scores_gt,
            'gt_points': gt_points,
            'points': points,
            'energies': energies,
            'deltas': deltas,
            'energy_vectors': energy_vectors,
        })

        break

    # [ax.axis('off') for ax in axs.ravel()]
    # ax.set_title(f"gt {image_data.name}")

    # stk = ImageStackDisplay(
    #     axs=axes, display_method=display_method, plot_data_list=results
    # )
    # fig.canvas.mpl_connect('key_press_event', stk.key)
    # PLOTS.append(stk)
    print(model.energy_setup.energy_names)
    display_method(0, results, model.energy_setup.energy_names)
    plt.show()


if __name__ == '__main__':
    check_delta_e()
