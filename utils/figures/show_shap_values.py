import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from matplotlib import patches
from torch.nn import Module
from tqdm import tqdm

from base.shapes.rectangle import show_rectangles, Rectangle
from models.mpp.custom_types import ImageWMaps
from models.mpp.data_loaders import labels_to_rectangles
from models.mpp.energies.combination.mlp import MLPEnergyCombinator
from models.mpp.energies.energy_utils import ENERGY_NAMES
from models.mpp.energies.energy_setups.energy_setup_legacy import make_energies
from models.mpp.mpp_model import MPPModel
from models.mpp.point_set.energy_point_set import EPointsSet
from utils.data import get_model_config_by_name, fetch_data_paths
from utils.display.show_img_seq import ImageStackDisplay

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
                ax.plot([p1.y, p2.y], [p1.x, p2.x], lw=1 + norm_score * (lw_max - 1), c=col, alpha=0.6)
                placed_e.append(e)


def check_delta_e():
    models = ['mpp_hrc-001b', 'mpp_mlp-001b', 'mpp_mlpRE-001b', 'mpp_mlpRE-001b']

    config_file = get_model_config_by_name(models[1])
    with open(config_file, 'r') as f:
        config = json.load(f)

    model = MPPModel(config, phase='val', overwrite=False, load=True)

    dataset = 'DOTA_gsd50_miniplus'
    subset = 'val'

    paths_dict = fetch_data_paths(dataset, subset=subset)
    image_paths = paths_dict['images']
    annot_paths = paths_dict['annotations']
    meta_paths = paths_dict['metadata']
    id_re = re.compile(r'([0-9]+).*.png')

    results = []

    for pf, af, mf in zip(tqdm(image_paths, desc=f'inferring on {dataset}/{subset}'), annot_paths,
                          meta_paths):
        patch_id = int(id_re.match(os.path.split(pf)[1]).group(1))
        img = plt.imread(pf)[..., :3]
        with open(af, 'rb') as f:
            label_dict = pickle.load(f)
        with open(mf, 'r') as f:
            meta_dict = json.load(f)

        posnet_inference_path = os.path.join(model.data.posnet_infer_path, f'{patch_id:04}_results.pkl')
        shapenet_inference_path = os.path.join(model.data.shapenet_infer_path, f'{patch_id:04}_results.pkl')

        with open(posnet_inference_path, 'rb') as f:
            detection_map = pickle.load(f)['detection_map']

        with open(shapenet_inference_path, 'rb') as f:
            shapenet_inference = pickle.load(f)

        param_dist_maps = [np.moveaxis(p[0], 0, -1) for p in shapenet_inference['output']]
        mappings = shapenet_inference['mappings']

        gt_config = labels_to_rectangles(label_dict, Rectangle.PARAMETERS)

        image_data = ImageWMaps(
            image=img,
            name=f"{patch_id:04}",
            shape=img.shape[:2],
            detection_map=detection_map,
            param_dist_maps=param_dist_maps,
            mappings=mappings,
            param_names=Rectangle.PARAMETERS,
            labels=label_dict,
            gt_config=gt_config
        )

        unit_energies, pair_energies = make_energies(image_data, energy_cal=model.energy_calibration)

        gt_points = EPointsSet(
            points=gt_config,
            support_shape=image_data.shape,
            unit_energies_constructors=unit_energies,
            pair_energies_constructors=pair_energies,
        )

        scores = [gt_points.papangelou(u, energy_combinator=model.energy_model, remove_u_from_point_set=True,
                                       return_energy_delta=False)
                  for u in gt_points]

        deltas = [gt_points.papangelou(u, energy_combinator=model.energy_model, remove_u_from_point_set=True,
                                       return_energy_delta=True)
                  for u in gt_points]

        energies = [gt_points.energy_graph.compute_subset(subset=[u], energy_combinator=model.energy_model)
                    for u in gt_points]

        energy_vectors = gt_points.energy_graph.compute_subset(subset=gt_points, energy_combinator=model.energy_model,
                                                               return_vector=True)

        energy_vectors = np.stack([energy_vectors[k] for k in ENERGY_NAMES], axis=-1)

        energy_model: MLPEnergyCombinator = model.energy_model
        data = torch.tensor(energy_vectors).float()
        e = shap.DeepExplainer(energy_model.model, data)
        shap_values = e.shap_values(data)
        # print(shap_values)

        results.append({
            'image_data': image_data,
            'scores': scores,
            'gt_points': gt_points,
            'energies': energies,
            'deltas': deltas,
            'energy_vectors': energy_vectors,
            'shap_values': shap_values,
            'shap_explanation':e
        })

        break

    shap.summary_plot(results[0]['shap_values'], results[0]['energy_vectors'], feature_names=ENERGY_NAMES)

    # [ax.axis('off') for ax in axs.ravel()]
    # ax.set_title(f"gt {image_data.name}")

    plt.show()


if __name__ == '__main__':
    check_delta_e()
