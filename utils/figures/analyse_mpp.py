import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
from base.shapes.rectangle import show_rectangles
from matplotlib import cm
from matplotlib.colors import Normalize
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.data_loaders import load_image_w_maps
from models.mpp.energies.energy_setups.energy_setup_legacy import make_energies
from models.mpp.mpp_model import MPPModel
from models.mpp.point_set.energy_point_set import EPointsSet
from utils.data import get_model_config_by_name, get_inference_path
import numpy as np


def show_papangelou(points: EPointsSet, image_data: ImageWMaps, model: MPPModel, ax: plt.Axes, fig: plt.Figure,
                    colorbar=True, energy_delta=False):
    scores = [points.papangelou(u, energy_combinator=model.energy_model, remove_u_from_point_set=True,
                                return_energy_delta=energy_delta)
              for u in points]

    max_score = 1.0 if len(scores) == 0 else np.quantile(np.abs(scores), 0.9)
    ax.set_title("papangelou intensity")
    ax.imshow(image_data.image)
    if energy_delta:
        vmin, vmax = -max_score, max_score
        cmap = 'coolwarm'
    else:
        vmin, vmax = 0, max_score
        cmap = 'viridis'
    show_rectangles(ax, points.points, cmap=cmap, scores=scores, min_score=vmin, max_score=vmax)

    if colorbar:
        fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)


def show():
    dataset = 'DOTA_gsd50_miniplus'
    subset = 'val'

    patch_id = '0683'
    models_list = [
        ('mpp_log-200', 'log (OC)'),
        ('mpp_lin-200', 'lin (OC)'),
        ('mpp_mlp-200', 'mlp (OC)'),
        ('mpp_hrcM-000', 'hrc (M)'),
        ('mpp_hrc-200', 'hrc (OC)')
    ]

    plt.ion()
    n_rows, n_cols = 1, len(models_list)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False, sharey='all', sharex='all')

    for i, (model_name, model_label) in enumerate(models_list):
        config_file = get_model_config_by_name(model_name)
        with open(config_file, 'r') as f:
            config = json.load(f)

        rng = np.random.default_rng(0)
        try:
            model = MPPModel(config, phase='val', overwrite=False, load=True)
        except ModuleNotFoundError as e:
            logging.error(f"failed to load deprecated model at {config_file}")
            raise e

        image_data = load_image_w_maps(patch_id, dataset, subset, position_model='posvec_dota_24',
                                       shape_model='shape_dota_22')

        unit_energies, pair_energies = make_energies(image_data, energy_cal=model.energy_calibration)

        inf_file = os.path.join(
            '/workspaces/nef/data/ayana/share/Airbus/datasets_jm/inference/',
            dataset, subset, model_name,
            f"{patch_id}_results.pkl"
        )

        with open(inf_file, 'rb') as f:
            detection_dict = pickle.load(f)

        infer_points = EPointsSet(
            points=detection_dict['detection_points'],
            support_shape=image_data.shape,
            unit_energies_constructors=unit_energies,
            pair_energies_constructors=pair_energies,
        )

        show_papangelou(infer_points, image_data, model, axs[0, i], fig, energy_delta=True)
        axs[0, i].set_title(f"energy delta {model_label}")
        
    fig.show()


if __name__ == '__main__':
    show()
