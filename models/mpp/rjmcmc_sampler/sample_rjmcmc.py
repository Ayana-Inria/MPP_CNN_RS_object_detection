import logging
import time
from typing import List, Union

import numpy as np
from numpy.random import Generator

from base.shapes.rectangle import Rectangle
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.custom_types.energy import EnergyCombinationModel
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.kernels.make_kernels import make_kernels
from models.mpp.rjmcmc_sampler.rjmcmc import RJMCMC
from models.mpp.rjmcmc_sampler.stopping import StopOnMaxIter
from models.shape_net.mappings import output_vector_to_value
from utils.nms import nms_distance

BURN_IN = 3000
MAX_ITER = 10000


def naive_detection(image_data: ImageWMaps, detection_threshold: float):
    detection_map = image_data.detection_map
    detections_centers = np.array(np.where(detection_map >= detection_threshold)).T
    detections_scores = detection_map[detections_centers[:, 0], detections_centers[:, 1]]
    nms_centers, nms_scores = nms_distance(detections_centers, detections_scores, threshold=6)
    dist_maps = [np.expand_dims(np.moveaxis(d, -1, 0), 0) for d in image_data.param_dist_maps]
    values_map = output_vector_to_value(dist_maps, image_data.mappings)
    pred_params = [
        [values_map[0][0][c[0], c[1]], values_map[1][0][c[0], c[1]], values_map[2][0][c[0], c[1]]]
        for c in
        nms_centers]

    return [Rectangle(x=c[0], y=c[1], size=p[0], ratio=p[1], angle=p[2]) for c, p in zip(nms_centers, pred_params)]


def sample_rjmcmc(image_data: ImageWMaps, rng: Generator, num_samples: int,
                  energy_combinator: EnergyCombinationModel, init_config: Union[str, List[Rectangle]],
                  init_temperature: float,
                  alpha_t: Union[float,str], burn_in: int, energy_setup: EnergySetup,
                  samples_interval: int, target_temperature: float, verbose: int = 0, iter_multiplier: float = None,
                  use_split_merge: bool = False
                  ):
    unit_energies, pair_energies = energy_setup.make_energies(image_data)

    # ---- init points
    if init_config == 'gt':
        init_config = image_data.gt_config
    elif init_config is None:
        init_config = []
    elif init_config == 'naive':
        detection_threshold = energy_setup.detection_threshold
        init_config = naive_detection(
            image_data=image_data,
            detection_threshold=detection_threshold)

    if iter_multiplier is not None:
        burn_in = burn_in * iter_multiplier
        samples_interval = samples_interval * iter_multiplier
        alpha_t = np.power(alpha_t, 1 / iter_multiplier)

    if alpha_t == 'auto':
        alpha_t = np.power(target_temperature / init_temperature, 1 / burn_in)
        target_temperature = 0
        print(f"auto alpha: {alpha_t}")

    intensity = max(1, len(init_config))
    kernels, p_kernels = make_kernels(image_data, intensity=intensity, rng=rng, use_split_merge=use_split_merge)

    points = EPointsSet(
        points=init_config,
        support_shape=image_data.shape,
        unit_energies_constructors=unit_energies,
        pair_energies_constructors=pair_energies,
    )
    # --- simulate RJMCMC
    max_iter = burn_in + (num_samples + 1) * samples_interval
    start = time.perf_counter()
    rjmcmc = RJMCMC(
        t0=init_temperature,
        t_target=target_temperature,
        alpha_t=alpha_t,
        kernels=kernels,
        p_kernels=p_kernels,
        initial_state=points.copy(),
        energy_combinator=energy_combinator,
        # stopping_condition=CustomStopping(max_iter=MAX_ITER, min_iter=BURN_IN+num),
        stopping_condition=StopOnMaxIter(max_iter),
        rng=rng,
        sampling_rule=lambda step: step >= burn_in and step % samples_interval == 0,
        verbose=verbose,
    )
    last_state, summaries = rjmcmc.run()
    end = time.perf_counter()
    logging.info(f'rjmcmc on image {image_data.name} ran in {end - start:.2f}s ({(end - start) / max_iter:.1e}s/iter) '
                 f'(int. {intensity} | iter {max_iter} | num_samples {num_samples})')

    if num_samples == 1:
        return [last_state[-1].points]
    else:
        return [s.points for s in last_state[-num_samples:]]
