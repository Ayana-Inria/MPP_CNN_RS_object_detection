from copy import copy
from typing import List, Tuple

import numpy as np

from base.shapes.rectangle import Rectangle
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.energies.energy_utils import EnergySetup
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import Kernel
from models.mpp.rjmcmc_sampler.kernels.make_kernels import make_kernels
from models.shape_net.mappings import ValueMapping

PERTURBATION_LIGHT = {
    "move_proba": 0.1,
    "param_shift_proba": [0.1, 0.1, 0.1],
    "position_sigma": 1,
    "param_sigmas": [0.02, 0.02, 0.02],
    "point_number_sigma": 0.1,
    "no_addition": True
}

PERTURBATION_MEDIUM = {
    "move_proba": 0.5,
    "param_shift_proba": [0.5, 0.5, 0.5],
    "position_sigma": 5,
    "param_sigmas": [0.1, 0.1, 0.1],
    "point_number_sigma": 1.0
}

PERTURBATION_HP_MEDIUM = {
    "move_proba": 0.8,
    "param_shift_proba": [0.9, 0.9, 0.9],
    "position_sigma": 5,
    "param_sigmas": [0.1, 0.1, 0.1],
    "point_number_sigma": 1.0
}

PERTURBATION_MEDIUM_OVERLAP = {
    "move_proba": 0.8,
    "param_shift_proba": [0.9, 0.9, 0.9],
    "position_sigma": 5,
    "param_sigmas": [0.1, 0.1, 0.1],
    "point_number_sigma": 5.0,
    "make_overlap": 0.9
}

PERTURBATION_STRONG = {
    "move_proba": 0.9,
    "param_shift_proba": [0.9, 0.9, 0.9],
    "position_sigma": 20,
    "param_sigmas": [0.5, 0.5, 0.5],
    "point_number_sigma": 10.0
}


def sample_perturbations(image_data: ImageWMaps = None, gt_rectangles: List[Rectangle] = None,
                         rng: np.random.Generator = None, image_shape: Tuple[int, int] = None,
                         mappings: List[ValueMapping] = None, move_proba: float = None,
                         param_shift_proba: List[float] = None,
                         position_sigma: float = None,
                         param_sigmas: List[float] = None,
                         make_overlap: float = None,
                         no_addition: bool = False,
                         point_number_sigma: float = None, n_samples: int = 1):
    if image_data is not None:
        gt_rectangles = image_data.gt_config
        image_shape = image_data.shape
        mappings = image_data.mappings
    else:
        assert gt_rectangles is not None
        assert image_shape is not None
        assert mappings is not None

    results = []
    for _ in range(n_samples):
        new_points = [copy(p) for p in gt_rectangles]

        # add/remove points
        points_init_nb = len(gt_rectangles)
        new_points_nb = int(np.clip(rng.normal(points_init_nb, point_number_sigma), a_min=0, a_max=1e4))
        if no_addition:
            new_points_nb = np.clip(new_points_nb, 0, points_init_nb)
        if new_points_nb < points_init_nb:
            new_points_id = rng.choice(range(points_init_nb), size=new_points_nb, replace=False)
            new_points = [new_points[i] for i in new_points_id]
        elif new_points_nb > points_init_nb:
            for _ in range(new_points_nb - points_init_nb):
                if make_overlap is not None and rng.random() <= make_overlap:
                    p = copy(rng.choice(new_points))
                    new_points.append(p)
                else:
                    pos = rng.integers((0, 0), image_shape)
                    params = {p: rng.uniform(m.v_min, m.v_max) for p, m in zip(Rectangle.PARAMETERS, mappings)}
                    p = Rectangle(
                        x=pos[0], y=pos[1],
                        **params
                    )
                    new_points.append(p)
        else:
            pass
        # generate point shifts
        for p in new_points:
            if rng.random() < move_proba:
                # move point
                shift = rng.normal(0, position_sigma, size=2)
                pos = (p.x + shift[0], p.y + shift[1])
                p.x, p.y = np.clip(pos, (0, 0), (image_shape[0] - 1, image_shape[1] - 1)).astype(int)

            for i, (mapping, param) in enumerate(zip(mappings, Rectangle.PARAMETERS)):
                if rng.random() < param_shift_proba[i]:
                    # shift param p
                    v_min, v_max = mapping.v_min, mapping.v_max
                    new_value = p.__getattribute__(param) + rng.normal(0, param_sigmas[i] * (v_max - v_min))
                    new_value = np.clip(
                        ((new_value - v_min) % (v_max - v_min)) + v_min if mappings[i].is_cyclic else new_value,
                        v_min, v_max
                    )
                    p.__setattr__(param, new_value)
        results.append(new_points)
    return results


def sample_multiple_kernel_perturbations(image_data: ImageWMaps, n_samples: int, rng: np.random.Generator,
                                         energy_setup: EnergySetup, iter_per_point: float,
                                         return_perturbations: bool = False, aggregate_pert: bool = False,
                                         use_split_merge: bool = False):
    if image_data.gt_config_set is None:
        uec, pec = energy_setup.make_energies(image_data=image_data)
        points = EPointsSet(
            points=image_data.gt_config,
            support_shape=image_data.shape,
            unit_energies_constructors=uec, pair_energies_constructors=pec
        )
    else:
        points = image_data.gt_config_set
    kernels, p_kernels = make_kernels(
        image_data, intensity=1.0, rng=rng, use_split_merge=use_split_merge
    )
    results = []
    perts = []
    for _ in range(n_samples):
        new_points, perturbations = sample_kernel_perturbations(
            kernels=kernels, p_kernels=p_kernels, points=points, rng=rng, iter_per_point=iter_per_point,
            aggregate_pert=aggregate_pert
        )
        results.append(new_points)
        perts.append(perturbations)
    if return_perturbations:
        return perts
    return results


def sample_kernel_perturbations(kernels: List[Kernel], p_kernels: List[float], iter_per_point: float,
                                points: EPointsSet, rng: np.random.Generator, aggregate_pert: bool = False):
    assert len(kernels) == len(p_kernels)
    new_points = points.copy()
    perturbations = []
    n_iter = int(iter_per_point * len(points))
    for i in range(n_iter):
        kernel: Kernel = rng.choice(kernels, p=p_kernels)
        pert = kernel.sample_perturbation(x=new_points.points, rng=rng)
        perturbations.append(pert)
        new_points = new_points.apply_perturbation(pert)

    if aggregate_pert:
        perturbations = aggregate_perturbations(perturbations)
    return new_points, perturbations


class DummyKernel:
    pass


def aggregate_perturbations(perturbations: List[Perturbation]):
    all_additions = set()
    all_removals = set()

    for p in perturbations:
        if type(p.addition) is list:
            added = p.addition
        elif p.addition is not None:
            added = [p.addition]
        else:
            added = []

        if type(p.removal) is list:
            removed = p.removal
        elif p.removal is not None:
            removed = [p.removal]
        else:
            removed = []

        for point in added:
            if point in all_removals:
                # addition = cancel removal
                all_removals.remove(point)
            else:
                # just add, since it has never been removed
                all_additions.add(point)

        for point in removed:
            if point in all_additions:
                # removing added points = not adding it
                all_additions.remove(point)
            else:
                # removing not already added point
                all_removals.add(point)

    return Perturbation(type=DummyKernel, removal=list(all_removals), addition=list(all_additions))
