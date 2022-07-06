import logging
from copy import copy
from typing import List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import norm

from base.shapes.base_shapes import Point
from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.point_set.point_set import PointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import Kernel
from models.shape_net.mappings import ValueMapping
from utils.sampler2d import sample_point_2d


class GaussianTranslationKernel(Kernel):
    def __init__(self, p_kernel: float, sigma: float, shape: Tuple[int, int]):
        self.p_kernel = p_kernel
        self.sigma = sigma
        self.shape = shape
        self.distribution = lambda delta: norm.pdf(delta[0], scale=self.sigma) * norm.pdf(delta[1], scale=self.sigma)

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 0:
            # select a point uniformly amongst all
            p = x.random_choice(rng)
            # sample location delta
            delta = rng.normal((0, 0), self.sigma)
            new_coordinates = (p.get_coord() + delta).astype(int)
            # clip to stay inside image
            new_coordinates = np.clip(new_coordinates, (0, 0), (self.shape[0] - 1, self.shape[1] - 1))
            p_new = copy(p)
            p_new.x = new_coordinates[0]
            p_new.y = new_coordinates[1]
            return Perturbation(self.__class__, addition=p_new, removal=p, data={'delta': delta})
        else:
            # if there is no point, we can't do anything
            return Perturbation(self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n != 0:
            delta = u.data['delta']
            return self.p_kernel * self.distribution(delta) / n

        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n != 0:
            delta = u.data['delta']
            return self.p_kernel * self.distribution(-delta) / n
        else:
            return self.p_kernel


class DataDrivenTranslationKernel(Kernel):

    def __init__(self, p_kernel: float, max_delta: int, normalised_density: np.ndarray):
        self.p_kernel = p_kernel
        self.max_delta = max_delta
        self.normalised_density = normalised_density
        self.max_x = normalised_density.shape[0]
        self.max_y = normalised_density.shape[1]

    def _get_local_density(self, p: Point):
        local_slice = np.s_[
                      max(0, p.x - self.max_delta):min(p.x + self.max_delta + 1, self.max_x),
                      max(0, p.y - self.max_delta):min(p.y + self.max_delta + 1, self.max_y)
                      ]
        return self.normalised_density[local_slice]

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 0:
            # select a point uniformly amongst all
            p = x.random_choice(rng)
            # sample a new location for this point, within max_dist distance
            local_density = self._get_local_density(p)
            coor = sample_point_2d(local_density.shape, density=local_density, rng=rng)[0]
            p_new = copy(p)
            p_new.x = coor[0] + max(0, p.x - self.max_delta)
            p_new.y = coor[1] + max(0, p.y - self.max_delta)
            # p_new = Point(coor[0] + max(0, p.x - self.max_delta), coor[1] + max(0, p.y - self.max_delta),
            #               radius=p.radius)
            return Perturbation(self.__class__, addition=p_new, removal=p)
        else:
            # if there is no point, we can't do anything
            return Perturbation(self.__class__)

    def _move_density(self, p_start, p_end, n_points):
        local_density = self._get_local_density(p_start)
        local_density = local_density / np.sum(local_density)
        point_density = local_density[p_end.x - max(0, p_start.x - self.max_delta),
                                      p_end.y - max(0, p_start.y - self.max_delta)]
        return self.p_kernel * point_density / n_points

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n != 0:
            return self._move_density(u.removal, u.addition, n)

        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n != 0:
            return self._move_density(u.addition, u.removal, n)
        else:
            return self.p_kernel


class GaussianShapeTransformKernel(Kernel):
    def __init__(self, p_kernel: float, sigma: float, param_names: List[str], mappings: List[ValueMapping]):
        self.p_kernel = p_kernel
        self.sigma_per_param = [sigma * (m.v_max - m.v_min) for m in mappings]
        self.param_names = param_names
        self.mappings = mappings
        self.n_params = len(mappings)
        assert len(mappings) == len(param_names)

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 0:
            p = x.random_choice(rng)
            chosen_param_id = rng.integers(self.n_params)
            chosen_param = self.param_names[chosen_param_id]
            value_delta = rng.normal(0, self.sigma_per_param[chosen_param_id])
            new_value = p.__getattribute__(chosen_param) + value_delta
            mapping = self.mappings[chosen_param_id]
            if mapping.is_cyclic:
                new_value = (new_value % (mapping.v_max - mapping.v_min)) + mapping.v_min
            else:
                new_value = np.clip(new_value, mapping.v_min, mapping.v_max)
            new_p = copy(p)
            new_p.__setattr__(chosen_param, new_value)
            return Perturbation(self.__class__, removal=p, addition=new_p,
                                data={'param_id': chosen_param_id, 'delta': value_delta})
        else:
            return Perturbation(self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n > 0:
            chosen_param_id = u.data['param_id']
            delta = u.data['delta']
            density = stats.norm.pdf(delta, scale=self.sigma_per_param[chosen_param_id])
            return self.p_kernel * density / n
        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        return self.forward_probability(x, u)  # since the distribution is symetrical


class DataDrivenShapeTransformKernel(Kernel):
    def __init__(self, p_kernel: float, params_density_maps: List[np.ndarray], param_names: List[str],
                 mappings: List[ValueMapping], re_normalize: bool):
        self.p_kernel = p_kernel
        self.params_density_maps = params_density_maps
        self.param_names = param_names
        self.n_params = len(param_names)
        self.mappings = mappings
        if re_normalize:
            for i in range(len(self.params_density_maps)):
                density_map = self.params_density_maps[i]
                sum_pp = np.sum(density_map, axis=-1, keepdims=True)
                if np.any(np.abs(1-sum_pp) > 1e-3):
                    logging.warning(f"density does not sum to one for param {i} :\n"
                                    f"max error={np.max(np.abs(1-sum_pp))}")
                self.params_density_maps[i] = density_map / sum_pp

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 0:
            # select point of interest
            p = x.random_choice(rng)
            # select changed param
            param_id = rng.integers(0, self.n_params)

            p_new = copy(p)
            param_density = self.params_density_maps[param_id][p.x, p.y]
            n_classes = len(param_density)
            try:
                new_param_class_value = rng.choice(range(n_classes), p=param_density)
            except ValueError as e:
                print(f"this does not sum to 1 :\nsum({param_density})={np.sum(param_density)}")
                raise e

            new_param_value = self.mappings[param_id].class_to_value(new_param_class_value)
            p_new.__setattr__(self.param_names[param_id], new_param_value)

            return Perturbation(self.__class__, addition=p_new, removal=p,
                                data={'param_id': param_id, 'param_density': param_density,
                                      'new_param_class_value': new_param_class_value})
        else:
            # no points, can't do anything
            return Perturbation(self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n > 0:
            new_param_class_value = u.data['new_param_class_value']
            choice_density = u.data['param_density'][new_param_class_value]
            return self.p_kernel * choice_density / n
        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n > 0:
            param_id = u.data['param_id']
            prev_param_id = self.mappings[param_id].value_to_class(
                u.removal.__getattribute__(self.param_names[param_id]))
            choice_density = u.data['param_density'][prev_param_id]
            return self.p_kernel * choice_density / n
        else:
            return self.p_kernel
