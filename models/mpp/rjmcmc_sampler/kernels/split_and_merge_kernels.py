from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from base.shapes.rectangle import Rectangle
from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.point_set.point_set import PointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import Kernel
from models.shape_net.mappings import ValueMapping


@dataclass
class SplitSampler:
    pos_radius: float
    shape_sigmas: List[float]
    mappings: List[ValueMapping]

    def __post_init__(self):
        self.scaled_shaped_sigmas = [s * m.range for m, s in zip(self.mappings, self.shape_sigmas)]
        self.n_params = len(Rectangle.PARAMETERS)
        # self.pdf_scale = np.prod([m.range for m in self.mappings])
        self.pdf_scale = 1.0

    def sample(self, rng: np.random.Generator):
        pos_deltas = rng.uniform((0, 0), self.pos_radius)
        while np.linalg.norm(pos_deltas) > self.pos_radius:
            pos_deltas = rng.uniform((0, 0), self.pos_radius)
        shape_delta = rng.normal((0,) * self.n_params, self.scaled_shaped_sigmas)
        return pos_deltas, shape_delta

    def pdf(self, pos_deltas, shape_deltas) -> float:
        p_pos = 1 / (np.pi * self.pos_radius * self.pos_radius)
        p_shape = [norm.pdf(d, scale=s) for d, s in zip(shape_deltas, self.scaled_shaped_sigmas)]
        return np.prod(p_pos) * np.prod(p_shape) * self.pdf_scale


class SplitKernel(Kernel):

    def __init__(self, p_split: float, p_merge: float, split_sampler: SplitSampler, support_shape: Tuple[int, int],
                 intensity: float, merge_radius: float):
        self.p_split = p_split
        self.p_merge = p_merge
        self.split_sampler: SplitSampler = split_sampler
        self.shape = support_shape
        self.intensity = intensity
        self.radius = merge_radius
        assert self.radius == self.split_sampler.pos_radius

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 0:
            # choose p0
            p = x.random_choice(rng)
            # draw position delta and parameters delta
            pos_delta, shape_delta = self.split_sampler.sample(rng)

            new_p0 = Rectangle(
                x=int(np.clip(p.x - pos_delta[0], 0, self.shape[0] - 1)),
                y=int(np.clip(p.y - pos_delta[1], 0, self.shape[1] - 1)),
                **{a: m.clip(p.__getattribute__(a) - d) for a, d, m in
                   zip(Rectangle.PARAMETERS, shape_delta, self.split_sampler.mappings)}
            )

            new_p1 = Rectangle(
                x=int(np.clip(p.x + pos_delta[0], 0, self.shape[0] - 1)),
                y=int(np.clip(p.y + pos_delta[1], 0, self.shape[1] - 1)),
                **{a: m.clip(p.__getattribute__(a) + d) for a, d, m in
                   zip(Rectangle.PARAMETERS, shape_delta, self.split_sampler.mappings)}
            )

            return Perturbation(self.__class__, addition=[new_p0, new_p1], removal=p,
                                data={'pos_delta': pos_delta, 'shape_delta': shape_delta})
        else:
            return Perturbation(self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n != 0:
            # choose p0
            p = 1 / n
            # draw position delta and parameters delta
            p = p * self.split_sampler.pdf(u.data['pos_delta'], u.data['shape_delta'])
            return self.p_kernel * p / self.intensity
        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x) + 1
        if n > 1:
            n_neighbors_0 = len(x.get_potential_neighbors(u.addition[0], radius=self.radius)) + 1
            n_neighbors_1 = len(x.get_potential_neighbors(u.addition[1], radius=self.radius)) + 1
            if n_neighbors_0 == 0 and n_neighbors_1 == 0:
                return self.p_merge
            # choose 0 * choose 1 in the vicinity of 0
            p0 = (1 / n) * (1 / n_neighbors_0)
            # choose 1 * choose 1 in the vicinity of 0
            p1 = (1 / n) * (1 / n_neighbors_1)
            return self.p_merge * (p0 + p1)
        else:
            return self.p_merge

    @property
    def p_kernel(self) -> float:
        return self.p_split


class MergeKernel(Kernel):

    def __init__(self, p_split: float, p_merge: float, split_sampler: SplitSampler, support_shape: Tuple[int, int],
                 intensity: float, merge_radius: float):
        self.p_split = p_split
        self.p_merge = p_merge
        self.split_sampler: SplitSampler = split_sampler
        self.shape = support_shape
        self.intensity = intensity
        self.radius = merge_radius
        assert self.radius == self.split_sampler.pos_radius

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        if len(x) > 1:
            # choose p0
            p0 = x.random_choice(rng)
            # choose p1 in the vicinity of p0
            neighbors = x.get_neighbors(p0, radius=self.radius)
            n_neighbor = len(neighbors)
            data = {'n_neighbors': n_neighbor}
            if n_neighbor == 0:
                return Perturbation(self.__class__, data=data)
            p1 = rng.choice(list(neighbors))
            # make new point as an average of the two
            p_new = Rectangle(

                x=int(np.clip((p0.x + p1.x)/2, 0, self.shape[0] - 1)),
                y=int(np.clip((p0.y + p1.y)/2, 0, self.shape[0] - 1)),
                **{a: m.clip((p0.__getattribute__(a) + p1.__getattribute__(a))/2) for a, m in
                   zip(Rectangle.PARAMETERS, self.split_sampler.mappings)}
            )

            return Perturbation(self.__class__, addition=p_new, removal=[p0, p1], data=data)
        else:
            return Perturbation(self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n > 1:
            n_neighbors = u.data['n_neighbors']
            if n_neighbors == 0:
                return self.p_kernel
            # choose p0 * choose p1 in the vicinity of p0
            p = (1 / n) * (1 / n_neighbors)
            return self.p_kernel * p
        else:
            return self.p_kernel

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x) - 1
        if n != 0:
            if u.removal is None:
                return self.p_split
            # choose point
            p = 1 / n
            # draw position delta and parameters delta
            p0, p1 = u.removal[0], u.removal[1]
            pos_delta = [(p0.x - p1.x) / 2, (p0.y - p1.y) / 2]
            shape_delta = [(p0.__getattribute__(a) - p1.__getattribute__(a)) / 2 for a in Rectangle.PARAMETERS]
            p = p * self.split_sampler.pdf(pos_delta, shape_delta)
            return self.p_split * p / self.intensity
        else:
            return self.p_split

    @property
    def p_kernel(self) -> float:
        return self.p_merge
