from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from base.shapes.base_shapes import Point
from base.shapes.rectangle import Rectangle
from models.mpp.point_set.point_set import PointsSet
from models.shape_net.mappings import ValueMapping
from utils.sampler2d import sample_point_2d


class ShapeSampler:

    @abstractmethod
    def sample(self, x: PointsSet) -> Point:
        pass

    @abstractmethod
    def get_normalised_density(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_point_density(self, u: Point) -> float:
        pass

    @abstractmethod
    def get_pixel_count(self) -> float:
        pass


class UniformPointSampler(ShapeSampler):

    def __init__(self, image_shape: Tuple[int, int]):
        self.image_shape = image_shape
        self._normalized_density = np.ones(self.image_shape, dtype=float) / (self.image_shape[0] * self.image_shape[1])

    def get_normalised_density(self) -> np.ndarray:
        return self._normalized_density

    def sample(self, x: PointsSet) -> Point:
        coordinates = sample_point_2d(
            img_shape=self.image_shape
        )[0]
        return Point(x=coordinates[0], y=coordinates[1])

    def get_point_density(self, u: Point) -> float:
        return self._normalized_density[u.x, u.y]

    def get_pixel_count(self) -> float:
        return self._normalized_density.size


class PointSampler(ShapeSampler):

    def __init__(self, normalized_density: np.ndarray):
        self._normalized_density = normalized_density

    def get_normalised_density(self) -> np.ndarray:
        return self._normalized_density

    def sample(self, x: PointsSet) -> Point:
        coordinates = sample_point_2d(
            img_shape=self._normalized_density.shape,
            density=self._normalized_density,
            skip_normalization=True
        )[0]
        return Point(x=coordinates[0], y=coordinates[1])

    def get_point_density(self, u: Point) -> float:
        return self._normalized_density[u.x, u.y]

    def get_pixel_count(self) -> float:
        return self._normalized_density.size


@dataclass
class RectangleSampler(ShapeSampler):
    detection_map: np.ndarray
    param_dist_maps: List[np.ndarray]
    param_names: List[str]
    mappings: List[ValueMapping]
    rng: np.random.Generator

    def __post_init__(self):
        self.normalised_detection_map = self.detection_map / np.sum(self.detection_map)
        self.norm_constant = np.prod(self.detection_map.shape) * np.prod([m.n_classes for m in self.mappings])

    def sample(self, x: PointsSet) -> Rectangle:
        pos_x, pos_y = sample_point_2d(
            img_shape=self.detection_map.shape,
            density=self.normalised_detection_map,
            skip_normalization=True, rng=self.rng)[0]
        params = {n: self._sample_param(self.mappings[i], self.param_dist_maps[i][pos_x, pos_y]) for i, n in
                  enumerate(self.param_names)}  # todo is the sampling correct ?

        return Rectangle(x=pos_x, y=pos_y, **params)

    def get_normalised_density(self) -> np.ndarray:
        return self.normalised_detection_map

    def get_point_density(self, u: Rectangle) -> float:
        params_proba = [
            self._param_density(self.mappings[i], self.param_dist_maps[i][u.x, u.y], u.__getattribute__(attr))
            for i, attr in enumerate(self.param_names)
        ]
        return self.normalised_detection_map[u.x, u.y] * np.prod(params_proba) * self.norm_constant

    def get_pixel_count(self) -> float:
        return self.normalised_detection_map.size

    def _sample_param(self, mapping: ValueMapping, distribution: np.ndarray) -> float:
        try:
            return self.rng.choice(mapping.feature_mapping, p=distribution)  # todo use interpolation
        except ValueError:
            return self.rng.choice(mapping.feature_mapping, p=distribution / np.sum(distribution))

    @staticmethod
    def _param_density(mapping: ValueMapping, distribution, value: float) -> float:
        return distribution[mapping.value_to_class(value)]


@dataclass
class UniformRectangleSampler(ShapeSampler):
    shape: Tuple[int, int]
    param_names: List[str]
    mappings: List[ValueMapping]
    rng: np.random.Generator

    def __post_init__(self):
        self.normalized_density = np.ones(self.shape)
        self.normalized_density = self.normalized_density / np.sum(self.normalized_density)
        self.n_classes = [m.n_classes for m in self.mappings]

    def sample(self, x: PointsSet) -> Point:
        pos_x, pos_y = self.rng.integers((0, 0), self.shape)
        params = {n: self.rng.uniform(m.v_min, m.v_max) for m, n in
                  zip(self.mappings, self.param_names)}  # todo is the sampling correct ?

        return Rectangle(x=pos_x, y=pos_y, **params)

    def get_normalised_density(self) -> np.ndarray:
        return self.normalized_density

    def get_point_density(self, u: Point) -> float:
        return 1.0

    def get_pixel_count(self) -> float:
        return self.shape[0] * self.shape[1]
