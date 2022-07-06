from dataclasses import dataclass
from typing import List

import numpy as np

from base.shapes.base_shapes import Point
from base.shapes.rectangle import Rectangle
from models.mpp.energies.base_energies import PairEnergyConstructor, UnitEnergyConstructor


@dataclass
class RectangleOverlapEnergy(PairEnergyConstructor):
    def compute_one_interaction(self, u_1: Rectangle, u_2: Rectangle) -> float:
        poly_1 = u_1.poly
        poly_2 = u_2.poly
        inter = poly_1.intersection(poly_2).area
        min_area = min(poly_1.area, poly_2.area)
        return inter / (min_area + 1e-6)

    def reduce_point_interactions(self, values: List[float]) -> float:
        return np.max(values)

    def __hash__(self):
        return id(self)


@dataclass
class ShapeAlignmentEnergy(PairEnergyConstructor):
    rewarding: bool
    angle_param_name: str = 'angle'

    def _get_angle(self, u: Point):
        return u.__getattribute__(self.angle_param_name)

    @staticmethod
    def response_function(angle_delta, rewarding: bool):
        return 1 - np.abs(np.cos(angle_delta)) - int(rewarding)

    def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
        delta = self._get_angle(u_1) - self._get_angle(u_2)
        return self.response_function(delta, self.rewarding)

    def reduce_point_interactions(self, values: List[float]) -> float:
        if self.rewarding:
            return np.min(values)
        else:
            return np.max(values)

    def __hash__(self):
        return id(self)


@dataclass
class AreaPriorEnergy(UnitEnergyConstructor):
    min_area: float
    max_area: float

    @staticmethod
    def response_function(x, min_a, max_a):
        return np.maximum(0, np.maximum(min_a - x, x - max_a))

    def compute(self, u: Rectangle) -> float:
        area = u.poly.area
        return self.response_function(area, self.min_area, self.max_area)

    def __hash__(self):
        return id(self)


@dataclass
class RatioPriorEnergy(UnitEnergyConstructor):
    target_ratio: float

    def compute(self, u: Rectangle) -> float:
        return abs(self.target_ratio - u.ratio)

    def __hash__(self):
        return id(self)
