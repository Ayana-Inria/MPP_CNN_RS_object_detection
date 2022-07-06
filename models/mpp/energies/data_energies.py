from dataclasses import dataclass
from typing import List

import numpy as np

from base.shapes.base_shapes import Point
from base.shapes.rectangle import Rectangle
from models.mpp.energies.base_energies import UnitEnergyConstructor
from models.shape_net.mappings import ValueMapping


@dataclass
class PositionEnergy(UnitEnergyConstructor):
    detection_map: np.ndarray
    threshold: float

    def __post_init__(self, ):
        self.energy_map = -2 * (self.detection_map - self.threshold)

    def compute(self, u: Point) -> float:
        return self.energy_map[u.x, u.y]

    def __hash__(self):
        return id(self)


@dataclass
class ShapeEnergy(UnitEnergyConstructor):
    parameter_energy_map: List[np.ndarray]
    mappings: List[ValueMapping]
    param_names: List[str]

    def __post_init__(self):
        # todo make interpolation methods
        n_classes = self.parameter_energy_map[0].shape[-1]

    def compute(self, u: Rectangle) -> float:
        params = [u.__getattribute__(p) for p in self.param_names]
        densities = [
            self.parameter_energy_map[i][u.x, u.y, self.mappings[i].value_to_class(params[i])]
            for i in range(3)]  # todo use interpolation, for now I sample a constant per category
        return float(np.mean(densities))

    def __hash__(self):
        return id(self)


@dataclass
class SingleMarkEnergy(UnitEnergyConstructor):
    parameter_energy_map: np.ndarray
    mapping: ValueMapping
    param_name: str

    def __post_init__(self):
        n_classes = self.parameter_energy_map.shape[-1]

    def compute(self, u: Rectangle) -> float:
        param = u.__getattribute__(self.param_name)
        density = self.parameter_energy_map[u.x, u.y, self.mapping.value_to_class(param)]
        # todo use interpolation, for now I sample a constant per category
        return density

    def __hash__(self):
        return id(self)
