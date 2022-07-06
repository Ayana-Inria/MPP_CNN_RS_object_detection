from dataclasses import dataclass

from base.shapes.base_shapes import Point
from models.mpp.energies.base_energies import UnitEnergyConstructor


@dataclass
class ConstantEnergy(UnitEnergyConstructor):
    value: float

    def compute(self, u: Point) -> float:
        return self.value

    def __hash__(self):
        return id(self)
