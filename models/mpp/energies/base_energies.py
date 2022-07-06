from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

from base.shapes.base_shapes import Point


@dataclass
class UnitEnergyConstructor:
    name: str
    """
    Energy term for a single point
    """

    @abstractmethod
    def compute(self, u: Point) -> float:
        pass

    def instanciate(self, u: Point):
        return UnitEnergy(point=u, constructor=self)

    @abstractmethod
    def __hash__(self):
        pass


@dataclass
class UnitEnergy:
    point: Point
    constructor: UnitEnergyConstructor
    value: float = None

    def compute(self, lazy=True) -> float:
        if self.value is None or not lazy:
            self.value = self.constructor.compute(u=self.point)
        return self.value


@dataclass
class PairEnergyConstructor:
    name: str
    max_dist: float

    @abstractmethod
    def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
        pass

    @abstractmethod
    def reduce_point_interactions(self, values: List[float]) -> float:
        """
        compute the energy value for the overall point given all of the one to one interaction,
        should be a min or max function depending if the interaction is penalized or rewarded
        Parameters
        ----------
        values : list of energy values of all interactions of that point

        Returns
        -------

        """

    def instanciate(self, u_1: Point, u_2: Point):
        return PairEnergy(point_1=u_1, point_2=u_2, constructor=self)

    @abstractmethod
    def __hash__(self):
        pass


@dataclass
class PairEnergy:
    point_1: Point
    point_2: Point
    constructor: PairEnergyConstructor
    value: float = None

    def compute(self, lazy=True) -> float:
        if self.value is None or not lazy:
            self.value = self.constructor.compute_one_interaction(u_1=self.point_1, u_2=self.point_2)
        return self.value

    def get_other_point(self, current: Point):
        return self.point_1 if self.point_2 is current else self.point_2


EnergyConstructor = Union[UnitEnergyConstructor, PairEnergyConstructor]
