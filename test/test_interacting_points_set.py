from typing import List

import numpy as np

from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import DeathKernel
from models.mpp.rjmcmc_sampler.kernels.transform_kernels import DataDrivenTranslationKernel
from models.mpp.custom_types import Perturbation
from base.shapes.base_shapes import Point

BUNCH_OF_POINTS = [
    Point(0, 0),
    Point(1, 0),
    Point(1, 1),
    Point(3, 0),
    Point(5, 5),
    Point(5, 6),
    Point(8, 0),
]


class TestUnitEnergy(UnitEnergyConstructor):

    def __hash__(self):
        return self.__class__.__name__.__hash__()

    def compute(self, u: Point) -> float:
        return 1.0


class TestPairEnergy(PairEnergyConstructor):
    max_dist = 3.0

    def __hash__(self):
        return self.__class__.__name__.__hash__()

    def reduce_point_interactions(self, values: List[float]) -> float:
        return np.max(values)

    def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
        if np.linalg.norm(u_1.get_coord() - u_2.get_coord()) < self.max_dist:
            return 1.0
        return 0


def test___init__():
    points = EPointsSet(
        points=BUNCH_OF_POINTS,
        support_shape=(10, 8),
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=3)]
    )

    for p in BUNCH_OF_POINTS:
        assert p in points
        assert p in points.points
        assert p in points.energy_graph.ue_per_point
        assert p in points.energy_graph.pe_per_point


def test_add():
    points = EPointsSet(
        points=BUNCH_OF_POINTS,
        support_shape=(10, 8),
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=3)]
    )

    u = Point(0, 1)

    points.add(u)

    assert u in points
    assert u in points.points
    assert u in points.energy_graph.ue_per_point
    assert u in points.energy_graph.pe_per_point
    assert len(points.energy_graph.ue_per_point[u]) == 1
    assert len(points.energy_graph.pe_per_point[u]) >= 1


def test_remove():
    points = EPointsSet(
        points=BUNCH_OF_POINTS,
        support_shape=(10, 8),
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=3)]
    )

    u = BUNCH_OF_POINTS[0]

    points.remove(u)

    assert u not in points
    assert u not in points.points
    assert u not in points.energy_graph.ue_per_point
    assert u not in points.energy_graph.pe_per_point
    for p in points:
        for pe in points.energy_graph.pe_per_point[p]:
            other = pe.point_1 if p is pe.point_2 else pe.point_2
            assert other is not u


def test_copy():
    points = EPointsSet(
        points=BUNCH_OF_POINTS,
        support_shape=(10, 8),
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=3)]
    )

    points_2 = points.copy()

    assert points is not points_2
    assert points.points._local_sets is not points_2.points._local_sets
    assert points.energy_graph is not points_2.energy_graph

    u = Point(9, 8)

    points.add(u)

    assert u in points
    assert u not in points_2


def test___iter__():
    points = EPointsSet(
        points=BUNCH_OF_POINTS,
        support_shape=(10, 8),
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=3)]
    )

    it = iter(points)

    for _ in BUNCH_OF_POINTS:
        p = next(it)
        assert p in BUNCH_OF_POINTS

    checks = [False for _ in BUNCH_OF_POINTS]
    for p in points:
        assert p in BUNCH_OF_POINTS
        i = BUNCH_OF_POINTS.index(p)
        # print(p)
        checks[i] = True
    assert all(checks)


def test_total_energy():
    class UE1(UnitEnergyConstructor):

        def __hash__(self):
            return self.__class__.__name__.__hash__()

        def compute(self, u: Point) -> float:
            return 1.0

    class PE1(PairEnergyConstructor):

        def __hash__(self):
            return self.__class__.__name__.__hash__()

        def reduce_point_interactions(self, values: List[float]) -> float:
            return np.max(values)

        def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
            if np.linalg.norm(u_1.get_coord() - u_2.get_coord()) < self.max_dist:
                return 1.0
            return 0

    some_points = [
        Point(0, 0),
        Point(0, 1),
        Point(0, 4)
    ]

    points = EPointsSet(
        points=some_points,
        support_shape=(10, 10),
        unit_energies_constructors=[UE1(weight=1.0)],
        pair_energies_constructors=[PE1(weight=1.0, max_dist=3)]
    )

    energy = points.total_energy()
    assert energy == 5.0
    # 3 * UE + 2 * PE = 5.0
    # since Pair energies are counted twice, once per interacting point

    points.add(Point(0, 5))
    energy = points.total_energy()
    assert energy == 8.0
    # 4 * UE + 4 * PE = 8.0

    some_points = [
        Point(0, 0),
        Point(0, 1),
        Point(1, 0)
    ]
    points = EPointsSet(
        points=some_points,
        support_shape=(10, 10),
        unit_energies_constructors=[UE1(weight=1.0)],
        pair_energies_constructors=[PE1(weight=1.0, max_dist=3)]
    )
    energy = points.total_energy()
    assert energy == 6.0
    # 3 ue + 3 pe
    # each points interacts with the other two, but the max makes the contribution be one per point


def test_energy_delta():
    class UE1(UnitEnergyConstructor):

        def __hash__(self):
            return self.__class__.__name__.__hash__()

        def compute(self, u: Point) -> float:
            return 1.0

    class PE1(PairEnergyConstructor):

        def __hash__(self):
            return self.__class__.__name__.__hash__()

        def reduce_point_interactions(self, values: List[float]) -> float:
            return np.max(values)

        def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
            if np.linalg.norm(u_1.get_coord() - u_2.get_coord()) < self.max_dist:
                return 1.0
            return 0

    some_points = [
        Point(0, 0),
        Point(0, 1),
        Point(0, 5)
    ]
    points_0 = EPointsSet(
        points=some_points,
        support_shape=(10, 10),
        unit_energies_constructors=[UE1(weight=1.0)],
        pair_energies_constructors=[PE1(weight=1.0, max_dist=3)]
    )
    pert1 = Perturbation(type=DeathKernel, removal=some_points[2])

    energy_0 = points_0.total_energy()
    assert energy_0 == 5.0

    delta = points_0.energy_delta(pert1)
    assert delta == -1.0

    points_1 = points_0.apply_perturbation(pert1)

    energy_1 = points_1.total_energy()
    assert energy_1 == 4.0

    assert energy_1 == energy_0 + delta

    pert2 = Perturbation(type=DataDrivenTranslationKernel, removal=some_points[2], addition=Point(1, 0))

    energy_0 = points_0.total_energy()
    assert energy_0 == 5.0

    delta = points_0.energy_delta(pert2)
    assert delta == 1.0

    points_2 = points_0.apply_perturbation(pert2)

    energy_2 = points_2.total_energy()
    assert energy_2 == 6.0

    assert energy_2 == energy_0 + delta
