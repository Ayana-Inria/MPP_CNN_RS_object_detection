import time
from typing import List

import numpy as np

from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.point_set.energy_graph import EnergyGraph
from models.mpp.point_set.point_set import PointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import BirthKernel, DeathKernel
from models.mpp.rjmcmc_sampler.kernels.transform_kernels import DataDrivenTranslationKernel
from models.mpp.custom_types import Perturbation
from base.shapes.base_shapes import Point


class TestUnitEnergy(UnitEnergyConstructor):
    # returns  -10

    def __hash__(self):
        return id(self)

    def compute(self, u: Point) -> float:
        return -10.0


class TestPairEnergy(PairEnergyConstructor):
    # returns 1 if distance <= max_dist

    def __hash__(self):
        return id(self)

    def reduce_point_interactions(self, values: List[float]) -> float:
        return np.max(values)

    def compute_one_interaction(self, u_1: Point, u_2: Point) -> float:
        return float(np.linalg.norm(u_1.get_coord() - u_2.get_coord()) <= self.max_dist)


def test_energygraph():
    eg = EnergyGraph(
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=1.0)],
    )
    ps = PointsSet(
        support_shape=(64, 64),
        maximum_interaction_radius=32
    )

    # add a point
    p1 = Point(10, 10)
    ps.add(p1)
    eg.add_point(p1, ps)

    assert p1 in eg.ue_per_point
    assert len(eg.ue_per_point[p1]) == 1
    assert p1 in eg.pe_per_point
    assert len(eg.pe_per_point[p1]) == 0

    # add a point that interacts with the first one
    p2 = Point(10, 11)
    ps.add(p2)
    eg.add_point(p2, ps)

    assert p2 in eg.ue_per_point
    assert len(eg.ue_per_point[p2]) == 1
    assert p2 in eg.pe_per_point
    assert len(eg.pe_per_point[p2]) == 1
    assert len(eg.pe_per_point[p1]) == 1
    assert eg.pe_per_point[p2][0].point_2 is p1
    assert eg.pe_per_point[p2][0] is eg.pe_per_point[p1][0]

    # add a point that interacts with none
    p3 = Point(20, 20)
    ps.add(p3)
    eg.add_point(p3, ps)

    assert p3 in eg.ue_per_point
    assert len(eg.ue_per_point[p2]) == 1
    assert p3 in eg.pe_per_point
    assert len(eg.pe_per_point[p3]) == 0
    assert len(eg.pe_per_point[p2]) == 1
    assert len(eg.pe_per_point[p1]) == 1

    # remove the second point
    ps.remove(p2)
    eg.remove_point(p2)
    assert len(eg.pe_per_point[p3]) == 0
    assert len(eg.pe_per_point[p1]) == 0


#
# def test_copy():
#     assert False

def test_total_energy():
    eg = EnergyGraph(
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=1.0)],
    )
    ps = PointsSet(
        support_shape=(64, 64),
        maximum_interaction_radius=32
    )

    assert eg.total_energy(ps) == 0.0

    # add a point
    p1 = Point(10, 10)
    ps.add(p1)
    eg.add_point(p1, ps)

    assert eg.total_energy(ps) == -10.0

    # add a point that interacts with the first one
    p2 = Point(10, 11)
    ps.add(p2)
    eg.add_point(p2, ps)

    assert eg.total_energy(ps) == 2 * -10.0 + 2 * 1.0

    # add a point that interacts with none
    p3 = Point(20, 20)
    ps.add(p3)
    eg.add_point(p3, ps)

    assert eg.total_energy(ps) == 3 * -10.0 + 2 * 1.0

    # remove the second point
    ps.remove(p2)
    eg.remove_point(p2)
    assert eg.total_energy(ps) == 2 * -10.0


def test_compute_subset():
    eg = EnergyGraph(
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=1.0)],
    )
    ps = PointsSet(
        support_shape=(64, 64),
        maximum_interaction_radius=32
    )

    assert eg.total_energy(ps) == 0.0

    # add a point
    p1 = Point(10, 10)
    ps.add(p1)
    eg.add_point(p1, ps)

    assert eg.compute_subset([p1]) == -10.0

    # add a point that interacts with the first one
    p2 = Point(10, 11)
    ps.add(p2)
    eg.add_point(p2, ps)

    assert eg.compute_subset([p1]) == -10.0 + 1 * 1.0
    assert eg.compute_subset([p2]) == -10.0 + 1 * 1.0

    # add a point that interacts with none
    p3 = Point(20, 20)
    ps.add(p3)
    eg.add_point(p3, ps)

    assert eg.compute_subset([p1, p2]) == 2 * -10.0 + 2 * 1.0
    assert eg.compute_subset([p3]) == -10.0
    assert eg.compute_subset([p3, p2]) == 2 * -10.0 + 1 * 1.0

    # remove the second point
    ps.remove(p2)
    eg.remove_point(p2)
    assert eg.compute_subset([p1]) == -10.0
    assert eg.compute_subset([p3]) == -10.0
    assert eg.compute_subset([p1, p3]) == 2 * -10.0


def test_compute_delta():
    eg = EnergyGraph(
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=1.0)],
    )
    ps = PointsSet(
        support_shape=(64, 64),
        maximum_interaction_radius=32
    )

    # add a point
    p1 = Point(10, 10)
    delta = eg.compute_delta(ps, Perturbation(type=BirthKernel, removal=None, addition=p1))
    assert delta == -10.0
    ps.add(p1)
    eg.add_point(p1, ps)

    # add a point that interacts with the first one
    p2 = Point(10, 11)
    delta = eg.compute_delta(ps, Perturbation(type=BirthKernel, removal=None, addition=p2))
    assert delta == -10.0 + 2 * 1.0
    ps.add(p2)
    eg.add_point(p2, ps)

    # add a point that interacts with none
    p3 = Point(20, 20)
    delta = eg.compute_delta(ps, Perturbation(type=BirthKernel, removal=None, addition=p3))
    assert delta == -10.0
    ps.add(p3)
    eg.add_point(p3, ps)

    p32 = Point(10, 12)
    delta = eg.compute_delta(ps, Perturbation(type=DataDrivenTranslationKernel, removal=p3, addition=p32))
    assert delta == 1.0  # p2 already interacts with p1, only p3 has new interactions
    ps.remove(p3)
    eg.remove_point(p3)
    ps.add(p32)
    eg.add_point(p32, ps)

    p4 = Point(5, 5)
    delta = eg.compute_delta(ps, Perturbation(type=BirthKernel, removal=None, addition=p4))
    assert delta == -10.0
    ps.add(p4)
    eg.add_point(p4, ps)

    p5 = Point(5, 6)
    delta = eg.compute_delta(ps, Perturbation(type=BirthKernel, removal=None, addition=p5))
    assert delta == -10.0 + 2 * 1.0
    ps.add(p4)
    eg.add_point(p4, ps)

    p6 = Point(5, 7)
    ps.add(p6)
    eg.add_point(p6, ps)

    p61 = Point(5, 8)
    delta = eg.compute_delta(ps, Perturbation(type=DataDrivenTranslationKernel, removal=p6, addition=p61))
    assert delta == 0.0
    ps.remove(p6)
    eg.remove_point(p6)
    ps.add(p61)
    eg.add_point(p61, ps)

    # remove the second point
    delta = eg.compute_delta(ps, Perturbation(type=DeathKernel, removal=p2, addition=None))
    assert delta == +10.0 - 3 * 1.0
    ps.remove(p2)
    eg.remove_point(p2)


def test_time_compute_delta():
    nb_points = 200
    s = (128, 128)
    rng = np.random.default_rng(0)
    ps2 = PointsSet(
        support_shape=s,
        maximum_interaction_radius=32
    )
    eg2 = EnergyGraph(
        unit_energies_constructors=[TestUnitEnergy(weight=1.0)],
        pair_energies_constructors=[TestPairEnergy(weight=1.0, max_dist=1.0)],
    )
    for _ in range(nb_points):
        p = Point(rng.integers(s[0]), rng.integers(s[1]))
        ps2.add(p)
        eg2.add_point(p, ps2)

    n_test = 10000
    tot_time = 0
    for _ in range(n_test):
        r = rng.random()
        if r < 1 / 3:
            pert = Perturbation(
                type=BirthKernel, removal=None, addition=Point(x=rng.integers(s[0]), y=rng.integers(s[1]))
            )
        elif r < 2 / 3:
            if len(ps2) > 0:
                removal = ps2.random_choice(rng)
            else:
                removal = None
            pert = Perturbation(
                type=DeathKernel, removal=removal, addition=None
            )
        else:
            if len(ps2) > 0:
                removal = ps2.random_choice(rng)
            else:
                removal = None
            pert = Perturbation(
                type=DataDrivenTranslationKernel, removal=removal, addition=Point(x=rng.integers(s[0]), y=rng.integers(s[1]))
            )
        start = time.perf_counter()
        eg2.compute_delta(ps2, pert)
        end = time.perf_counter()
        tot_time += end - start

    print(f"Compute delta time {tot_time / n_test:.2e}")

# def test_energies_as_dict():
#     assert False
