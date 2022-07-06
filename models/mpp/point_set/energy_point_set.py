from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np

# todo rename to EPointSet
from models.mpp.custom_types.energy import EnergyCombinationModel
from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.point_set.energy_graph import EnergyGraph
from models.mpp.point_set.point_set import PointsSet
from base.shapes.base_shapes import Point
from models.mpp.rjmcmc_sampler.kernels.base_kernels import BirthKernel


class EPointsSet:

    def __init__(self, points: Iterable[Point], support_shape: Tuple[int, int],
                 unit_energies_constructors: List[UnitEnergyConstructor],
                 pair_energies_constructors: List[PairEnergyConstructor], debug=False):
        self.debug = debug

        for c_list in [unit_energies_constructors, pair_energies_constructors]:
            for c1 in c_list:
                for c2 in c_list:
                    if c1 is not c2:
                        assert c1.name != c2.name  # does not support same name energies

        if debug:
            logging.warning('activated debug mode in EPointsSet, may slow down execution')
        if len(pair_energies_constructors) > 0:
            self.maximum_interaction_radius = np.max([pec.max_dist for pec in pair_energies_constructors])
        else:
            self.maximum_interaction_radius = 0
        self.points: PointsSet = PointsSet(
            support_shape=support_shape,
            maximum_interaction_radius=self.maximum_interaction_radius
        )
        self.energy_graph = EnergyGraph(
            unit_energies_constructors=unit_energies_constructors,
            pair_energies_constructors=pair_energies_constructors,
        )
        for u in points:
            self.points.add(u)
            self.energy_graph.add_point(u, self.points)

    def __copy__(self):
        new_x = EPointsSet(
            points=[],
            support_shape=self.points.support_shape,
            unit_energies_constructors=self.energy_graph.ue_constructors,
            pair_energies_constructors=self.energy_graph.pe_constructors
        )
        new_x.points = self.points.copy()
        new_x.energy_graph = self.energy_graph.copy()
        return new_x

    def copy(self):
        return self.__copy__()

    def __len__(self):
        return len(self.points)

    def __contains__(self, u: Point):
        return u in self.points

    def __iter__(self):
        return self.points.__iter__()

    def add(self, u: Point):
        self.points.add(u)
        self.energy_graph.add_point(u, self.points)

    def remove(self, u: Point):
        self.points.remove(u)
        self.energy_graph.remove_point(u)

    def total_energy(self, force_update=False) -> float:
        return self.energy_graph.total_energy(points_set=self.points)

    def energy_delta(self, p: Perturbation, energy_combinator: EnergyCombinationModel = None):
        try:
            return self.energy_graph.compute_delta(
                self.points, pert=p,
                energy_combinator=energy_combinator)
        except KeyError as e:
            self.energy_graph.check_integrity()
            if type(p.removal) is list:
                removed = p.removal
            elif p.removal is not None:
                removed = [p.removal]
            else:
                removed = []

            for r in removed:
                if r not in self.points:
                    logging.error(f"point to remove {r} not in points set {self}")
            raise e

    def papangelou(self, u: Point, energy_combinator: EnergyCombinationModel = None,
                   remove_u_from_point_set: bool = False, return_energy_delta: bool = False):
        if u in self.points:
            if not remove_u_from_point_set:
                print(f"point {u} is already in current set, cannot compute papangelou conditional intensity")
                raise ValueError
            else:
                removal = Perturbation(type=BirthKernel, removal=u, addition=None)
                delta = - self.energy_delta(p=removal, energy_combinator=energy_combinator)  # -(old - new)
        else:
            addition = Perturbation(type=BirthKernel, removal=None, addition=u)
            delta = self.energy_delta(p=addition, energy_combinator=energy_combinator)  # new - old
        if return_energy_delta:
            return delta
        return np.exp(-delta)

    def apply_perturbation(self, p: Perturbation, inplace=False) -> EPointsSet:
        if inplace:
            new_x = self
        else:
            new_x = self.copy()
        if p.removal is not None:
            removed = p.removal if type(p.removal) is list else [p.removal]
            for r in removed:
                if self.debug:
                    assert r in new_x
                    assert r in new_x.energy_graph.pe_per_point
                    assert r in new_x.energy_graph.ue_per_point

                new_x.remove(r)

                if self.debug:
                    neigh = new_x.points.get_potential_neighbors(r,
                                                                 radius=self.maximum_interaction_radius,
                                                                 exclude_itself=True)
                    for n in neigh:
                        assert r not in new_x.energy_graph.get_interacting_points(n)

        if p.addition is not None:
            added = p.addition if type(p.addition) is list else [p.addition]
            for a in added:
                new_x.add(a)
                if self.debug:
                    assert a in new_x.energy_graph.pe_per_point

        if self.debug:
            for u in new_x:
                assert u in new_x.energy_graph.ue_per_point
                assert u in new_x.energy_graph.pe_per_point
            for u in new_x.energy_graph.pe_per_point.keys():
                assert u in new_x

        return new_x

    def unapply_perturbation(self, p: Perturbation) -> EPointsSet:
        new_x = self.copy()
        if p.addition is not None:
            added = p.addition if type(p.addition) is list else [p.addition]
            for a in added:
                new_x.remove(a)
        if p.removal is not None:
            removed = p.removal if type(p.removal) is list else [p.removal]
            for r in removed:
                new_x.add(r)
        return new_x
