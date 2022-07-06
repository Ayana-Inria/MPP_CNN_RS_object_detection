from typing import List, Dict, Union, Set, Any

import numpy as np

from base.shapes.base_shapes import Point
from models.mpp.custom_types.energy import EnergyCombinationModel, ConfigurationEnergyVector, PointEnergyVector
from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor, UnitEnergy, PairEnergy
from models.mpp.point_set.point_set import PointsSet


def append_to_dict_of_list(d: Dict[Any, List], update_d):
    for key, value in update_d.items():
        if key in d:
            d[key].append(value)
        else:
            d[key] = [value]


class EnergyGraph:
    def __init__(self, unit_energies_constructors: List[UnitEnergyConstructor],
                 pair_energies_constructors: List[PairEnergyConstructor]):
        self.ue_constructors = unit_energies_constructors
        self.pe_constructors = pair_energies_constructors

        if len(pair_energies_constructors) > 0:
            self.max_interaction_dist = np.max([pec.max_dist for pec in pair_energies_constructors])
        else:
            self.max_interaction_dist = 1

        # self.unit_energies: List[UnitEnergy] = []
        # self.pair_energies: List[PairEnergy] = []

        self.ue_per_point: Dict[Point, List[UnitEnergy]] = {}
        self.pe_per_point: Dict[Point, List[PairEnergy]] = {}

        ue_keys = [uec.name for uec in self.ue_constructors]
        pe_keys = [pec.name for pec in self.pe_constructors]
        self.energies_keys = ue_keys + pe_keys
        try:
            assert len(self.energies_keys) == len(set(self.energies_keys))
        except AssertionError as e:
            print(f"duplicate energy names in {self.energies_keys}")
            raise e

    def add_point(self, u: Point, points_set: PointsSet):
        """
        adds the energies associated to a point and (todo) returns the energy delta
        Parameters
        ----------
        u : point to add
        points_set : set of points

        Returns
        -------

        """
        self.ue_per_point[u] = list()
        self.pe_per_point[u] = list()

        for uec in self.ue_constructors:
            new_ue: UnitEnergy = uec.instanciate(u)
            self.ue_per_point[u].append(new_ue)

        potential_neighbors = points_set.get_potential_neighbors(u, radius=self.max_interaction_dist)
        potential_neighbors = list(potential_neighbors)
        neighbors_distances = [np.linalg.norm(u.get_coord() - n.get_coord()) for n in potential_neighbors]
        # neighbors_distances = list(map(
        #     lambda n: np.linalg.norm(u.get_coord() - n.get_coord()),
        #     potential_neighbors))  # todo this is costly
        for pec in self.pe_constructors:
            pec_max_dist = pec.max_dist
            for neighbor, d in zip(potential_neighbors, neighbors_distances):
                if d <= pec_max_dist:
                    new_pe: PairEnergy = pec.instanciate(u, neighbor)
                    self.pe_per_point[u].append(new_pe)
                    self.pe_per_point[neighbor].append(new_pe)

    def remove_point(self, u: Point):
        # remove unit energy
        self.ue_per_point.pop(u)

        # remove all pair energies from connected points
        for pe in self.pe_per_point[u]:
            # find connected point
            connected_point = pe.get_other_point(u)
            # remove that pe from that connected point
            self.pe_per_point[connected_point].remove(pe)

        self.pe_per_point.pop(u)

    def __copy__(self):
        new = EnergyGraph(
            unit_energies_constructors=self.ue_constructors,
            pair_energies_constructors=self.pe_constructors
        )

        new.pe_per_point = {point: pe_list.copy() for point, pe_list in self.pe_per_point.items()}
        new.ue_per_point = {point: ue_list.copy() for point, ue_list in self.ue_per_point.items()}
        return new

    def copy(self):
        return self.__copy__()

    def total_energy(self, points_set: PointsSet, force_update=False) -> float:
        return self.compute_subset(subset=points_set)

    def compute_subset(self, subset: Union[Set[Point], List[Point], PointsSet], force_update: bool = False,
                       energy_combinator: EnergyCombinationModel = None, return_vector=False):
        config_energy_vector: ConfigurationEnergyVector = {k: [] for k in self.energies_keys}
        for u in subset:
            energy_vector: PointEnergyVector = {k: 0 for k in self.energies_keys}
            for ue in self.ue_per_point[u]:
                energy_vector[ue.constructor.name] = ue.compute(lazy=not force_update)

            pe_per_kind: Dict[PairEnergyConstructor, List[float]] = {}
            for pe in self.pe_per_point[u]:
                pe_kind = pe.constructor
                if pe_kind in pe_per_kind:
                    pe_per_kind[pe_kind].append(pe.compute())
                else:
                    pe_per_kind[pe_kind] = [pe.compute()]

            for pe_kind, values in pe_per_kind.items():
                energy_vector[pe_kind.name] = pe_kind.reduce_point_interactions(values)

            append_to_dict_of_list(config_energy_vector, energy_vector)

        if return_vector:
            return config_energy_vector

        if energy_combinator is None:
            energy = np.sum([np.sum(values) for values in config_energy_vector.values()])
        else:
            energy = energy_combinator.compute(config_energy_vector)

        return energy

    def compute_delta(self, points_set: PointsSet, pert: Perturbation,
                      energy_combinator: EnergyCombinationModel = None):
        # get all interacting points

        added_set = set()
        removed_set = set()
        if pert.addition is not None:
            if type(pert.addition) is list:
                added_set |= set(pert.addition)
            else:
                added_set |= {pert.addition}
        if pert.removal is not None:
            if type(pert.removal) is list:
                removed_set |= set(pert.removal)
            else:
                removed_set |= {pert.removal}

        if pert.addition is not None:
            if type(pert.addition) is list:
                connected_points_add = \
                    set().union(*[points_set.get_potential_neighbors(a, radius=self.max_interaction_dist)
                                  for a in pert.addition]) - removed_set
            else:
                connected_points_add = \
                    points_set.get_potential_neighbors(pert.addition, radius=self.max_interaction_dist) - removed_set
        else:
            connected_points_add = set()

        if pert.removal is not None:
            if type(pert.removal) is list:
                connected_points_rem = \
                    set().union(*[points_set.get_potential_neighbors(r, radius=self.max_interaction_dist)
                                  for r in pert.removal]) - added_set
            else:
                connected_points_rem = \
                    points_set.get_potential_neighbors(pert.removal, radius=self.max_interaction_dist) - added_set
        else:
            connected_points_rem = set()

        # points that are not modified
        unchanged_subset = connected_points_add | connected_points_rem
        # initial state of the subset
        initial_subset = unchanged_subset | removed_set
        new_subset = unchanged_subset | added_set

        # compute the previous contribution of the pe
        # the removed point stills contributes here as it is part of it
        initial_subset_energy = self.compute_subset(
            subset=initial_subset,
            energy_combinator=energy_combinator)

        # add and remove points
        added_points = None
        if pert.addition is not None:
            if type(pert.addition) is list:
                added_points = pert.addition
            else:
                added_points = [pert.addition]
            for a in added_points:
                points_set.add(a)
                self.add_point(a, points_set)
        removed_points = None
        if pert.removal is not None:
            if type(pert.removal) is list:
                removed_points = pert.removal
            else:
                removed_points = [pert.removal]
            for r in removed_points:
                points_set.remove(r)
                self.remove_point(r)

        # compute energy
        new_subset_energy = self.compute_subset(
            subset=new_subset - removed_set,
            energy_combinator=energy_combinator)

        # revert to initial state
        if pert.addition is not None:
            for a in added_points:
                points_set.remove(a)
                self.remove_point(a)
        if pert.removal is not None:
            for r in removed_points:
                points_set.add(r)
                self.add_point(r, points_set)

        return new_subset_energy - initial_subset_energy

    def energies_as_dict(self, points_set: PointsSet):

        pe_keys = [pec.name for pec in self.pe_constructors]
        energies_per_point: Dict[Point, Dict[str, float]] = {u: {k: 0 for k in self.energies_keys} for u in points_set}
        energies_per_type: Dict[str, List[float]] = {k: [] for k in self.energies_keys}

        for u in points_set:
            unit_energies = {}
            for ue in self.ue_per_point[u]:
                name = ue.constructor.name
                value = ue.compute()
                unit_energies[name] = value

                energies_per_type[name].append(value)

            pe_per_kind: Dict[PairEnergyConstructor, List[float]] = {}
            for pe in self.pe_per_point[u]:

                pe_kind = pe.constructor
                if pe_kind in pe_per_kind:
                    pe_per_kind[pe_kind].append(pe.compute())
                else:
                    pe_per_kind[pe_kind] = [pe.compute()]

                name = pe.constructor.name
                value = pe.compute()

                if name in energies_per_type:
                    energies_per_type[name].append(value)
                else:
                    energies_per_type[name] = [value]

            pair_energies = {pec.name: pec.reduce_point_interactions(values) for pec, values in
                             pe_per_kind.items()}
            for pek in pe_keys:
                if pek not in pair_energies:
                    pair_energies[pek] = 0
            for name, value in pair_energies.items():
                energies_per_type[name].append(value)

            energies_per_point[u] = {**unit_energies, **pair_energies}

        return energies_per_point, energies_per_type

    def get_interacting_points(self, u: Point):
        assert u in self.pe_per_point
        return [pe.get_other_point(u) for pe in self.pe_per_point[u]]

    def check_integrity(self):
        """
        checks if the graph is well defined
        :return: True if correct
        :rtype:
        """
        for p, ue_list in self.ue_per_point.items():
            for ue in ue_list:
                assert ue.point is p
        for p, pe_list in self.pe_per_point.items():
            for pe in pe_list:
                assert pe.point_1 is p or pe.point_2 is p
                assert pe.point_1 is not pe.point_2
                assert pe.point_1 in self.pe_per_point
                assert pe.point_2 in self.pe_per_point
                assert pe in self.pe_per_point[pe.point_1]
                assert pe in self.pe_per_point[pe.point_2]
