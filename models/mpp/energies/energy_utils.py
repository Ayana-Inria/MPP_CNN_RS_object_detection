from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import List, Union, Tuple

import numpy as np
from numpy.random import Generator

from base.shapes.rectangle import Rectangle
from models.mpp.custom_types.image_w_maps import ImageWMaps
from models.mpp.energies.base_energies import UnitEnergyConstructor, PairEnergyConstructor
from models.mpp.point_set.energy_point_set import EPointsSet


class EnergySetup(ABC):
    @property
    @abstractmethod
    def energy_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def make_energies(self, image_data: ImageWMaps
                      ) -> Tuple[List[UnitEnergyConstructor], List[PairEnergyConstructor]]:
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, image_configs: List[ImageWMaps], rng: Generator, save_path: str = None):
        raise NotImplementedError

    @abstractmethod
    def load_calibration(self, save_dir: str):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def detection_threshold(self) -> float:
        raise NotImplementedError


def names_from_energies(energies: List[Union[PairEnergyConstructor, UnitEnergyConstructor]]) -> List[str]:
    names = []
    for e in energies:
        names.append(e.name)

    return names


def compute_energy_vector(points: Union[List[Rectangle], EPointsSet], unit_energies: List[UnitEnergyConstructor],
                          pair_energies: List[PairEnergyConstructor], support_shape, energy_names: List[str],
                          return_names=False):
    if type(points) is not EPointsSet:
        points_set = EPointsSet(
            points=points,
            support_shape=support_shape,
            unit_energies_constructors=unit_energies,
            pair_energies_constructors=pair_energies,
        )
    else:
        points_set = points

    energies_per_type = points_set.energy_graph.compute_subset(subset=points_set, return_vector=True)

    vector = np.array([energies_per_type[k] for k in energy_names]).T
    if return_names:
        return vector, list(energies_per_type.keys())
    return vector


def compute_many_energy_vectors(configurations: List[List[Rectangle]], image_config: ImageWMaps,
                                ue: List[UnitEnergyConstructor], pe: List[PairEnergyConstructor],
                                energy_names: List[str],
                                multiprocess=True):
    compute_non_scaled_energy_vector = partial(
        compute_energy_vector, unit_energies=ue, pair_energies=pe,
        energy_names=energy_names,
        support_shape=image_config.detection_map.shape[:2], return_names=False)
    if multiprocess:
        with Pool() as p:
            energy_vectors = p.map(compute_non_scaled_energy_vector, configurations)
    else:
        energy_vectors = list(map(compute_non_scaled_energy_vector, configurations))
    return np.concatenate(energy_vectors, axis=0)
