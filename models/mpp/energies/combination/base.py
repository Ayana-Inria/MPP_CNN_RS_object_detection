from abc import ABC, abstractmethod

from torch import Tensor

from models.mpp.custom_types.energy import EnergyCombinationModel


class WeightModel(ABC):
    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def as_dict(self):
        raise NotImplementedError

    @abstractmethod
    def get_energy_combination_function(self) -> EnergyCombinationModel:
        raise NotImplementedError

    @abstractmethod
    def get_decision_function(self):
        raise NotImplementedError

    @abstractmethod
    def regularisation_term(self, **kwargs):
        raise NotImplementedError