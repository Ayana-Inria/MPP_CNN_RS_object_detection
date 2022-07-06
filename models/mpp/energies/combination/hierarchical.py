from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module, functional

from models.mpp.custom_types.energy import EnergyCombinationModel, ConfigurationEnergyVector
from models.mpp.energies.combination.base import WeightModel


@dataclass
class HierarchicalEnergyCombinator(EnergyCombinationModel):
    weights_data: np.ndarray
    weights_prior: np.ndarray
    data_prior_weights: np.ndarray
    detection_threshold: float
    bias: float = 0.0

    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        indicator = np.less_equal(vectors['PositionEnergy'], self.detection_threshold)
        data_term = np.multiply(self.weights_data[0], vectors['PositionEnergy']) + \
                    indicator * np.multiply(self.weights_data[1], vectors['ShapeEnergy'])
        prior_term = indicator * (
                np.multiply(self.weights_prior[0], vectors['RectangleOverlapEnergy']) +
                np.multiply(self.weights_prior[1], vectors['ShapeAlignmentEnergy']) +
                np.multiply(self.weights_prior[2], vectors['AreaPriorEnergy'])
        )

        return float(
            np.sum(self.data_prior_weights[0] * data_term + self.data_prior_weights[1] * prior_term + self.bias))


@dataclass
class ManualHierarchicalEnergyCombinator(EnergyCombinationModel):
    weights_dict: Dict[str, float]
    indicator_energy: str
    detection_threshold: float = 0.0

    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        indicator = np.less_equal(vectors[self.indicator_energy], self.detection_threshold)
        indicator_energy = np.multiply(self.weights_dict[self.indicator_energy], vectors[self.indicator_energy])
        energies = np.sum(
            [np.multiply(w, vectors[k]) for k, w in self.weights_dict.items() if k != self.indicator_energy],
            axis=0)

        return float(np.sum(indicator_energy + indicator * energies))


class HierarchicalEnergyModel(Module, WeightModel):
    def __init__(self, threshold: float, learn_bias=False):
        super(HierarchicalEnergyModel, self).__init__()

        self.data_prior_weight = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self.data_weight = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self.prior_weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
        self.threshold_detection = threshold
        if learn_bias:
            self.bias = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        else:
            self.bias = torch.tensor(0.0, requires_grad=True)
        # self.threshold_detection = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, x: Tensor):
        prior_weight_sm = functional.softmax(self.prior_weight, dim=0)
        data_weight_sm = functional.softmax(self.data_weight, dim=0)
        data_prior_weight_sm = functional.softmax(self.data_prior_weight, dim=0)

        indicator = torch.less_equal(x[:, 0], self.threshold_detection)
        data_term = data_weight_sm[0] * x[:, 0] + indicator * data_weight_sm[1] * x[:, 1]
        prior_term = indicator * (
                prior_weight_sm[0] * x[:, 2] +
                prior_weight_sm[1] * x[:, 3] +
                prior_weight_sm[2] * x[:, 4]
        )
        return torch.sum(data_prior_weight_sm[0] * data_term + data_prior_weight_sm[1] * prior_term + self.bias)

    def regularisation_term(self, **kwargs):
        prior_weight_sm = functional.softmax(self.prior_weight, dim=0)
        data_weight_sm = functional.softmax(self.data_weight, dim=0)
        data_prior_weight_sm = functional.softmax(self.data_prior_weight, dim=0)

        return torch.square(1 - data_prior_weight_sm[0]) + \
               torch.square(1 - data_prior_weight_sm[1]) + \
               torch.square(1 - data_weight_sm[0]) + \
               torch.square(1 - data_weight_sm[1]) + \
               torch.square(1 - prior_weight_sm[0]) + \
               torch.square(1 - prior_weight_sm[1]) + \
               torch.square(1 - prior_weight_sm[2])

    def get_energy_combination_function(self) -> EnergyCombinationModel:
        return HierarchicalEnergyCombinator(
            weights_data=functional.softmax(self.data_weight, dim=0).detach().numpy(),
            weights_prior=functional.softmax(self.prior_weight, dim=0).detach().numpy(),
            data_prior_weights=functional.softmax(self.data_prior_weight, dim=0).detach().numpy(),
            detection_threshold=float(self.threshold_detection),
            bias=float(self.bias.detach().cpu())
        )

    def get_decision_function(self):
        weights_data = functional.softmax(self.data_weight, dim=0).detach().numpy()
        weights_prior = functional.softmax(self.prior_weight, dim=0).detach().numpy()
        data_prior_weights = functional.softmax(self.data_prior_weight, dim=0).detach().numpy()
        detection_threshold = float(self.threshold_detection)
        bias = float(self.bias.detach().cpu())

        def fun(vector: np.ndarray):
            indicator = np.less_equal(vector[:, 0], detection_threshold)
            data_term = weights_data[0] * vector[:, 0] + indicator * weights_data[1] * vector[:, 1]
            prior_term = indicator * (
                    weights_prior[0] * vector[:, 2] +
                    weights_prior[1] * vector[:, 3] +
                    weights_prior[2] * vector[:, 4])
            return data_prior_weights[0] * data_term + data_prior_weights[1] * prior_term + bias

        return fun

    def as_dict(self):
        weights_data = functional.softmax(self.data_weight, dim=0).detach().numpy()
        weights_prior = functional.softmax(self.prior_weight, dim=0).detach().numpy()
        data_prior_weights = functional.softmax(self.data_prior_weight, dim=0).detach().numpy()
        detection_threshold = float(self.threshold_detection)
        bias = float(self.bias.detach().cpu())
        return {
            'data_weight': data_prior_weights[0],
            'prior_weight': data_prior_weights[1],
            'PositionEnergy_indicator_threshold': detection_threshold,
            'PositionEnergy_data_weight': weights_data[0],
            'ShapeEnergy_data_weight': weights_data[1],
            'RectangleOverlapEnergy_prior_weight': weights_prior[0],
            'ShapeAlignmentEnergy_prior_weight': weights_prior[1],
            'AreaPriorEnergy_prior_weight': weights_prior[2],
            'bias': bias
        }
