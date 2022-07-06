from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module, functional

from models.mpp.custom_types.energy import EnergyCombinationModel, ConfigurationEnergyVector
from models.mpp.energies.combination.base import WeightModel
from models.mpp.energies.energy_utils import ENERGY_NAMES


@dataclass
class LinearEnergyCombinator(EnergyCombinationModel):
    weights: np.ndarray
    energy_names: List[str]
    bias: float = 0.0

    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        return float(
            np.sum(
                [np.sum(np.multiply(self.weights[i], vectors[k])) + self.bias if len(vectors[k]) > 0 else 0 for i, k in
                 enumerate(self.energy_names)]))


class LinearEnergyModel(Module, WeightModel):
    def regularisation_term(self, **kwargs):
        return torch.tensor(0.0)

    def __init__(self, learn_bias: bool = False):
        super(LinearEnergyModel, self).__init__()
        self.weights = nn.Parameter(torch.tensor([1.0] * len(ENERGY_NAMES), requires_grad=True))
        if learn_bias:
            self.bias = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        else:
            self.bias = torch.tensor(0.0, requires_grad=True)

    def forward(self, x: Tensor):
        norm_weights = functional.softmax(self.weights, dim=0)
        return torch.sum(torch.sum(norm_weights * x, dim=-1) + self.bias)

    def get_np_weights(self):
        norm_weights = functional.softmax(self.weights, dim=0)
        return norm_weights.detach().cpu().numpy()

    def get_energy_combination_function(self) -> EnergyCombinationModel:
        weights_np = self.get_np_weights()

        return LinearEnergyCombinator(weights_np, bias=float(self.bias.detach().cpu()), energy_names=ENERGY_NAMES)

    def get_decision_function(self):
        weights_np = self.get_np_weights()
        bias = float(self.bias.detach().cpu())

        def fun(vector: np.ndarray):
            return np.dot(vector, weights_np) + bias

        return fun

    def as_dict(self):
        np_weights = self.get_np_weights()
        return {
            **{k + '_weight': np_weights[i] for i, k in enumerate(ENERGY_NAMES)},
            'bias': float(self.bias.detach().cpu())
        }
