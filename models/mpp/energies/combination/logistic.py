from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module, functional

from models.mpp.custom_types.energy import EnergyCombinationModel, ConfigurationEnergyVector
from models.mpp.energies.combination.base import WeightModel
from utils.math_utils import sigmoid


@dataclass
class LogisticEnergyCombinator(EnergyCombinationModel):
    weights: np.ndarray
    bias: float
    energy_names: List[str]

    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        v_array = np.array([vectors[k] for k in self.energy_names]).T
        if len(v_array) == 0:
            return 0.0
        return float(
            np.sum(2 * sigmoid(np.sum(self.bias + self.weights * v_array, axis=-1)) - 1)
        )


class LogisticEnergyModel(Module, WeightModel):

    def __init__(self, energy_names: List[str], use_bias: bool = True):
        super(LogisticEnergyModel, self).__init__()
        self.energy_names = energy_names
        self.weights = nn.Parameter(torch.tensor([1.0] * len(self.energy_names), requires_grad=True))

        if use_bias:
            self.bias = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        else:
            self.bias = torch.tensor(0.0, requires_grad=False)

    def forward(self, x: Tensor):
        return torch.sum(2 * torch.sigmoid(torch.sum(self.bias + self.weights * x, dim=-1)) - 1)

    def get_np_weights(self):
        return self.weights.detach().cpu().numpy(), float(self.bias.detach().cpu())

    def as_dict(self):
        np_weights, np_bias = self.get_np_weights()
        return {
            **{k + '_weight': np_weights[i] for i, k in enumerate(self.energy_names)},
            'bias': np_bias
        }

    def get_energy_combination_function(self) -> EnergyCombinationModel:
        np_weights, np_bias = self.get_np_weights()
        return LogisticEnergyCombinator(
            weights=np_weights, bias=np_bias, energy_names=self.energy_names
        )

    def get_decision_function(self):
        np_weights, np_bias = self.get_np_weights()

        def fun(vector: np.ndarray):
            return 2 * sigmoid(np.sum(np_bias + np_weights * vector, axis=-1)) - 1

        return fun

    def regularisation_term(self, **kwargs):
        return torch.tensor(0.0)

#
# @dataclass
# class HierarchicalLogisticEnergyCombinator(EnergyCombinationModel):
#     energy_names: List[str]
#     weights_1: np.ndarray
#     weights_2: np.ndarray
#     bias_1: float
#     bias_2: float
#     indicator_bias: float
#     indicator_weight: float
#
#     def compute(self, vectors: ConfigurationEnergyVector) -> float:
#         v_array = np.array([vectors[k] for k in ENERGY_NAMES]).T
#         if len(v_array) == 0:
#             return 0.0
#
#         indicator = sigmoid(v_array[:, [0]] * self.indicator_weight + self.indicator_bias)
#         res = self.bias_1 + v_array[:, [0]] * self.weights_1 + indicator * (
#                 self.bias_2 + v_array[:, 1:] * self.weights_2)
#         return float(np.sum(2 * sigmoid(np.sum(res, axis=-1)) - 1))
#
#
# class HierarchicalLogisticEnergyModel(Module, WeightModel):
#     def __init__(self, use_bias: bool = True):
#         super(HierarchicalLogisticEnergyModel, self).__init__()
#         self.weights_1 = nn.Parameter(torch.tensor([1.0], requires_grad=True))
#         self.weights_2 = nn.Parameter(torch.tensor([1.0] * (len(ENERGY_NAMES) - 1), requires_grad=True))
#
#         if use_bias:
#             self.bias_1 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
#             self.bias_2 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
#         else:
#             self.bias_1 = torch.tensor(0.0, requires_grad=False)
#             self.bias_2 = torch.tensor(0.0, requires_grad=False)
#
#         self.indicator_bias = nn.Parameter(torch.tensor(0.0, requires_grad=True))
#         self.indicator_weight = nn.Parameter(torch.tensor(0.0, requires_grad=True))
#
#     def forward(self, x: Tensor, skip_sum=False):
#         # indicator = functional.relu(x[:, [0]] - self.indicator_bias) / (x[:, [0]] - self.indicator_bias)
#         indicator = torch.sigmoid(x[:, [0]] * self.indicator_weight + self.indicator_bias)
#         res = self.bias_1 + x[:, [0]] * self.weights_1 + indicator * (self.bias_2 + x[:, 1:] * self.weights_2)
#
#         if skip_sum:
#             return 2 * torch.sigmoid(torch.sum(res, dim=-1)) - 1
#         return torch.sum(2 * torch.sigmoid(torch.sum(res, dim=-1)) - 1)
#
#     def get_np_weights(self):
#         return np.concatenate([self.weights_1.detach().cpu().numpy(), self.weights_2.detach().cpu().numpy()], axis=0), \
#                np.array([float(self.bias_1.detach().cpu()), float(self.bias_2.detach().cpu())])
#
#     def as_dict(self):
#         np_weights, np_bias = self.get_np_weights()
#         i_b = float(self.indicator_bias.detach().cpu())
#         i_w = float(self.indicator_weight.detach().cpu())
#         return {
#             **{k + '_weight': np_weights[i] for i, k in enumerate(ENERGY_NAMES)},
#             **{f'bias_{i}': np_bias[i] for i in range(2)},
#             'indicator_bias': i_b,
#             'indicator_weight': i_w
#         }
#
#     def get_energy_combination_function(self) -> EnergyCombinationModel:
#         np_weights, np_bias = self.get_np_weights()
#         i_b = float(self.indicator_bias.detach().cpu())
#         i_w = float(self.indicator_weight.detach().cpu())
#         return HierarchicalLogisticEnergyCombinator(
#             weights_1=np_weights[[0]], weights_2=np_weights[1:],
#             bias_1=np_bias[0], bias_2=np_bias[1],
#             indicator_bias=i_b, indicator_weight=i_w
#         )
#
#     def get_decision_function(self):
#
#         def fun(vector: np.ndarray):
#             x = torch.tensor(vector)
#             return self.forward(x, skip_sum=True).detach().numpy()
#
#         return fun
#
#     def regularisation_term(self, **kwargs):
#         return torch.tensor(0.0)
