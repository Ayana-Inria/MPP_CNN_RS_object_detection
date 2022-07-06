from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module

from models.mpp.custom_types.energy import EnergyCombinationModel, ConfigurationEnergyVector
from models.mpp.energies.combination.base import WeightModel


@dataclass
class MLPEnergyCombinator(EnergyCombinationModel):
    model: Module
    energy_names: List[str]
    raw_energy: bool = False

    @torch.no_grad()
    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        if len(vectors[self.energy_names[0]]) == 0:
            return 0.0
        tensor = torch.tensor(np.stack([vectors[k] for k in self.energy_names], axis=-1)).float()
        if not self.raw_energy:
            return float(np.sum(2 * self.model.forward(tensor).detach().numpy() - 1))
        else:
            return float(self.model.forward(tensor).detach().numpy().sum())


class MLPEnergyModel(Module, WeightModel):
    def __init__(self, energy_names: List[str], hidden_features=8, hidden_layers=2, raw_energy=False):
        self.energy_names = energy_names
        super(MLPEnergyModel, self).__init__()
        self.raw_energy = raw_energy
        layers = [
            nn.Linear(in_features=len(energy_names), out_features=hidden_features),
            nn.ReLU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(in_features=hidden_features, out_features=hidden_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_features, out_features=1))
        if not self.raw_energy:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        if not self.raw_energy:
            return torch.sum(2 * self.layers.forward(x.float()) - 1)
        else:
            return torch.sum(self.layers.forward(x.float()))

    def regularisation_term(self, E_plus, E_minus):
        if self.raw_energy:
            return torch.square(E_minus) + torch.square(E_plus)
        return torch.tensor(0.0)

    def get_decision_function(self):
        if not self.raw_energy:
            def fun(vector: np.ndarray):
                return 2 * self.layers.forward(torch.tensor(vector).float()).detach().numpy() - 1
        else:
            def fun(vector: np.ndarray):
                return self.layers.forward(torch.tensor(vector).float()).detach().numpy()

        return fun

    def get_energy_combination_function(self) -> EnergyCombinationModel:
        return MLPEnergyCombinator(model=self.layers, raw_energy=self.raw_energy, energy_names=self.energy_names)

    def as_dict(self):
        return {}
