from abc import abstractmethod
from typing import Dict, List

ConfigurationEnergyVector = Dict[str, List[float]]
PointEnergyVector = Dict[str, float]


class EnergyCombinationModel:
    @abstractmethod
    def compute(self, vectors: ConfigurationEnergyVector) -> float:
        pass
