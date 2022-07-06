from dataclasses import dataclass
from typing import Union, Type


@dataclass
class RJMCMCStateSummary:
    iter: int
    n_points: int
    temperature: float = None
    energy: Union[None, float] = None
    kernel: Union[None, Type] = None
    move_accepted: Union[None, bool] = None
    alpha: Union[None, float] = None
    initial_energy: Union[None, float] = None
    proposed_energy: Union[None, float] = None