from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from models.mpp.custom_types.rjmcmc import RJMCMCStateSummary


class StoppingCondition:
    @abstractmethod
    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        pass

    @abstractmethod
    def print(self, states: List[RJMCMCStateSummary]) -> str:
        pass


class CompositeStopping(StoppingCondition):
    def __init__(self, sub_conditions: List[StoppingCondition]):
        self.sub_conditions = sub_conditions

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        for s in self.sub_conditions:
            if not s.do_stop(states):
                return False
        return True

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        for s in self.sub_conditions:
            if not s.do_stop(states):
                return s.print(states)
        return 'STOP'


class StopOnMaxIter(StoppingCondition):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        return states[-1].iter >= self.max_iter

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        return f"{states[-1].iter} < {self.max_iter}"


class StopOnRejects(StoppingCondition):
    def __init__(self, max_rejects: int):
        self.max_rejects = max_rejects
        self._cumulative_rejects = 0

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        self._cumulative_rejects = 0 if states[-1].move_accepted else self._cumulative_rejects + 1
        return self._cumulative_rejects >= self.max_rejects

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        return f"{self._cumulative_rejects} < {self.max_rejects}"


class StopOnDeltaU(StoppingCondition):
    def __init__(self, epsilon: float = 1e-2, consecutive: int = 10):
        self.epsilon = epsilon
        self.consecutive = consecutive
        self.non_zero_deltaU_list = []

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        if states[-1].move_accepted:
            if states[-1].proposed_energy != 0:
                self.non_zero_deltaU_list.append(
                    abs((states[-1].proposed_energy - states[-1].initial_energy) / states[-1].proposed_energy))
        if len(self.non_zero_deltaU_list) < self.consecutive:
            return False
        else:
            return np.max(self.non_zero_deltaU_list[-self.consecutive:]) < self.epsilon

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        if len(self.non_zero_deltaU_list) < self.consecutive:
            if len(self.non_zero_deltaU_list) > 0:
                return f"{np.max(self.non_zero_deltaU_list):.2e} >= {self.epsilon:.2e}"
            else:
                return f"{np.nan} >= {self.epsilon:.2e}"
        else:
            return f"{np.max(self.non_zero_deltaU_list[-self.consecutive:]):.2e} >= {self.epsilon:.2e}"


class StopOnApprovalRate(StoppingCondition):
    def __init__(self, min_rate, smoothing=100):
        self.min_rate = min_rate
        self.smoothing = smoothing
        self.rate = 1.0

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        if len(states) > self.smoothing:
            self.rate = np.mean([s.move_accepted for s in states[-self.smoothing:]])
            return self.rate < self.min_rate
        else:
            return False

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        return f"{self.rate:.2f}>{self.min_rate:.2f}"


@dataclass
class CustomStopping(StoppingCondition):
    """
    Stops if |delta u/u| on acceted changes < delta_u_threshold and last added point was at least
    last_added_point iter ago or
    iter >= max_iter
    """
    max_iter: int
    min_iter: int
    delta_u_threshold: float = 1e-3
    delta_u_n_avg: int = 100
    last_delta_point: int = 1000
    update_interval: int = 1000
    _last_delta_u = None

    def do_stop(self, states: List[RJMCMCStateSummary]) -> bool:
        i = states[-1].iter
        if i >= self.max_iter:
            return True
        if i <= self.min_iter:
            return False

        if i % self.update_interval == 0:
            if i <= self.last_delta_point:
                return False
            n_points = np.array([s.n_points for s in states[-self.last_delta_point:]])
            accepted = np.array([s.move_accepted for s in states[1:]])
            if np.sum(accepted) <= self.delta_u_n_avg:
                return False
            energies = np.array([s.energy for s in states[1:]])
            accepted_energies = energies[accepted][-self.delta_u_n_avg:]
            delta_u = np.mean(np.abs(np.diff(accepted_energies) / accepted_energies[:-1]))
            self._last_delta_u = delta_u
            if np.all(n_points == n_points[-1]):
                if delta_u <= self.delta_u_threshold:
                    return True

        return False

    def print(self, states: List[RJMCMCStateSummary]) -> str:
        if self._last_delta_u is not None:
            return f'delta_u/u={self._last_delta_u:.0e}'
        return '??'
