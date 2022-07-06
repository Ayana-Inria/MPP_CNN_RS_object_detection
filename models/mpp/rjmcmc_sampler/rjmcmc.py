import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple, Union

import numpy as np
from tqdm import tqdm

from models.mpp.custom_types.energy import EnergyCombinationModel
from models.mpp.custom_types.rjmcmc import RJMCMCStateSummary
from models.mpp.point_set.energy_point_set import EPointsSet
from models.mpp.rjmcmc_sampler.kernels.base_kernels import Kernel
from models.mpp.rjmcmc_sampler.stopping import StoppingCondition

EPS = 1e-16


class RJMCMCTimer:

    def __init__(self):
        self.last_tick = None
        self.timings: Dict[str, List[float]] = {'total': [], 'n_points': []}
        self.start_tick = None

    def start_step(self):
        self.start_tick = time.perf_counter()
        self.last_tick = self.start_tick

    def checkpoint(self, key):
        now = time.perf_counter()
        try:
            self.timings[key].append(now - self.last_tick)
        except KeyError:
            self.timings[key] = [now - self.last_tick]
        self.last_tick = now

    def end_step(self, n_points):
        self.timings['total'].append(time.perf_counter() - self.start_tick)
        self.timings['n_points'].append(n_points)

    def show_results(self):
        points_number = np.array(self.timings['n_points'])
        for k, l in self.timings.items():
            if k != 'n_points':
                l = np.array(l)
                print(f"{k:20}: "
                      f"{np.mean(l) / 1000:.2e} s | "
                      f"{np.mean((l[points_number > 0] / points_number[points_number > 0])) / 1000:.2e} s/point")


@dataclass
class RJMCMC:
    t0: float
    kernels: List[Kernel]
    p_kernels: List[float]
    initial_state: EPointsSet
    stopping_condition: StoppingCondition
    rng: np.random.Generator
    energy_combinator: EnergyCombinationModel = None
    t_target: float = 0
    sampling_rule: Callable[[int], bool] = None
    do_annealing = True
    alpha_t: float = None
    verbose: int = 0

    def __post_init__(self):

        assert len(self.kernels) == len(self.p_kernels)
        assert (not self.do_annealing) or (self.alpha_t is not None)
        assert self.t0 >= self.t_target
        self._timer = RJMCMCTimer()
        self._temp: float = self.t0
        self._iter: int = 0
        self._state_log: List[EPointsSet] = [self.initial_state]
        self._state_summaries: List[RJMCMCStateSummary] = \
            [RJMCMCStateSummary(n_points=len(self.initial_state), iter=self._iter)]

        self._pbar = tqdm(desc="RJMCMC",position=0,leave=True) if self.verbose > 0 else None

    def __copy__(self):
        raise NotImplementedError

    def step(self, return_state=False):
        if self.stopping_condition.do_stop(self._state_summaries):
            raise StopIteration
        else:
            self._timer.start_step()
            k1: Kernel = self.rng.choice(self.kernels, p=self.p_kernels)
            self._timer.checkpoint('sample_kernel')

            x0 = self._state_log[-1]
            # generate perturbation
            u1 = k1.sample_perturbation(x0.points, self.rng)
            self._timer.checkpoint('sample_perturbation')
            # compute energies
            energy_x0 = self._state_summaries[-1].energy
            if energy_x0 is None:
                energy_x0 = x0.total_energy()  # at first iter energy has to be computed
            energy_delta = x0.energy_delta(u1, energy_combinator=self.energy_combinator)
            # energy_x1 = x1.total_energy()
            self._timer.checkpoint('compute_energy')
            energy_x1 = energy_x0 + energy_delta

            # compute green ratio for step 1
            log_alpha_1 = (-energy_delta / self._temp) \
                          + np.log(k1.backward_probability(x0.points, u1) + EPS) \
                          - np.log(k1.forward_probability(x0.points, u1) + EPS)
            # alpha_1 = np.exp(-energy_delta / temp) * k1.backward_probability(x0.points, u1) / k1.forward_probability(
            #     x0.points, u1)
            self._timer.checkpoint('compute_alpha')

            # accepted = rng.random() < alpha_1
            accepted = np.log(self.rng.random() + EPS) < log_alpha_1
            with warnings.catch_warnings(): # alpha_1 is only used for display, any overflow is not critical
                warnings.simplefilter("ignore")
                alpha_1 = np.exp(log_alpha_1)
            if accepted:
                x1 = x0.apply_perturbation(u1, inplace=True)
            else:
                x1 = x0
            self._timer.checkpoint('apply_perturbation')

            x1_n_points = len(x1)

            new_state_summary = RJMCMCStateSummary(
                iter=self._iter,
                temperature=self._temp,
                energy=energy_x1 if accepted else energy_x0,
                n_points=x1_n_points,
                kernel=k1.__class__,
                move_accepted=accepted,
                alpha=alpha_1,
                initial_energy=energy_x0,
                proposed_energy=energy_x1
            )
            self._state_summaries.append(new_state_summary)

            if self.sampling_rule is not None and self.sampling_rule(self._iter):
                self._state_log.append(x1.copy())
            else:
                self._state_log[0] = x1

            self._timer.checkpoint('log')

            if self._pbar is not None:
                self._pbar.update(1)
                self._pbar.set_postfix(
                    temp=self._temp,
                    n_points=x1_n_points,
                    k_type=type(k1).__name__,
                    r=alpha_1,
                    apprv=str(accepted),
                    energy=energy_x1 if accepted else energy_x0,
                    stopInfo=self.stopping_condition.print(states=self._state_summaries)
                )
            self._timer.checkpoint('print')
            self._iter += 1
            if self.do_annealing and self._temp > self.t_target:
                self._temp *= self.alpha_t
            self._timer.end_step(n_points=x1_n_points)

            if return_state:
                return new_state_summary, x1.copy()
            return new_state_summary

    def __iter__(self):
        return self

    def __next__(self):
        return self.step()

    def run(self, show_timing=False) -> Tuple[Union[List[EPointsSet], EPointsSet], List[RJMCMCStateSummary]]:
        for _ in self.__iter__():
            pass
        if show_timing:
            print("Timings--------------------------------------------------------------------------")
            self._timer.show_results()

        return_states = self._state_log

        return return_states, self._state_summaries

    def get_timings(self):
        return self._timer

    def get_state_log(self):
        return self._state_log

# def rjmcmc(t0: float, kernels: List[Kernel], p_kernels: List[float], initial_state: EPointsSet,
#            stopping_condition: StoppingCondition, rng: np.random.Generator,
#            energy_combinator: EnergyCombinationModel = None, t_target: float = 0,
#            log_states: bool = False, sampling_rule: Callable[[int], bool] = None,
#            do_annealing=True, alpha_t: float = None, verbose=0, return_timings=False):
#     assert len(kernels) == len(p_kernels)
#     assert (not do_annealing) or (alpha_t is not None)
#     assert t0 >= t_target
#     temp = t0
#     raise DeprecationWarning
#
#     timer = RJMCMCTimer()
#
#     if log_states:
#         logging.warning('point sets will be logged as copies, this can slow down the execution')
#     i = 0
#     last_state = [initial_state]
#     state_summaries: List[RJMCMCStateSummary] = [RJMCMCStateSummary(n_points=len(initial_state), iter=i)]
#
#     pbar = tqdm() if verbose > 0 else None
#
#     while not stopping_condition.do_stop(state_summaries):
#         timer.start_step()
#         k1: Kernel = rng.choice(kernels, p=p_kernels)
#         timer.checkpoint('sample_kernel')
#
#         x0 = last_state[-1]
#         # generate perturbation
#         u1 = k1.sample_perturbation(x0.points, rng)
#         timer.checkpoint('sample_perturbation')
#         # compute energies
#         energy_x0 = state_summaries[-1].energy
#         if energy_x0 is None:
#             energy_x0 = x0.total_energy()  # at first iter energy has to be computed
#         energy_delta = x0.energy_delta(u1, energy_combinator=energy_combinator)
#         # energy_x1 = x1.total_energy()
#         timer.checkpoint('compute_energy')
#         energy_x1 = energy_x0 + energy_delta
#
#         # compute green ratio for step 1
#         log_alpha_1 = (-energy_delta / temp) \
#                       + np.log(k1.backward_probability(x0.points, u1) + EPS) \
#                       - np.log(k1.forward_probability(x0.points, u1) + EPS)
#         # alpha_1 = np.exp(-energy_delta / temp) * k1.backward_probability(x0.points, u1) / k1.forward_probability(
#         #     x0.points, u1)
#         timer.checkpoint('compute_alpha')
#
#         # accepted = rng.random() < alpha_1
#         accepted = np.log(rng.random() + EPS) < log_alpha_1
#         alpha_1 = np.exp(log_alpha_1)
#         if accepted:
#             x1 = x0.apply_perturbation(u1, inplace=True)
#         else:
#             x1 = x0
#         timer.checkpoint('apply_perturbation')
#
#         x1_n_points = len(x1)
#
#         state_summaries.append(RJMCMCStateSummary(
#             iter=i,
#             temperature=temp,
#             energy=energy_x1 if accepted else energy_x0,
#             n_points=x1_n_points,
#             kernel=k1.__class__,
#             move_accepted=accepted,
#             alpha=alpha_1,
#             initial_energy=energy_x0,
#             proposed_energy=energy_x1
#         ))
#
#         if log_states or (sampling_rule is not None and sampling_rule(i)):
#             last_state.append(x1.copy())
#         else:
#             last_state[0] = x1
#
#         timer.checkpoint('log')
#
#         if pbar is not None:
#             pbar.update(1)
#             pbar.set_postfix(
#                 temp=temp,
#                 n_points=x1_n_points,
#                 k_type=type(k1).__name__,
#                 r=alpha_1,
#                 apprv=str(accepted),
#                 energy=energy_x1 if accepted else energy_x0,
#                 stopInfo=stopping_condition.print(states=state_summaries)
#             )
#
#         timer.checkpoint('print')
#
#         i += 1
#         if do_annealing and temp > t_target:
#             temp *= alpha_t
#
#         timer.end_step(n_points=x1_n_points)
#
#     if verbose > 0:
#         print()
#     if return_timings:
#         print("Timings--------------------------------------------------------------------------")
#         timer.show_results()
#
#     last_state = last_state if (log_states or sampling_rule is not None) else last_state[0]
#     if return_timings:
#         return last_state, state_summaries, timer.timings
#     return last_state, state_summaries
