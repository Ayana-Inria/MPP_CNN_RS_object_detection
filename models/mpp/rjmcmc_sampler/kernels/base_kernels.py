from abc import abstractmethod
from typing import Tuple, Callable, Union

import numpy as np

from models.mpp.custom_types.perturbation import Perturbation
from models.mpp.point_set.point_set import PointsSet
from models.mpp.rjmcmc_sampler.kernels.shape_samplers import ShapeSampler

Measure = Callable[[Tuple[int, int]], float]


class Kernel:
    """
    abstract class for proposition kernels
    """

    @abstractmethod
    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        pass

    @abstractmethod
    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        pass

    @abstractmethod
    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        pass


class BirthKernel(Kernel):
    def __init__(self, p_birth: float, p_death: float, sampler: ShapeSampler, intensity: Union[float, np.ndarray],
                 verbose=0):
        """

        Parameters
        ----------
        p_birth : birth kernel choice probability
        p_death : death kernel choice probability
        sampler : method that samples new point proposals
        intensity : intensity of the underlying point process
        verbose : will say things if > 0, maybe
        """
        self.p_birth = p_birth
        self.p_death = p_death
        self.sampler = sampler
        self.verbose = verbose
        self.global_intensity = float(np.sum(intensity))

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        new_born = self.sampler.sample(x)

        return Perturbation(self.__class__, addition=new_born)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        return (self.p_birth * self.sampler.get_point_density(u.addition)) / self.global_intensity

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x) + 1
        if n == 0:
            return self.p_death  # from a zero point state,
        return self.p_death / n

    @property
    def p_kernel(self) -> float:
        return self.p_birth

    def __repr__(self):
        return f'BirthKernel'


class DeathKernel(Kernel):
    def __init__(self, p_birth: float, p_death: float, sampler: ShapeSampler, intensity: Union[float, np.ndarray],
                 verbose=0):
        """

        Parameters
        ----------
        p_birth : birth kernel choice probability
        p_death : death kernel choice probability
        sampler : method that samples new point proposals
        intensity : intensity of the underlying point process
        verbose : will say things if > 0, maybe
        """
        self.p_birth = p_birth
        self.p_death = p_death
        self.sampler = sampler
        self.verbose = verbose
        self.global_intensity = float(np.sum(intensity))
        self.verbose = verbose

    def sample_perturbation(self, x: PointsSet, rng: np.random.Generator) -> Perturbation:
        nx = len(x)
        if nx > 0:
            return Perturbation(type=self.__class__, removal=x.random_choice(rng))
        return Perturbation(type=self.__class__)

    def forward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        n = len(x)
        if n == 0:
            return self.p_death
        else:
            return self.p_death / n

    def backward_probability(self, x: PointsSet, u: Perturbation) -> float:
        assert u.type == self.__class__
        p_rem = u.removal
        if p_rem is None:
            # then the point process has zero points
            return self.p_death
        else:
            return (self.p_birth * self.sampler.get_point_density(p_rem)) / self.global_intensity

    @property
    def p_kernel(self) -> float:
        return self.p_death

    def __repr__(self):
        return f'DeathKernel'
