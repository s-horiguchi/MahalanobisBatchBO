import math
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .utils import BoundsType


class Objective(metaclass=ABCMeta):

    D: int
    bounds: BoundsType
    bounds_min: Tensor
    bounds_max: Tensor
    minimum: Optional[Tensor] = None
    fmin: Optional[float] = None
    sd_noise: float = 0.0
    name: str = "Objective"

    def normalize(self, X: Tensor) -> Tensor:
        """
        X_norm is in [-1, 1]^D
        """
        xmin = self.bounds_min
        xmax = self.bounds_max
        X_norm = (2 * X - xmin - xmax) / (xmax - xmin)
        return X_norm

    def unnormalize(self, X_norm: Tensor) -> Tensor:
        """
        X_norm is in [-1, 1]^D
        """
        xmin = self.bounds_min
        xmax = self.bounds_max
        X = 0.5 * X_norm * (xmax - xmin) + 0.5 * (xmin + xmax)
        return X

    def clip(self, X: Tensor) -> Tensor:
        lower = self.bounds_min.unsqueeze(0)
        upper = self.bounds_max.unsqueeze(0)
        return torch.max(torch.min(X, upper), lower)

    @abstractmethod
    def _objective_true(self, X: Tensor) -> Tensor:
        """
        X : (N, D)
        """
        raise NotImplementedError

    def objective_true(self, X: Tensor) -> Tensor:
        """
        X : (N, D)
        """
        X = torch.atleast_2d(X)
        assert X.shape[1] == self.D
        return self._objective_true(X)

    def objective_noisy(self, X: Tensor) -> Tensor:
        """
        X : (N, D)
        """
        obj = self.objective_true(X)
        if math.isclose(self.sd_noise, 0):
            noise = Tensor([0.0])
        else:
            noise = torch.normal(mean=0, std=self.sd_noise, size=obj.shape)
        return obj + noise

    def __str__(self):
        return f"{self.name}_D{self.D}"


# Example set of objective functions

class Identity(Objective):
    def __init__(
        self, D: int = 2, sd_noise: float = 0.0,
        bounds: Optional[Tensor] = None
    ):
        self.D = D
        if bounds is None:
            bounds = Tensor([(-1., +1.) for i in range(D)])
        self.bounds = bounds
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]

        self.sd_noise = sd_noise
        self.name = "Identity"

    def _objective_true(self, X):
        # identity function _f(x) = x
        # so that f(x) = Ax
        # note this is a multivalued function

        return X


class Indicator(Objective):
    def __init__(
        self, D: int = 2, sd_noise: float = 0.0,
        bounds: Optional[Tensor] = None
    ):
        self.D = D
        if bounds is None:
            bounds = Tensor([(-1., +1.) for i in range(D)])
        self.bounds = bounds
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]

        self.sd_noise = sd_noise
        self.name = "Indicator"

    def _objective_true(self, X):
        # f(x) = 1 if x in bounds
        # f(x) = 0 if x not in bounds
        isinbounds = (
            (X >= self.bounds_min) & (X <= self.bounds_max)
        ).all(axis=1)
        y = torch.zeros(X.shape[0])
        y[isinbounds] = 1.0
        return torch.atleast_2d(y).t()


class Branin(Objective):
    def __init__(self, sd_noise=0.0):
        self.D = 2
        self.bounds = torch.Tensor([(-5, 10), (0, 15)])
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = torch.Tensor([(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)])
        self.fmin = 0.397887

        self.sd_noise = sd_noise
        self.name = "Branin"

    def _objective_true(self, X):
        a = 1
        b = 5.1 / (4 * math.pi**2)
        c = 5 / math.pi
        r = 6
        s = 10
        t = 1. / (8 * math.pi)

        x1 = X[:, 0]
        x2 = X[:, 1]
        y = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*torch.cos(x1) + s
        return torch.atleast_2d(y).t()


class Sixhump(Objective):
    def __init__(self, sd_noise=0.0):
        self.D = 2
        self.bounds = torch.Tensor([(-5, 5), (-5, 5)])
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = torch.Tensor([
            (+0.0898, -0.7126),
            (-0.0898, +0.7126),
        ])
        self.fmin = -1.03162843  # actual minimum seems to be a little larger

        self.sd_noise = sd_noise
        self.name = "Sixhump"

    def _objective_true(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        y = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2
        return torch.atleast_2d(y).t()


class Hartmann6(Objective):
    def __init__(self, sd_noise=0.0):
        self.D = 6
        self.bounds = torch.Tensor([(0, 1) for i in range(6)])
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = torch.Tensor([
            (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        ])
        self.fmin = -3.32237  # actual minimum seems to be a little larger

        self.sd_noise = sd_noise
        self.name = "Hartmann6"

    def _objective_true(self, X):
        _alpha = torch.Tensor([1.0, 1.2, 3.0, 3.2])
        _A = torch.Tensor(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        _P = 10 ** (-4) * torch.Tensor(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        # (i,j,k) -> (i,j)
        t = torch.sum(_A[None, :, :] * torch.square(X[:, None, :] - _P[None, :, :]), dim=2)
        # (i,j) -> (i,)
        y = -torch.matmul(torch.exp(-t), _alpha)
        return torch.atleast_2d(y).t()


class Colville(Objective):
    def __init__(self, sd_noise=0.0):
        self.D = 4
        self.bounds = torch.Tensor([(-10, 10)] * 4)
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = torch.Tensor([
            (1.0, 1.0, 1.0, 1.0),
        ])
        self.fmin = 0.0

        self.sd_noise = sd_noise
        self.name = "Colville"

    def _objective_true(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        y = 100 * (x1**2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90*(x3**2 - x4)**2 + \
            10.1*((x2 - 1)**2 + (x4 - 1)**2) + 19.8*(x2 - 1)*(x4 - 1)
        return torch.atleast_2d(y).t()


class Goldstein(Objective):
    def __init__(self, sd_noise=0.0):
        self.D = 2
        self.bounds = torch.Tensor([(-2, 2)] * 2)
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = torch.Tensor([
            (0, -1),
        ])
        self.fmin = 3.0

        self.sd_noise = sd_noise
        self.name = "Goldstein"

    def _objective_true(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        y = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) \
            * (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
        return torch.atleast_2d(y).t()


class RoverTrajectoryObjective(Objective):
    def __init__(self, sd_noise=0.0):
        from .box2d import create_large_domain

        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        self._f = create_large_domain(
            force_start=False,
            force_goal=False,
            start_miss_cost=l2cost,
            goal_miss_cost=l2cost)

        self.D = self._f.traj.npoints * 2  # = 60
        self.bounds = torch.Tensor(np.repeat(self._f.s_range, self._f.traj.npoints, axis=1).T)
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]

        self.sd_noise = sd_noise
        self.name = "RoverTrajectory"

    def _objective_true(self, X):
        n = X.shape[0]
        Y = torch.Tensor([- self._f(X[i]) for i in range(n)])
        return torch.atleast_2d(Y).t()
