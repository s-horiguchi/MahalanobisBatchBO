import math
import os
from typing import Optional, Type

import numpy as np
import torch
from torch import Tensor

from .objective import Objective
from .utils import BoundsType


class HighDimObjectiveLinear(Objective):
    def __init__(
        self, D_org: int, D: int,
        bounds: BoundsType,
        affine_matrix: Tensor,
        affine_vector: Optional[Tensor] = None,
        sd_noise: float = 0.0,
        out_of_bound_org: Optional[float] = None,
        bounds_org: Optional[BoundsType] = None,
        name: str = "Objective",
    ):
        """
        Parameters
        ----------
        D_org         : orignal low dimension
        D             : extended high dimension
        bounds        : bounds for high dimension [(xmin1, xmax1),...]
        affine_matrix : D_org x D matrix
        affine_vector : 1 x D_org vector(None is zero vector)
        out_of_bound_org : None for not restrict evaluation. Otherwise,
                           objective value out of bounds_org is replaced
                           with this value.
        bounds_org       : bounds for low dimension,
                           needed if out_of_bound_org is not None

        ----- Note about affine transformation ----
        Let :
        X      is in [bounds_min, bounds_max]^D
        X_norm is in [-1, 1]^D
        X_org  is in R^D_org (polyhedron)

        Then :
        X_org  = affine_matrix.dot(X) + affine_vector

        """
        assert D >= D_org
        assert len(bounds) == D

        self.D = D
        self.bounds = torch.Tensor(bounds)
        self.bounds_min = self.bounds[:, 0]
        self.bounds_max = self.bounds[:, 1]
        self.minimum = None
        self.fmin = None

        self.D_org = D_org

        # Affine Matrix
        assert affine_matrix.shape == (D_org, D)
        if affine_vector is None:
            # if None, use zeros
            _affine_vector = torch.zeros((1, D_org))
        else:
            _affine_vector = torch.atleast_2d(affine_vector)
            assert _affine_vector.shape == (1, D_org)
        self.affine_matrix = affine_matrix
        self.affine_vector = _affine_vector

        self.sd_noise = sd_noise
        self.name = name

        # Out of Bounds
        self.bounds_org = bounds_org
        if out_of_bound_org is None:
            self.out_of_bound_org = None
        else:
            # out_of_bound_org shoud be float (e.g. NaN, 0,...)
            self.out_of_bound_org = float(out_of_bound_org)
            assert bounds_org is not None
            assert len(bounds_org) == D_org
        if bounds_org is not None:
            self.bounds_org_min = torch.Tensor(bounds_org)[:, 0]
            self.bounds_org_max = torch.Tensor(bounds_org)[:, 1]

    def load_affine(self, dirname: str = "saved_objectives") -> None:
        fname = os.path.join(dirname, f"{self}_objective.pt")
        b, m, v = torch.load(fname)
        self.bounds = b
        self.affine_matrix = m
        self.affine_vector = v
        return

    def save_affine(self, dirname: str = "saved_objectives") -> None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, f"{self}_objective.pt")
        torch.save((
            self.bounds,
            self.affine_matrix,
            self.affine_vector,
            ), fname)
        return

    def isin_bounds_org(self, X_org: Tensor) -> Tensor:
        """
        X : (N, D_org)
        """
        if self.out_of_bound_org is None:
            return torch.ones_like(X_org, dtype=torch.bool)
        else:
            return torch.all(
                torch.logical_and(
                    X_org >= self.bounds_org_min,
                    X_org <= self.bounds_org_max
                ),
                dim=1
            )

    def _objective_true(self, X: Tensor) -> Tensor:
        """
        X : (N, D_org)
        """
        raise NotImplementedError

    def transform(self, X: Tensor) -> Tensor:
        """
        X : (N, D)
        """
        X = torch.atleast_2d(X)
        assert X.shape[1] == self.D
        X_org = torch.matmul(self.affine_matrix, X.t()).t()
        X_org += self.affine_vector
        return X_org

    def objective_true(self, X: Tensor) -> Tensor:
        """
        X : (N, D)
        """
        X_org = self.transform(X)
        if self.out_of_bound_org is None:
            return self._objective_true(X_org)
        else:
            isin_bounds = self.isin_bounds_org(X_org)
            Y_valid = self._objective_true(X_org[isin_bounds])
            if Y_valid.dim() == 1:
                Y = torch.empty(X.shape[0], dtype=Y_valid.dtype)
                Y[:] = self.out_of_bound_org
                Y[isin_bounds] = Y_valid
            elif Y_valid.dim() == 2:
                Y = torch.empty(
                    X.shape[0], Y_valid.shape[1],
                    dtype=Y_valid.dtype
                )
                Y[:] = self.out_of_bound_org
                Y[isin_bounds] = Y_valid
            return Y


class HighDimObjectiveLast(HighDimObjectiveLinear):
    def __init__(
        self, true_obj: Type[Objective], D: int,
        bound_extend: Optional[float] = None,
        sd_noise: float = 0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        true_obj      : True objective with low dimensional domain
        D             : extended high dimension
        bound_extend  : length of bound (xmax-xmin) for extended dim,
                        or None to use the mean length of bounds_org
        """
        self.true_obj = true_obj()  # instantiate
        D_org = self.true_obj.D
        bounds_org = self.true_obj.bounds
        if hasattr(self.true_obj, "minimum"):
            self.minimum_org = self.true_obj.minimum
        if hasattr(self.true_obj, "fmin"):
            self.fmin = self.true_obj.fmin

        affine_matrix = torch.zeros((D_org, D))
        affine_matrix[:, -D_org:] = torch.eye(D_org)
        if bound_extend is None:
            bound_extend = np.mean([bmax-bmin for bmin, bmax in bounds_org])
        else:
            bound_extend = float(bound_extend)
        bounds = torch.cat((
            Tensor([(-bound_extend/2, +bound_extend/2)]*(D-D_org)),
            Tensor(bounds_org)
        ), dim=0)
        super().__init__(
            D_org=D_org, D=D, bounds=bounds,
            affine_matrix=affine_matrix,
            affine_vector=None,
            sd_noise=sd_noise,
            name=self.true_obj.name + "Last",
            bounds_org=bounds_org,
            **kwargs)

    def _objective_true(self, X):
        return self.true_obj._objective_true(X)


class HighDimObjectiveRandom(HighDimObjectiveLinear):
    def __init__(
        self, true_obj: Type[Objective], D: int,
        sd_noise: float = 0.0,
        out_of_bound_org: Optional[float] = None,
        seed: Optional[int] = None,
        sparsity: float = 0,
        pickle: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        true_obj : Type[Objective]
            True objective with low dimensional domain
        D : int
            extended high dimension
        out_of_bound_org : float or None, default=None
            None for not restrict evaluation.
            Otherwise, objective value out of bounds_org is replaced with this value.
        seed: int or None, default=None
            Fix numpy & pytorch random seed if integer is provided
        sparsity: float, default=0
            Randomly pick `D` times `sparsity` number of dimensions
            and corresponding entries of `affine_matrix` become zeros
        pickle : bool or None, default=None
            If True, save and load generated affine matrix.
            If None, `pickle` becomes True when seed is None, and becomes False otherwise.
        """
        self.true_obj = true_obj()  # instantiate
        D_org = self.true_obj.D
        bounds_org = self.true_obj.bounds
        if hasattr(self.true_obj, "minimum"):
            self.minimum_org = self.true_obj.minimum
        if hasattr(self.true_obj, "fmin"):
            self.fmin = self.true_obj.fmin

        if seed is None:
            name = self.true_obj.name + f"Random_sp{sparsity}"
        else:
            name = self.true_obj.name + f"Random{seed}_sp{sparsity}"
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.seed = seed
        self.sparsity = sparsity

        # sparsity determines the number of active dimensions
        assert 0 <= sparsity < 1.0
        D_nonzero = D - int(math.floor(D * sparsity))
        assert 0 < D_nonzero <= D
        # randomly choose nonzero dimension indices
        dims_nonzero, _ = torch.sort(torch.randperm(D)[:D_nonzero])

        # For A[:, dims_nonzero], each entry is N(0, 1) and normalize by L1 norm
        # Other entries are exactly zeros
        A = torch.zeros((D_org, D))
        A[:, dims_nonzero] = torch.normal(mean=0, std=1, size=(D_org, D_nonzero))
        A /= torch.atleast_2d(
            torch.clamp(torch.norm(A, p=1, dim=1),
                        min=1e-20)).t()

        if seed is not None:
            np.random.seed(None)

        bounds_org_min = torch.Tensor(bounds_org)[:, 0]
        bounds_org_max = torch.Tensor(bounds_org)[:, 1]
        org_range = (bounds_org_max - bounds_org_min)/2
        org_center = (bounds_org_min + bounds_org_max)/2
        affine_matrix = A * torch.atleast_2d(org_range).t()
        bounds = torch.Tensor([(-1.0, +1.0) for i in range(D)])

        super().__init__(
            D_org=D_org, D=D, bounds=bounds,
            affine_matrix=affine_matrix,
            affine_vector=org_center,
            sd_noise=sd_noise,
            out_of_bound_org=out_of_bound_org,
            bounds_org=bounds_org,
            name=name
        )

        self.D_nonzero = D_nonzero
        self.dims_nonzero = dims_nonzero

        if pickle:
            try:
                # load parameters if exist
                self.load_affine()
            except FileNotFoundError:
                # save the parameters for later reuse
                self.save_affine()

    def _objective_true(self, X):
        return self.true_obj._objective_true(X)
