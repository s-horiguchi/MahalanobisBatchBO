import itertools
import math

import numpy as np
import pytest
import torch
from problem import DesignMethod, get_design
from problem.highdim import HighDimObjectiveLast, HighDimObjectiveRandom
from problem.objective import Identity, Indicator

d_true = 2  # should be the same as defined in Identity and Indicator
D = 20


@pytest.fixture
def obj_last():
    return HighDimObjectiveLast(
        true_obj=Indicator,
        D=D, sd_noise=0,
        out_of_bound_org=math.nan,
        bound_extend=10,  # new bounds will be (-5, +5)
    )


def test_obj_last_bounds(obj_last):
    N_sample = 100
    bounds = obj_last.bounds*1.5
    X = get_design(bounds, N_sample, method=DesignMethod.RANDOM)

    bounds_min_last = obj_last.bounds[-d_true:, 0]
    bounds_max_last = obj_last.bounds[-d_true:, 1]
    isin_bounds = (
        (X[:, -d_true:] >= bounds_min_last) & (X[:, -d_true:] <= bounds_max_last)
    ).all(axis=1)
    notin_bounds = torch.logical_not(isin_bounds)

    Y = obj_last.objective_noisy(X)
    assert (Y[isin_bounds] == 1.0).all() \
        and torch.isnan(Y[notin_bounds]).all()


@pytest.fixture
def obj_random():
    return HighDimObjectiveRandom(
        true_obj=Identity,
        D=D, sd_noise=0,
        out_of_bound_org=math.nan,
        seed=0, sparsity=0.1,
    )


def test_obj_random_bounds(obj_random):
    N_sample = 100
    bounds = obj_random.bounds*1.1
    X = get_design(bounds, N_sample, method=DesignMethod.RANDOM)

    isin_bounds = torch.all(
        (X[:, :] >= -1) & (X[:, :] <= +1),
        dim=1
    )
    Y = obj_random.objective_noisy(X)
    assert torch.logical_not(torch.isnan(Y[isin_bounds])).all()


def test_obj_random_coverage_subset(obj_random):
    N_sample = 100
    bounds = obj_random.bounds
    X = get_design(bounds, N_sample, method=DesignMethod.RANDOM)

    Y = obj_random.objective_noisy(X)

    assert torch.all(
        torch.logical_and(Y >= obj_random.bounds_org_min, Y <= obj_random.bounds_org_max),
        dim=1
    ).all()


@pytest.mark.skip
def test_obj_random_coverage_superset(obj_random):
    print(obj_random.affine_matrix)
    print(obj_random.affine_vector)
    # N_sample = 100000
    # bounds = obj_random.bounds
    # X = get_design(bounds, N_sample, method="random")
    X = torch.Tensor(list(
        itertools.product([-1, +1], repeat=D)
    ))

    Y = obj_random.objective_noisy(X)

    H, xedges, yedges = np.histogram2d(
        Y[:, 0].numpy(), Y[:, 1].numpy(),
        range=[(-1, +1), (-1, +1)], bins=4,
    )
    print(H)
    print(xedges, yedges)
    raise
