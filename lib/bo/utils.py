from typing import Dict, Mapping, Tuple

import torch
from problem import BoundsType
from torch import Tensor


def normalize(
    X: Tensor,
    bounds: BoundsType,
    thruorigin: bool = True
) -> Tensor:
    """
    if thruorigin is True:
        [bounds[0], bounds[1]] -> [-1, +1]^D
    else:
        [bounds[0], bounds[1]] -> [0, +1]^D
    """
    bounds = torch.Tensor(bounds)
    if thruorigin:
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        width = (bounds[:, 1] - bounds[:, 0]) / 2
        return (X - center) / width
    else:
        mins = bounds[:, 0]
        width = bounds[:, 1] - bounds[:, 0]
        return (X - mins) / width


def unnormalize(
    X: Tensor,
    bounds: BoundsType,
    thruorigin: bool = True
) -> Tensor:
    bounds = torch.Tensor(bounds)
    if thruorigin:
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        width = (bounds[:, 1] - bounds[:, 0]) / 2
        return X * width + center
    else:
        mins = bounds[:, 0]
        width = bounds[:, 1] - bounds[:, 0]
        return X * width + mins


def standardize(Y: Tensor) -> Tuple[Tensor, Dict]:
    params = {
        "mean": Y.mean(),
        "std": Y.std()
    }
    Y_st = (Y - params["mean"]) / params["std"]
    return Y_st, params


def unstandardize(Y_st: Tensor, params: Mapping) -> Tensor:
    return Y_st * params["std"] + params["mean"]
