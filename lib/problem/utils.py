from enum import Enum

import torch
from torch import Tensor

BoundsType = Tensor  # Union[List, Tensor]


class DesignMethod(Enum):
    LATIN = "latin"
    RANDOM = "random"
    SOBOL = "sobol"


def get_design(
    bounds: BoundsType,
    num_points: int,
    method: DesignMethod = DesignMethod.RANDOM,
) -> Tensor:
    dim = len(bounds)
    bounds = torch.Tensor(bounds)
    r = bounds[:, 1] - bounds[:, 0]

    if not (r >= 0).all():
        raise ValueError("bounds_min exceeds bounds_max")

    if method == DesignMethod.LATIN:
        from pyDOE import lhs
        samples = lhs(dim, num_points, criterion='center')
    elif method == DesignMethod.RANDOM:
        samples = torch.rand((num_points, dim))
    elif method == DesignMethod.SOBOL:
        from botorch.utils.sampling import draw_sobol_samples
        normalized_bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        samples = draw_sobol_samples(normalized_bounds, num_points, 1)[:, 0, :]
    else:
        raise TypeError("invalid method")

    return bounds[:, 0] + r * samples
