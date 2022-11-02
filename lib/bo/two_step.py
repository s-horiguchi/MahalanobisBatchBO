from typing import Tuple

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor


def isin_normX(x):
    bound_min = -1.
    bound_max = +1.
    return (x >= bound_min).all().item() and (x <= bound_max).all().item()


def sample_normX(D):
    bound_min = -1.
    bound_max = +1.
    bound_range = bound_max - bound_min
    return torch.rand(D, requires_grad=True) * bound_range + bound_min


@torch.no_grad()
def sample_X_from_Z(
    R: Tensor,
    z: Tensor,
    max_epoch: int = 1000,
    tol: float = 1e-5,
    max_restart: int = 100,
    verbose: bool = False,
) -> Tensor:
    """
    Find a point in {x in [-1,+1]^D | Rx = z}.
    Specifically, run a gradient descent of ||Rx - z||^2 from a random initialization.
    Terminal condition is #epoch > `max_epoch` or ||Rx - z||^2 < `tol`.
    If x jump out of [-1,+1]^D, restart the optimization from a new point.
    When #restart > `max_restart`, return the clamped pseudo-inverse.
    """
    d, D = R.size()
    R_copy = R.detach().clone()
    z_copy = z.detach().clone().squeeze()

    def loss_func(x):
        return 0.5 * torch.sum(torch.square(R_copy @ x - z_copy))

    def gradient(x):
        return R_copy.t() @ (R_copy @ x - z_copy)

    def exact_line_search(x):
        num = torch.sum(torch.square(R_copy.t() @ (R_copy @ x - z_copy)))
        den = torch.sum(torch.square(R_copy @ R_copy.t() @ (R_copy @ x - z_copy)))
        return - num / den

    for r in range(max_restart):
        # initialize
        x = sample_normX(D)

        for epoch in range(max_epoch):
            if not isin_normX(x):
                break
            loss = loss_func(x)
            if loss.item() < tol:
                return x.detach().clone()
            grad = gradient(x)
            alpha = exact_line_search(x)
            x += alpha * grad

        # return only if reached max_epoch, restart otherwise
        if epoch == max_epoch - 1:
            return x.detach().clone()

    if verbose:
        print("failed to find a point in X from Z")
    # When corresponding points not found, return clamped pseudo-inverse
    R_pinv = torch.pinverse(R_copy)
    return torch.clamp(torch.matmul(z, R_pinv.t()), min=-1, max=+1)


def optimize_acq_on_Z(
    acq_function: AcquisitionFunction,
    R: Tensor,
    num_restarts: int = 5,
    raw_samples: int = 100,
) -> Tuple[Tensor, Tensor]:
    d, D = R.size()
    R_pinv = torch.pinverse(R).detach().clone()
    # prepare inequality constraints `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
    #   as a list of tuples (indices, coefficients, rhs)
    inequality_constraints = [
        (torch.arange(d), R_pinv[i], -torch.ones(1))
        for i in range(D)
    ] + [
        (torch.arange(d), -R_pinv[i], -torch.ones(1))
        for i in range(D)
    ]

    def acq_function_on_Z(Z):
        X = torch.matmul(Z, R_pinv.t())
        return acq_function(X)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_function_on_Z,
        bounds=Tensor([(-1e8, 1e8)] * d).t(),  # -inf, +inf
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"method": "SLSQP", "batch_limit": 1},
        inequality_constraints=inequality_constraints,
        sequential=False,
    )
    return candidate, acq_value
