from typing import Callable

from torch import Tensor

from .turbo_1 import Turbo1 as Turbo1Class
from .turbo_m import TurboM as TurboMClass


def TurboM(
    objective: Callable,
    bounds: Tensor,
    T: int,
    n_init: int,
    B: int,
    n_trust_regions: int,
    ARD: bool = True,
    n_training_steps: int = 50,
    verbose: bool = True,
):
    def objective_numpy(X):
        Y = objective(Tensor(X)).detach().numpy()
        return Y
    if n_trust_regions == 1:
        turbo = Turbo1Class(
            f=objective_numpy,
            lb=bounds[:, 0].detach().numpy(),
            ub=bounds[:, 1].detach().numpy(),
            n_init=n_init,
            max_evals=T,
            batch_size=B,
            verbose=verbose,
            use_ard=ARD,
            max_cholesky_size=2000,
            n_training_steps=n_training_steps,
            device="cpu",
            dtype="float64",
        )
    else:
        turbo = TurboMClass(
            f=objective_numpy,
            lb=bounds[:, 0].detach().numpy(),
            ub=bounds[:, 1].detach().numpy(),
            n_init=n_init,
            max_evals=T,
            n_trust_regions=n_trust_regions,
            batch_size=B,
            verbose=verbose,
            use_ard=ARD,
            max_cholesky_size=2000,
            n_training_steps=n_training_steps,
            device="cpu",
            dtype="float64",
        )
    turbo.optimize()  # Run optimization
    return Tensor(turbo.X), Tensor(turbo.fX)
