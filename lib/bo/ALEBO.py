import warnings
from typing import Callable

import torch
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.service.managed_loop import optimize
from torch import Tensor


def parametrization_to_Tensor(parametrization: dict, dim: int):
    return Tensor([
        [parametrization.get(str(i)) for i in range(dim)]
    ])


def _run_ax_BO(
    strategy,
    objective: Callable,
    bounds: Tensor,
    T: int,
    emb_dim: int,
    num_init: int = 10,
):
    input_dim = bounds.size(0)

    # create evaluation_function by wrapping objective
    def eval_func(parametrization):
        x = parametrization_to_Tensor(parametrization, input_dim)
        y = objective(x).numpy()
        # print(y)
        return {"objective": (y, 0.0)}

    parameters = [
        {
            "name": str(i),
            "type": "range",
            "bounds": [bounds[i, 0].item(), bounds[i, 1].item()],
            "value_type": "float"
        } for i in range(input_dim)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            objective_name="objective",
            evaluation_function=eval_func,
            minimize=True,
            total_trials=T+num_init,
            generation_strategy=strategy,
        )
    objectives = Tensor([
        trial.objective_mean for trial in experiment.trials.values()
    ])
    params = torch.stack([
        parametrization_to_Tensor(trial.arm.parameters, input_dim)[0]
        for trial in experiment.trials.values()
    ])
    return params, objectives


def ALEBO(
    objective: Callable,
    bounds: Tensor,
    T: int,
    emb_dim: int,
    num_init: int = 10,
):
    input_dim = bounds.size(0)

    strategy = ALEBOStrategy(
        D=input_dim, d=emb_dim, init_size=num_init
    )

    return _run_ax_BO(strategy, objective, bounds, T, emb_dim, num_init)


def REMBO(
    objective: Callable,
    bounds: Tensor,
    T: int,
    emb_dim: int,
    num_init: int = 10,
    num_proj: int = 4,
):
    input_dim = bounds.size(0)

    strategy = REMBOStrategy(
        D=input_dim, d=emb_dim,
        k=num_proj,
        init_per_proj=num_init // num_proj
    )

    return _run_ax_BO(strategy, objective, bounds, T, emb_dim, num_init)


def HeSBO(
    objective: Callable,
    bounds: Tensor,
    T: int,
    emb_dim: int,
    num_init: int = 10,
    num_proj: int = 1,
):
    input_dim = bounds.size(0)

    strategy = HeSBOStrategy(
        D=input_dim, d=emb_dim,
        k=num_proj,
        init_per_proj=num_init // num_proj
    )

    return _run_ax_BO(strategy, objective, bounds, T, emb_dim, num_init)
