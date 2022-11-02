import math
import time
import traceback
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import _filter_kwargs, _get_extra_mll_args
from gpytorch import settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood, PredictiveLogLikelihood
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from .model import (MahalanobisApproximateGP, MahalanobisExactGP,
                    RBFApproximateGP, RBFExactGP)
from .utils import eval_loglikelihood

ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_GP(
        X, Y,
        restarts: int = 5,  # number of optimization restarts
        emb_dim: int = 0,  # 0 for RBF-ARD, 1 for RBF+ARD, >=1 for Mahalanobis
        RBF: bool = False,
        alpha: float = 0.0,  # regularization parameter
        learn_R: bool = True,  # learn embedding matrix of Mahalanobis Kernel (for emb_dim >= 1)
        R_init_var: float = 1.0,
        R_init_normalize: bool = False,
        approximate: bool = False,  # True for ExactGP, False for ApproximateGP
        N_inducing: int = 100,  # number of inducing points (for approximateGP)
        # size of mini-batch (for approximateGP), None for the entire training data size
        minibatchsize: Optional[int] = None,
        optimizer_cls: Type[Optimizer] = Adam,
        options: Optional[Dict[str, Any]] = None,  # options for optimizer
        bounds: Optional[Any] = None,
        approx_covar_dec: bool = False,
        approx_mll: bool = False,  # BBMM
        use_pcg: bool = False,  # preconditioned CG
        fast_pred_var: bool = False,  # Lanczos Variance Estimates
        dec_size: int = 100,  # max_root_decomposition_size
        jitter: float = 1e-6,
        track_iterations: bool = False,
        init_state_dict: Optional[Any] = None,  # set this state dict for model before optimization
        validation_ratio: Optional[int] = None,
        return_losses: bool = False,
):
    N = X.size(-2)
    if validation_ratio is None:
        N_train = N_val = N
        X_train = X_val = X
        Y_train = Y_val = Y
    else:
        N_val = int(N * validation_ratio)
        N_train = N - N_val
        perm = torch.randperm(N)
        is_train = perm[:N_train]
        is_val = perm[N_train:]
        X_train = X[..., is_train, :]
        X_val = X[..., is_val, :]
        Y_train = Y[..., is_train]
        Y_val = Y[..., is_val]

    loss_best = float('inf')
    m_state_dict_best = {}
    list_loss = []
    for _ in range(restarts):
        try:
            m, info = _fit_GP(
                X_train, Y_train, emb_dim, RBF,
                alpha, learn_R,
                R_init_var, R_init_normalize,
                approximate, N_inducing, minibatchsize,
                optimizer_cls, options, bounds,
                approx_covar_dec, approx_mll, use_pcg,
                fast_pred_var, dec_size, jitter,
                track_iterations,
                init_state_dict)
        except Exception:
            print(traceback.format_exc())
            print("skipping _fit_GP once")
            continue
        if validation_ratio is None:
            val_loss = info["fopt"]
        else:
            val_loss = eval_loglikelihood(m, X_val, Y_val.unsqueeze(1))
        list_loss.append(val_loss)
        if val_loss < loss_best:
            loss_best = val_loss
            m_state_dict_best = m.state_dict()
    # get best model
    m = create_GPModel(X, Y,
                       emb_dim, RBF, alpha, learn_R,
                       R_init_var, R_init_normalize,
                       approximate, N_inducing)
    m.load_state_dict(m_state_dict_best)
    if return_losses:
        return m, list_loss
    else:
        return m


def create_GPModel(
        train_X, train_Y,
        emb_dim, RBF, alpha, learn_R,
        R_init_var, R_init_normalize,
        approximate, N_inducing, noise_inducing=1e-3):
    N_train = train_X.size(-2)

    if RBF and emb_dim < 2:
        # RBF with/without ARD
        ARD = True if emb_dim == 1 else False

        if not approximate:
            model = RBFExactGP(train_X, train_Y, ARD=ARD)
        else:
            # choose points from train_X and add noise
            inducing_points = torch.normal(
                mean=train_X[torch.randint(N_train, (N_inducing,))],
                std=noise_inducing
            )
            model = RBFApproximateGP(
                train_X, train_Y, ARD=ARD,
                inducing_points=inducing_points,
                learn_inducing_locations=True
            )

    else:
        if not approximate:
            # exact GP
            model = MahalanobisExactGP(
                train_X, train_Y,
                emb_dim=emb_dim,
                alpha=alpha,
                learn_R=learn_R,
                R_init_var=R_init_var,
                R_init_normalize=R_init_normalize,
            )
        else:
            # approximate GP
            # choose points from train_X and add noise
            inducing_points = torch.normal(
                mean=train_X[torch.randint(N_train, (N_inducing,))],
                std=noise_inducing
            )
            model = MahalanobisApproximateGP(
                train_X, train_Y,
                emb_dim=emb_dim,
                inducing_points=inducing_points,
                alpha=alpha,
                learn_inducing_locations=True,
                learn_R=learn_R,
                R_init_var=R_init_var,
                R_init_normalize=R_init_normalize,
            )

    return model


def _fit_GP(
        train_X, train_Y,
        emb_dim: int = 0,  # 0 for RBF-ARD, 1 for RBF+ARD, >=1 for Mahalanobis
        RBF: bool = False,
        alpha: float = 0.0,  # regularization parameter
        learn_R: bool = True,  # learn embedding matrix of Mahalanobis Kernel (for d >= 1)
        R_init_var: float = 1.0,
        R_init_normalize: bool = False,
        approximate: bool = False,  # True for ExactGP, False for ApproximateGP
        N_inducing: int = 100,  # number of inducing points (for approximateGP)
        # size of mini-batch (for approximateGP), None for the entire training data size
        minibatchsize: Optional[int] = None,
        optimizer_cls: Type[Optimizer] = Adam,
        options: Optional[Dict[str, Any]] = None,  # options for optimizer
        bounds: Optional[Any] = None,  # parameter bounds
        approx_covar_dec: bool = True,  # Lanczos low-rank approx
        approx_mll: bool = True,  # BBMM
        use_pcg: bool = True,  # preconditioned CG
        fast_pred_var: bool = False,  # Lanczos Variance Estimates
        dec_size: int = 100,  # max_root_decomposition_size
        jitter: float = 1e-6,
        track_iterations: bool = False,
        init_state_dict: Optional[Any] = None,  # set this state dict for model before optimization
):
    N_train = train_X.size(-2)

    model = create_GPModel(train_X, train_Y,
                           emb_dim, RBF, alpha, learn_R,
                           R_init_var, R_init_normalize,
                           approximate, N_inducing)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    if approximate:
        mll = PredictiveLogLikelihood(model.likelihood, model, num_data=N_train)
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Train
    model.train()

    if not approximate or minibatchsize is None:
        # use entire training data
        dataloader: Union[List, DataLoader] = [(train_X, train_Y)]
    else:
        # minibatch
        dataset = TensorDataset(train_X, train_Y)
        dataloader = DataLoader(dataset,
                                batch_size=minibatchsize,
                                shuffle=True)

    # below are edited from fit_gpytorch_torch()
    optim_options: Dict[str, Any] = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude: Optional[List] = optim_options.pop("exclude", None)
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())

    optimizer: Optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound
    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    # param_trajectory: Dict[str, List[Tensor]] = {
    #     name: [] for name, param in mll.named_parameters()
    # }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    while not stop:
        loss_batch = []
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            with gpt_settings.fast_computations(
                    covar_root_decomposition=approx_covar_dec,
                    log_prob=approx_mll,
                    solves=use_pcg), \
                    gpt_settings.fast_pred_var(fast_pred_var), \
                    gpt_settings.max_root_decomposition_size(dec_size), \
                    gpt_settings.cholesky_jitter(float=jitter, double=jitter):
                output = mll.model(batch_X)
                # we sum here to support batch mode
                args = [output, batch_Y] + _get_extra_mll_args(mll)
                loss = -mll(*args).sum()
                loss.backward()

            optimizer.step()
            # project onto bounds:
            if bounds_:
                for pname, param in mll.named_parameters():
                    if pname in bounds_:
                        param.data = param.data.clamp(*bounds_[pname])
            loss_batch.append(loss.item())
        loss_epoch = sum(loss_batch) / len(loss_batch)
        if math.isnan(loss_epoch):
            break
            # raise ValueError("loss become NaN. decrease max iter?")
        loss_trajectory.append(loss_epoch)
        # for name, param in mll.named_parameters():
        #     param_trajectory[name].append(param.detach().clone())
        if optim_options["disp"] and (
                (i + 1) % 10 == 0 or i == (optim_options["maxiter"] - 1)
        ):
            print(f"Iter {i + 1}/{optim_options['maxiter']}: {loss_epoch}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss_epoch, time.time() - t1))
        i += 1
        stop = stopping_criterion.evaluate(fvals=torch.Tensor([loss_epoch]))

    info_dict = {
        "fopt": min(loss for loss in loss_trajectory if not math.isnan(loss)),
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }

    return model, info_dict
