from enum import Enum
from typing import Callable, Optional, Tuple

import torch
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from gp import MahalanobisKernel, fit_GP
from gpytorch import settings as gpt_settings
from problem import DesignMethod, get_design
from torch import Tensor

from .acquisition import EST, SimpleUpperConfidenceBound
from .two_step import optimize_acq_on_Z, sample_X_from_Z
from .utils import normalize, standardize, unnormalize


@torch.no_grad()
def relevance_region_sampler(
        max_samples: int,
        bounds: Tensor,
        count: int,
        model: Model,
        beta: float,
        min_ucb: float,
) -> Tensor:
    proposal = get_design(bounds, max_samples, DesignMethod.RANDOM)

    factor = 2 + (count / 10)
    mvn = model.likelihood(model(proposal))
    m = mvn.mean
    s = torch.sqrt(mvn.variance)
    lcb2 = (m - factor*beta*s).flatten()
    is_valid = lcb2 < min_ucb
    # print("valid:", torch.count_nonzero(is_valid), "/", max_samples)
    return proposal[is_valid]


@torch.no_grad()
def DPP_sampling(
        model: Model,
        k: int,
        sampler: Callable[[int, int], Tensor],
        noise_kernel: bool = False,
        verbose: bool = False
) -> Tensor:
    """
    continuous k-DPP sampling in relevance region
    ref: https://github.com/alireza70/Continuous-k-DPP-Sampling

    Args:
        model
        k
        sampler: sampler(max_samples, count)
        verbose
    """

    def conditional_sampler(
            cur_points: Tensor,
            maxIter=5000
    ):
        n = cur_points.size(0)
        noise = model.likelihood.noise.item()  # 1e-6
        count = 0
        while True:
            proposal = sampler(maxIter, count)
            count += 1
            test = torch.rand(proposal.size(0))
            for sample, u in zip(proposal, test):
                # compute the change in determinant after adding a new point
                if n > 0:
                    mvn_cur = model.likelihood(model(cur_points))
                    cov_post_cur = mvn_cur.covariance_matrix
                    if noise_kernel:
                        K_cur = torch.eye(n) + noise**(-2)*cov_post_cur
                    else:
                        K_cur = cov_post_cur
                    cur_det = torch.linalg.det(K_cur)

                    changed_points = torch.cat([cur_points, sample.unsqueeze(0)], dim=0)
                    mvn_cha = model.likelihood(model(changed_points))
                    cov_post_cha = mvn_cha.covariance_matrix
                    if noise_kernel:
                        K_cha = torch.eye(n+1) + noise**(-2)*cov_post_cha
                    else:
                        K_cha = cov_post_cha
                    changed_det = torch.linalg.det(K_cha)
                else:
                    cur_det = 1
                    changed_points = sample.unsqueeze(0)
                    mvn_cha = model.likelihood(model(changed_points))
                    if noise_kernel:
                        changed_det = 1 + noise**(-2) * mvn_cha.variance
                    else:
                        changed_det = mvn_cha.variance

                # check the criteria for acception
                if u <= changed_det / cur_det:
                    return sample

    # find warm start
    input_dim = model.train_inputs[0].size(-1)
    cur_point = torch.zeros((k, input_dim))
    for i in range(k):
        cur_point[i, :] = conditional_sampler(cur_point[0:i, :])
    # run Gibbs sampling
    for i in range(5 * k ** 2):
        index = int(torch.randint(low=0, high=k, size=[1]).item())
        points_after_removal = torch.cat([cur_point[:index], cur_point[index+1:]])
        new_point = conditional_sampler(points_after_removal)
        cur_point = torch.cat([points_after_removal, new_point.unsqueeze(0)])

    if verbose:
        print("gibbs sampling finished!\n")
    return cur_point


class BatchStrategy(Enum):
    RANDOM = "random"
    LATIN = "latin"
    SOBOL = "sobol"
    DPP = "dpp"
    DPP_RBF = "dpp_rbf"


# @profile
def DPP_Mahalanobis(
        objective: Callable,
        bounds: Tensor,
        T: int,  # total budget
        batch_size: int,  # batch size
        emb_dim: int,
        RBF: bool = False,
        alpha: float = 0.0,
        learn_R: bool = True,
        R_init_var: float = 1e-3,
        R_init_normalize: bool = False,
        approximate: bool = False,  # True for ExactGP, False for ApproximateGP
        N_inducing: int = 100,  # number of inducing points (for approximateGP)
        # size of mini-batch (for approximateGP), None for the entire training data size
        minibatch_size: Optional[int] = None,
        init_X: Optional[Tensor] = None,
        init_Y: Optional[Tensor] = None,
        num_init: Optional[int] = None,
        num_fit_restarts: int = 10,
        num_acq_restarts: int = 5,
        raw_samples: int = 100,  # number of anchor points for acq optimization
        Y_standardize: bool = True,
        options_EST: Optional[dict] = None,
        verbose: bool = False,
        fast_pred_var: bool = False,
        jitter: float = 1e-6,
        pseudo_inv: bool = False,
        batch_strategy: BatchStrategy = BatchStrategy.DPP,
        # None for DPP sampling (default), item of DesignMethod for random sampling
        noise_dpp_kernel: bool = False,  # add noise term in DPP kernel
        two_step: bool = False,
        num_sample_restarts: int = 1000,
) -> Tuple[Tensor, Tensor]:
    input_dim = bounds.size(0)

    # initial design
    if init_X is not None and init_Y is not None and init_X.size(0) == init_Y.size(0):
        X = init_X.detach().clone()
        Y = init_Y.detach().clone()
    elif num_init is not None and num_init > 0:
        X = get_design(bounds, num_init, DesignMethod.SOBOL)
        Y = objective(X)
    else:
        raise ValueError("init_X and init_Y, or num_init should be provided.")

    Y = Y.squeeze()

    bounds_norm = torch.stack([-torch.ones(input_dim), torch.ones(input_dim)]).t()

    for i in range(T // batch_size):
        if verbose:
            print(f"[{i}]")
        # Update GP model
        X_norm = normalize(X, bounds)
        if Y_standardize:
            Y_st, params_Yst = standardize(Y)
        else:
            Y_st = Y

        if verbose:
            print("fitting GP")
        model = fit_GP(
            X_norm, Y_st,
            restarts=num_fit_restarts,
            emb_dim=emb_dim,
            RBF=RBF,
            alpha=alpha,
            learn_R=learn_R,
            R_init_var=R_init_var,
            R_init_normalize=R_init_normalize,
            approximate=approximate,
            N_inducing=N_inducing,
            minibatchsize=minibatch_size,
            options={"disp": False, "maxiter": 1000, "lr": 0.1},
            approx_covar_dec=False,
            approx_mll=False,
            use_pcg=False,
            fast_pred_var=fast_pred_var,
            jitter=jitter
        )

        if verbose:
            print("selecting first point")
        if options_EST is None:
            options_EST = {}
        X_batch_norm = torch.empty([batch_size, input_dim])

        # turn off all approximations
        with gpt_settings.fast_computations(
                covar_root_decomposition=False,
                log_prob=False,
                solves=False), \
                gpt_settings.cholesky_jitter(float=jitter, double=jitter):
            est = EST(model, best_f=Y_st.min().item(), bounds=bounds_norm,
                      maximize=False, verbose=verbose, **options_EST)

            if (two_step or pseudo_inv) and \
                    isinstance(model.covar_module.base_kernel, MahalanobisKernel):
                Rvec = model.covar_module.base_kernel.Rvec
                shapeR = Rvec.shape[:-1] + torch.Size([emb_dim, input_dim])
                R = Rvec.view(shapeR)
                Z_opt, _ = optimize_acq_on_Z(
                    est, R,
                    num_restarts=num_acq_restarts, raw_samples=raw_samples)
                if pseudo_inv:
                    X_batch_norm[0] = sample_X_from_Z(R, Z_opt, max_restart=0)
                else:
                    X_batch_norm[0] = sample_X_from_Z(
                        R, Z_opt,
                        max_restart=num_sample_restarts,
                        verbose=verbose
                    )
            else:
                X_batch_norm[0], _ = optimize_acqf(
                    est, bounds=bounds_norm.t(),
                    q=1, num_restarts=num_acq_restarts, raw_samples=raw_samples)

            if batch_size > 1:
                if batch_strategy == BatchStrategy.RANDOM:
                    # randomly choose for the rest of the points
                    X_batch_norm[1:] = get_design(bounds_norm, batch_size-1, DesignMethod.RANDOM)
                elif batch_strategy == BatchStrategy.LATIN:
                    # randomly choose for the rest of the points
                    X_batch_norm[1:] = get_design(bounds_norm, batch_size-1, DesignMethod.LATIN)
                elif batch_strategy == BatchStrategy.SOBOL:
                    # randomly choose for the rest of the points
                    X_batch_norm[1:] = get_design(bounds_norm, batch_size-1, DesignMethod.SOBOL)
                else:
                    # DPP sampling in relevance region for the rest of the points
                    if batch_strategy == BatchStrategy.DPP_RBF:
                        # use RBF kernel
                        RBF_post = True
                        # learn kernel parameters
                        maxiter_post = 1000
                        restarts_post = num_fit_restarts
                    elif batch_strategy == BatchStrategy.DPP:
                        # use the same kernel as the first point
                        RBF_post = RBF
                        # suppress learning kernel parameters
                        maxiter_post = 10
                        restarts_post = 1

                    if verbose:
                        print("fitting GP again")
                    # get posterior after the first point
                    X_post_norm = torch.cat([X_norm, X_batch_norm[0:1]])
                    Y_post_st = torch.cat([Y_st, torch.zeros(1)])
                    model_post = fit_GP(
                        X_post_norm, Y_post_st, restarts=restarts_post,
                        emb_dim=emb_dim,
                        RBF=RBF_post,
                        alpha=alpha,
                        learn_R=learn_R,
                        R_init_var=R_init_var,
                        R_init_normalize=R_init_normalize,
                        approximate=approximate,
                        N_inducing=N_inducing,
                        minibatchsize=minibatch_size,
                        options={"disp": False, "maxiter": maxiter_post, "lr": 0.1},
                        init_state_dict=model.state_dict(),
                        approx_covar_dec=False,
                        approx_mll=False,
                        use_pcg=False,
                        fast_pred_var=fast_pred_var,
                        jitter=jitter
                    )
                    # set relevance region
                    # maximize -m + (-beta)s = minimize m+betas
                    ucb = SimpleUpperConfidenceBound(
                        model_post, beta=-est.beta.item(), maximize=False
                    )
                    _, min_ucb = optimize_acqf(
                        ucb, bounds=bounds_norm.t(),
                        q=1, num_restarts=num_acq_restarts, raw_samples=raw_samples)

                    def sampler(m, c):
                        return relevance_region_sampler(
                            max_samples=m,
                            bounds=bounds_norm,
                            count=c,
                            model=model_post,
                            beta=est.beta.item(),
                            min_ucb=min_ucb.item(),
                        )
                    if verbose:
                        print("sampling DPP")
                    # DPP sampling
                    X_batch_norm[1:] = DPP_sampling(model_post,
                                                    k=batch_size-1,
                                                    sampler=sampler,
                                                    noise_kernel=noise_dpp_kernel,
                                                    verbose=verbose)

            if pseudo_inv:
                if not isinstance(model.covar_module.base_kernel,
                                  MahalanobisKernel):
                    raise TypeError("pseudo_inv mode is only available for Mahalanobis Kernel")
                with torch.no_grad():
                    Rvec = model.covar_module.base_kernel.Rvec
                    shapeR = Rvec.shape[:-1] + torch.Size([emb_dim, input_dim])
                    R = Rvec.view(shapeR)
                    R_pinv = torch.pinverse(R)
                    X_batch_norm = torch.matmul(X_batch_norm, R.t()) @ R_pinv.t()
                    # project onto the box [-1, +1]
                    X_batch_norm = torch.clamp(X_batch_norm, min=-1, max=+1)

        # observe
        X_batch = unnormalize(X_batch_norm, bounds)
        Y_batch = objective(X_batch).squeeze(dim=1)
        X = torch.cat([X, X_batch])
        Y = torch.cat([Y, Y_batch])

    return X, Y
