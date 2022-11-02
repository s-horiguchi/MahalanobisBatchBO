import warnings

import torch
from gpytorch import settings as gpt_settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.warnings import NumericalWarning


def check_min_variance(mvn: MultivariateNormal) -> MultivariateNormal:
    # should be 1-d normal distribution
    if mvn.covariance_matrix.size(-2) != 1:
        raise TypeError

    # Check to make sure that variance isn't lower than minimum allowed value (default 1e-6).
    # This ensures that all variances are positive
    min_variance = gpt_settings.min_variance.value(mvn.covariance_matrix.dtype)
    if mvn.covariance_matrix.lt(min_variance).any():
        warnings.warn(
            f"Negative variance values detected. "
            "This is likely due to numerical instabilities. "
            f"Rounding negative variances up to {min_variance}.",
            NumericalWarning,
        )

        covar = mvn.covariance_matrix.clamp_min(min_variance)

        new_mvn = MultivariateNormal(
            mean=mvn.loc,
            covariance_matrix=covar
        )
        return new_mvn
    else:
        return mvn


def eval_loglikelihood(model, X, Y, jitter=1e-6):
    model.eval()
    with torch.no_grad(), gpt_settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False), \
            gpt_settings.cholesky_jitter(float=jitter, double=jitter):
        # supply X with shape (N_test, 1, D) -> get N_test 1d-norml dists
        output = model(X.unsqueeze(1))
        output = check_min_variance(output)
        # add observation noise and evaluate log likelihood
        obsv = model.likelihood(output)
        mllval = obsv.log_prob(Y)

    return mllval.mean().item()
