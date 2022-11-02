import math
from typing import Optional, Union

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from problem import DesignMethod, get_design
from torch import Tensor


class SimpleUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Modified Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + beta * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    Original UCB use `sqrt(beta)` instead of `beta`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta: Tensor = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        stddev = posterior.variance.view(batch_shape).sqrt()
        delta = self.beta.expand_as(mean) * stddev
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta


class EST(AnalyticAcquisitionFunction):
    r"""Single-outcome EST.

    `EST(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    `berta` is estimated

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EST = EST(model, best_f, bounds)
        >>> est = EST(test_X)
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            bounds: Tensor,
            num_basepoints: int = None,
            beta: Optional[Union[float, Tensor]] = None,
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
            basepoint_sampler: DesignMethod = DesignMethod.RANDOM,
            verbose: bool = False,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        assert not maximize  # TODO: support maximize
        self.basepoint_sampler = basepoint_sampler
        self.best_f = best_f
        self.bounds = bounds
        if num_basepoints is None:
            input_dim = model.train_inputs[0].size(-1)
            self.num_basepoints = min(2**input_dim, 10000)
        else:
            self.num_basepoints = num_basepoints

        if beta is None:
            beta = self.estimate_beta(best_f, verbose=verbose)

        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @torch.no_grad()
    def estimate_beta(
            self,
            best_f: Union[Tensor, float],
            binwidth: Optional[Union[Tensor, float]] = None,
            max_count: int = 10000,
            verbose: bool = False
    ) -> float:
        """
        Calculate beta by estimating the posterior expectation of minimum.

        Since the approximation formula is for discrete space,
        we use finite number of base points sampled from continuous domain.
        Note that the result depends on the number of base points
        """

        if verbose:
            print("generating base points")
        # Get posterior distributions of base points
        basepoints = get_design(self.bounds, self.num_basepoints, self.basepoint_sampler)
        if verbose:
            print("getting posterior at base points")
        post_basepoints = self._get_posterior(basepoints)
        mean = post_basepoints.mean
        std = torch.sqrt(post_basepoints.variance)

        # Estimate minimum value by numerical integration
        if binwidth is None:
            binwidth = torch.clamp(0.01 * torch.mean(std), min=0.01)
        if isinstance(binwidth, float):
            binwidth = Tensor([binwidth])
        if isinstance(best_f, float):
            best_f = Tensor([best_f])

        if verbose:
            print("starting numerical integration")
        w = 0 + best_f  # integrating variable
        m = 0 + best_f  # integrated value
        lpp = Tensor([-1e8])  # log prod Phi
        lpp_trajectory = []
        count = 0
        while lpp < -1e-5:
            cdf = 0.5 * (1 + torch.erf((w - mean) * std.reciprocal() / math.sqrt(2)))
            lpp = torch.log(1-cdf).sum()
            m -= (1 - torch.exp(lpp)) * binwidth
            w -= binwidth
            count += 1
            lpp_trajectory.append(lpp)
            if count > max_count:
                if verbose:
                    print(f"@EST (count > {max_count})")
                    print("w:", w)
                    print("m:", m)
                    print("binwidth:", binwidth)
                    print("logprodphi:", torch.Tensor(lpp_trajectory))
                break

        if verbose:
            print("min_est:", m.item(), "min_m:", mean.min().item(), "min_s:", std.min().item())

        # calculate beta from min_est
        if verbose and m > mean.min().item() - 1e-6:
            print("min_est > min(mean) - 1e-6")
        min_est = torch.clamp(m, max=mean.min().item() - 1e-6)
        beta = torch.min((mean - min_est) / std)
        if verbose:
            i_min = torch.argmin((mean - min_est) / std)
            print("beta=", beta.item(), "@mean:", mean[i_min].item(), "std:", std[i_min].item())
        # assert beta > 0
        return beta.item()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta: Tensor = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        stddev = posterior.variance.view(batch_shape).sqrt()
        delta = self.beta.expand_as(mean) * stddev
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta
