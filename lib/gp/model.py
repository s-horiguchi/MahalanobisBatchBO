from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)
from torch import Tensor

from .kernel import MahalanobisKernel
from .prior import SparseMatrixPrior


class MahalanobisApproximateGP(ApproximateGP, GPyTorchModel):
    """The GP with Mahalanobis Kernel.
    Uses the Mahalanobis kernel  ScaleKernel to add a kernel variance and a fitted constant mean.

    In non-batch mode, there is a single kernel that produces MVN predictions
    as usual for a GP.
    With b batches, each batch has its own set of kernel hyperparameters and
    each batch represents a sample from the hyperparameter posterior
    distribution. When making a prediction (with `__call__`), these samples are
    integrated over using moment matching. So, the predictions are an MVN as
    usual with the same shape as in non-batch mode.
    Args:
        emb_dim: embedding dimension d.
        inducing_points
        R: dimension-reducing map
        alpha
    """
    _num_outputs = 1  # for GPyTorchModel

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            emb_dim: int,
            inducing_points: Tensor,
            R=None,
            alpha=1.0,
            learn_inducing_locations=True,
            learn_R=True,
            R_init_var=1.0,
            R_init_normalize=False,
    ) -> None:
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        input_dim = train_X.size(-1)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MahalanobisKernel(
                input_dim=input_dim, emb_dim=emb_dim,
                dtype=train_X.dtype, device=train_X.device,
                R=R, learn_R=learn_R,
                R_init_var=R_init_var,
                R_init_normalize=R_init_normalize,
            ),
        )
        self.register_prior(
            "Rvec_prior",
            # gpytorch.priors.MultivariateNormalPrior(torch.zeros(input_dim*emb_dim),
            #                                         torch.eye(input_dim*emb_dim)),
            SparseMatrixPrior(Tensor([alpha]), input_dim, emb_dim),
            lambda module: module.covar_module.base_kernel.Rvec)

        self.likelihood = GaussianLikelihood()
        self.train_inputs = train_X
        self.train_targets = train_Y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MahalanobisExactGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # for GPyTorchModel

    def __init__(
        self, train_X: Tensor, train_Y: Tensor,
        emb_dim: int, R=None, alpha=1.0, learn_R=True,
        R_init_var=1.0, R_init_normalize=False,
    ) -> None:
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)

        input_dim = train_X.size(-1)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MahalanobisKernel(
                input_dim=input_dim, emb_dim=emb_dim,
                dtype=train_X.dtype, device=train_X.device,
                R=R, learn_R=learn_R,
                R_init_var=R_init_var,
                R_init_normalize=R_init_normalize,
            ),
        )
        self.register_prior(
            "Rvec_prior",
            # gpytorch.priors.MultivariateNormalPrior(torch.zeros(input_dim*emb_dim),
            #                                         torch.eye(input_dim*emb_dim)),
            SparseMatrixPrior(Tensor([alpha]), input_dim, emb_dim),
            lambda module: module.covar_module.base_kernel.Rvec)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFApproximateGP(ApproximateGP, GPyTorchModel):

    _num_outputs = 1  # for GPyTorchModel

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            inducing_points: Tensor,
            learn_inducing_locations=True,
            ARD: bool = False,
    ) -> None:
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        if ARD:
            ard_num_dims: Optional[int] = train_X.size(-1)
        else:
            ard_num_dims = None
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=ard_num_dims))

        self.likelihood = GaussianLikelihood()
        self.train_inputs = train_X
        self.train_targets = train_Y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFExactGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # for GPyTorchModel

    def __init__(self, train_x, train_y, ARD=False):
        likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        if ARD:
            ard_num_dims = train_x.size(-1)
        else:
            ard_num_dims = None
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=ard_num_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
