from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from torch import Tensor


class MahalanobisKernel(Kernel):
    """Mahalanobis Kernel

    Args:
        input_dim
        emb_dim
        R : a matrix that maps from input_dim space to emb_dim space
        batch_shape: Batch shape as usual for gpytorch kernels.
        dtype
        device
    """

    def __init__(
        self, input_dim: int, emb_dim: int, dtype: torch.dtype, device: torch.device,
        R: Optional[Tensor] = None, learn_R: bool = True,
        R_init_var=1, R_init_normalize=False,
        batch_shape=torch.Size([])
    ) -> None:
        super().__init__(
            has_lengthscale=False, ard_num_dims=None, batch_shape=batch_shape
        )
        self.d = emb_dim
        self.D = input_dim
        if R is None:
            # Initialize R with hypersphere sampling
            R = torch.randn(self.d, self.D, dtype=dtype, device=device) * R_init_var
            if R_init_normalize:
                R = R / torch.sqrt((R ** 2).sum(dim=0))

        Rvec = R.flatten().repeat(*batch_shape, 1)
        if learn_R:
            self.register_parameter("Rvec", torch.nn.Parameter(Rvec))
        else:
            self.register_buffer("Rvec", Rvec)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        """Compute kernel distance."""
        assert last_dim_is_batch is False
        # unpack R
        shapeR = self.Rvec.shape[:-1] + torch.Size([self.d, self.D])
        R = self.Rvec.view(shapeR)
        # Compute kernel distance
        z1 = torch.matmul(x1, torch.t(R))
        z2 = torch.matmul(x2, torch.t(R))
        return self.covar_dist(
            z1,
            z2,
            square_dist=True,
            diag=diag,
            dist_postprocess_func=postprocess_rbf,
            postprocess=True,
            **params,
        )
