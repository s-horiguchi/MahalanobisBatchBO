import torch
from gpytorch.priors import Prior
from torch.nn import Module as TModule


class SparseMatrixPrior(Prior):
    def __init__(self, alpha, input_dim: int, emb_dim: int, validate_args=False):
        TModule.__init__(self)
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        batch_shape = alpha.shape
        event_shape = torch.Size([input_dim*emb_dim])
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.register_buffer("alpha", alpha)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        R = value.view((self.emb_dim, self.input_dim))
        res = -0.5 * self.alpha * torch.sqrt(R.pow(2).sum(axis=0)).sum()
        return res
