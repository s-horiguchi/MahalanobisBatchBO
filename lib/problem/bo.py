from typing import Optional

from torch import Tensor

from .objective import Objective
from .utils import DesignMethod, get_design


class Problem(object):
    def __init__(
        self,
        objective: Objective,
        T: int = 100,
        B: int = 1,
        num_init: int = 2,
    ) -> None:
        self.T = T
        self.B = B
        self.num_init = num_init
        self.objective = objective
        self.design_samples: Optional[Tensor] = None
        self.design_values: Optional[Tensor] = None

    def init_design(
        self, method: DesignMethod = DesignMethod.LATIN
    ) -> None:
        assert isinstance(self.objective, Objective)
        self.design_samples = get_design(
            self.objective.bounds,
            self.num_init, method
        )
        self.design_values = self.objective.objective_noisy(
            self.design_samples
        )

    def __str__(self):
        return f"{self.objective}_B{self.B}_T{self.T}_I{self.num_init}"
