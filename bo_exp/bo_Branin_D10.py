import sys
import traceback
from typing import Any, Dict

from bo import BatchStrategy
from joblib import Parallel, cpu_count, delayed
from problem import DesignMethod, HighDimObjectiveRandom, Problem
from problem.objective import Branin
from utils import notify

from base_bo_exp import process_bo

# -- problem setting --
D = 10           # extended dimension
sd_noise = 0.0   # sd of gaussian observation noise
N_iter = 1       # repeated experiment
N_seed = 20      # different random seed
N_init = 10
T = 500   # total budget
B = 5

# -- experiment setting --
n_jobs = cpu_count()//4


# define problems
problems = []
for s in range(N_seed):
    obj = HighDimObjectiveRandom(
        true_obj=Branin,
        D=D, sd_noise=sd_noise, seed=s,
    )

    prob = Problem(objective=obj, T=T, B=B, num_init=N_init)
    prob.init_design(DesignMethod.SOBOL)
    problems.append(prob)


# define methods to compare
default_args = {
    "verbose": False,
    "num_fit_restarts": 10,
    "num_acq_restarts": 5,
    "R_init_var": 1e-4,
}
methods: Dict[str, Any] = {
    "RBF+ARD": {
        "emb_dim": 1,
        "RBF": True,
        **default_args},
    "Maha_d2": {
        "emb_dim": 2,
        **default_args},
    "Maha_d1": {
        "emb_dim": 1,
        **default_args},
    "Maha_d2_twostep": {
        "emb_dim": 2,
        "two_step": True,
        **default_args},
    "Maha_d2_pinv": {
        "emb_dim": 2,
        "pseudo_inv": True,
        **default_args},
    "TuRBO1_init10": {
        "n_trust_regions": 1,
        "n_init": 10,
        "verbose": False,
    },
    "TuRBO5_init10": {
        "n_trust_regions": 5,
        "n_init": 10,
        "verbose": False,
    }
}


if __name__ == "__main__":
    # if argument is given, overwrite the n_jobs
    if len(sys.argv) == 2:
        n_jobs = int(sys.argv[1])

    while True:
        try:
            Parallel(n_jobs=n_jobs)([
                delayed(process_bo)(prob, name, args, i)
                for i in range(N_iter)
                for name, args in methods.items()
                for prob in problems
            ])
        except MemoryError:
            notify("Memory Error!")
            continue
        except Exception:
            notify(traceback.format_exc())
            break  # continue
        else:
            break

    notify("finished {}".format(sys.argv[0]))
