import os
import time
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from bo import DPP_Mahalanobis
from bo.ALEBO import ALEBO, REMBO, HeSBO
from bo.SILBO import SILBO_TD
from bo.turbo import TurboM
from problem import Problem
from utils import notify


def process_bo(
        prob: Problem,
        name: str,
        args: Mapping[str, Any],
        i: int,
        dstdir: str = "./result_bo/"
) -> None:
    fname = os.path.join(
        dstdir,
        f"{prob}_{name}_{i}.json")
    if os.path.exists(fname):
        return

    torch.manual_seed(i)
    np.random.seed(i)

    list_name = []
    list_X = []
    list_Y = []
    list_time = []
    list_iter = []
    objective = prob.objective

    notify(f"start [{prob}] {name} ({i})")
    start_time = time.time()
    try:
        if name.startswith("SILBO") and prob.B == 1:
            X, Y = SILBO_TD(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T,
                init_X=prob.design_samples,
                init_Y=prob.design_values,
                **args
            )
        elif name.startswith("REMBO") and prob.B == 1:
            X, Y = REMBO(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T,
                num_init=prob.num_init,
                **args
            )
        elif name.startswith("HeSBO") and prob.B == 1:
            X, Y = HeSBO(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T,
                num_init=prob.num_init,
                **args
            )
        elif name.startswith("ALEBO") and prob.B == 1:
            X, Y = ALEBO(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T,
                num_init=prob.num_init,
                **args
            )
        elif name.startswith("TuRBO"):
            X, Y = TurboM(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T + prob.num_init,
                B=prob.B,
                **args  # n_init and n_trust_regions are required
            )
        else:
            X, Y = DPP_Mahalanobis(
                objective=objective.objective_noisy,
                bounds=objective.bounds,
                T=prob.T,
                batch_size=prob.B,
                init_X=prob.design_samples,
                init_Y=prob.design_values,
                **args
            )
    except Exception:
        notify(traceback.format_exc())
        notify(f"skipping [{prob}] {name} ({i})")
        return
    elapsed = time.time() - start_time
    list_name.append(name)
    list_iter.append(i)
    list_X.append(X.to('cpu').detach().numpy())
    list_Y.append(Y.to('cpu').detach().numpy())
    list_time.append(elapsed)
    notify("finish [{}] {} ({}): bestY:{} time:{}".format(
        prob, name, i, np.min(list_Y[-1]), elapsed)
    )

    df = pd.DataFrame({
        "name": pd.Series(list_name),
        "iter": pd.Series(list_iter),
        "X": pd.Series(list_X),
        "Y": pd.Series(list_Y),
        "time": pd.Series(list_time),
    })
    df.to_json(fname)
    return
