# Mahalanobis Batch BO


## Setup

    pip install requirements.txt


Rover trajectory objective function is available at [zi-w/Ensemble-Bayesian-Optimization](https://github.com/zi-w/Ensemble-Bayesian-Optimization). Place `rover_function.py` and `rover_utils.py` under the directory `lib/problem/box2d`. You also need to install [pybox2d](https://github.com/pybox2d/pybox2d).

For Anaconda,

    conda install -c conda-forge pybox2d

Otherwise, build pybox2d from the source.
See https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md


To run SILBO, get the source code from [cjfcsjt/SILBO](https://github.com/cjfcsjt/SILBO) and place them under `lib/bo/SILBO`.
You may need to manually install [ristretto](https://github.com/erichson/ristretto), GPy, matplotlib and cma.


To run ALEBO, install with pip:

    pip install ax-platform==v0.2.2


To run TuRBO, get the source code from [uber-research/TuRBO](https://github.com/uber-research/TuRBO) and place them under `lib/bo/turbo`.


## Run BO experiment

For example,

    cd bo_exp
    python bo_Branin_D100.py

