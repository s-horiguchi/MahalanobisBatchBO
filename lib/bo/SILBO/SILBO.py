"""
Modified from: https://github.com/cjfcsjt/SILBO/blob/master/SS_BO.py
"""
import math
from typing import Callable, Optional

import numpy as np
from scipy.linalg import block_diag, svd
import ristretto.svd
from torch import Tensor
import torch
from pyDOE import lhs
import GPy
import cma
from ..utils import normalize, unnormalize
from .acquisition import ACfunction
from .kernel_inputs import InputY, InputX, InputPsi


def SILBO_TD(
    objective: Callable,
    bounds: Tensor,
    T: int,
    low_dim: int,
    init_X: Optional[Tensor] = None,
    init_Y: Optional[Tensor] = None,
    X_norm_unlabeled: Optional[Tensor] = None,
    n_unlabeled: Optional[int] = None,
    matrix_type='simple', kern_inp_type='Y',
    ARD=False, variance=1., length_scale=None,
    slice_number=None, k=3,
):
    """
    Note that this function tries to maximize the objectivem, not minimize.
    So the sign of the value of the objective is flipped.
    """
    high_dim = bounds.size(0)

    if init_X is not None and init_Y is not None and init_X.size(0) == init_Y.size(0):
        X = init_X.detach().clone().numpy()
        Y = -init_Y.detach().clone().numpy()
        initial_n = X.shape[0]
    else:
        raise ValueError("should provide init_X, init_Y")

    if X_norm_unlabeled is None:
        if n_unlabeled is None:
            n_unlabeled = initial_n
        X_norm_unlabeled = lhs(high_dim, n_unlabeled) * 2.0 - 1.0
    else:
        X_norm_unlabeled = X_norm_unlabeled.numpy()
        n_unlabeled = X_norm_unlabeled.shape[0]

    if slice_number is None:
        slice_number = low_dim+1

    X_norm = normalize(Tensor(X), bounds).numpy()
    # get project matrix B using Semi-LSIR
    B = SSIR(low_dim, X_norm, Y, X_norm_unlabeled, slice_number, k=k)
    Z = np.matmul(X_norm, B)

    # Specifying the input type of kernel
    if kern_inp_type == 'Y':
        kern_inp = InputY(B)
        input_dim = low_dim
    elif kern_inp_type == 'X':
        kern_inp = InputX(B)
        input_dim = high_dim
    elif kern_inp_type == 'psi':
        kern_inp = InputPsi(B)
        input_dim = high_dim
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return

    # Generating GP model
    kern = GPy.kern.Matern52(input_dim=input_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(kern_inp.evaluate(Z), Y, kernel=kern)
    # m.likelihood.variance = 1e-6
    ac = ACfunction(B, m, initial_size=initial_n, low_dimension=low_dim)

    for i in range(T):
        #Updating GP model
        m.set_XY(kern_inp.evaluate(Z), Y)
        m.optimize()


        #find x_new to max UCB(BX)
        es = cma.CMAEvolutionStrategy(high_dim * [0], 0.5, {'bounds': [-1, 1]})
        iter = 0
        u=[]
        ac.set_fs_true(max(Y))
        #_, maxD = ac.acfunctionEI(max(f_s), low_dim)
        if i != 0 and (i) % 20 == 0:
            print("update")

            while not es.stop() and iter !=2:
                iter+=1
                x_new = es.ask()
                es.tell(x_new, [ac.newfunction(x)[0] for x in x_new]) #set UCB or EI in newfunction() manually
                # if i != 0 and (i) % 10 == 0:
                u.append(es.result[0])
                 #es.disp()  # doctest: +ELLIPSIS
             #return candidate x_new
            maxx_norm = es.result[0].reshape((1,high_dim))
        else:
            while not es.stop() and iter !=2:
                iter+=1
                x_new = es.ask()
                es.tell(x_new, [ac.newfunction(x)[0] for x in x_new]) #set UCB or EI in newfunction() manually
               #es.disp()  # doctest: +ELLIPSIS
             #return candidate x_new
            maxx_norm = es.result[0].reshape((1,high_dim))

        maxz = np.matmul(maxx_norm, B)
        Z = np.append(Z, maxz, axis=0)
        X_norm = np.append(X_norm, maxx_norm, axis=0)
        maxx = unnormalize(Tensor(maxx_norm), bounds).numpy()
        X = np.append(X, maxx, axis=0)
        Y = np.append(Y, -objective(Tensor(maxx)).numpy(), axis=0)

        #update project matrix B
        if i != 0 and (i) % 20 == 0:
            print("update")
            # get top "n_unlabeled" from X
            Xidex = np.argsort(-Y, axis=0).reshape(-1)[:n_unlabeled]
            Y_special = Y[Xidex]
            X_norm_special = X_norm[Xidex]
            # get top unlabeled data from CMA-ES
            X_norm_unlabeled = np.array(u)
            B = SSIR(low_dim, X_norm_special, Y_special, X_norm_unlabeled, slice_number, k=k)
            Z = np.matmul(X_norm, B)
            ac.resetflag(B)


        # Collecting data
        print("iter = ", i, "maxobj = ", np.max(Y))

    # return Y as minimization
    return Tensor(X), Tensor(-Y)


def SSIR(r, xl, y, xu, slice_number, k, alpha=0.1):
    # xt = np.concatenate( (xl,xu) ,axis=0) #xtotal
    nl = xl.shape[0]  # label_sample_size
    nu = xu.shape[0]  # unlabel_sample_size
    dim = xl.shape[1]  # dimension
    xly = np.concatenate((xl, y), axis=1)
    sort_xl = xly[np.argsort(xly[:, -1])][:, :-1]
    xt = np.concatenate((sort_xl, xu), axis=0)
    xt_mean = np.mean(xt, axis=0)

    sizeOfSlice = int(nl/slice_number)

    # slice_mean = np.zeros((slice_number, dim))
    # smean_c = np.zeros((slice_number, dim))
    W = []
    for i in range(slice_number):
        if i == slice_number - 1:
            temp = sort_xl[i*sizeOfSlice:, :]
        else:
            temp = sort_xl[i*sizeOfSlice:(i+1) * sizeOfSlice, :]
        ni = temp.shape[0]
        numNNi = min(ni, k)
        W.append(getWbyKNN(temp, numNNi, numNNi))
    W = block_diag(*W)
    zero = np.zeros((nu, nu))
    W = block_diag(W, zero)
    # fast svd
    cX = xt - np.tile(xt_mean, ((nl + nu), 1))
    # R = np.matmul(W,cX)
    Ra = np.matmul(cX.T, W)
    # U,S,Vh = svd.compute_rsvd(R,6)
    U1, S1, V1h = ristretto.svd.compute_rsvd(Ra, r)
    
    W = getWbyKNN(xt, k, 1)
    L = laplacian_matrix(W)
    Il = np.identity(nl)
    I = block_diag(Il, zero)  # noqa
    M = I + alpha * L + 0.01 * np.identity((nl + nu))

    L = np.linalg.cholesky(M)
    u, s, vh = np.linalg.svd(M)
    ss = np.diag(np.sqrt(s))
    J = np.matmul(ss, vh)  # noqa
    # J*cX*U*S^-1

    aa = np.matmul(np.diag(1.0/S1), U1.T)
    bb = np.matmul(cX.T, L)
    cc = np.matmul(aa, bb)

    u2, s2, vh2 = np.linalg.svd(cc)
    eii = np.matmul(np.matmul(U1, np.diag(1.0 / S1)), u2)

    return eii


def distance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def getWbyKNN(X, k, a=0):
    num = X.shape[0]
    S = np.zeros((num, num))
    W = np.zeros((num, num))
    for i in range(num):
        for j in range(i+1, num):
            S[i][j] = 1.0 * distance(X[i], X[j])
            S[j][i] = S[i][j]
    for i in range(num):
        index_array = np.argsort(S[i])
        W[i][index_array[1:k + 1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W + W) / 2
    if a != 0:
        for i in range(num):
            for j in range(num):
                if W[i][j] != 0:
                    W[i][j] = 1 / a
    return W


def laplacian_matrix(M, normalize=False):
    # compute the D = sum(A)
    D = np.sum(M, axis=1)
    # compute the laplacian matrix: L=D-A
    L = np.diag(D)-M
    # normalize
    if normalize:
        sqrtD = np.diag(1.0/(D**(0.5)))
        return np.dot(np.dot(sqrtD, L), sqrtD)
    return L
