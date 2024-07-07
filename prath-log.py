#!/home/judah/miniconda3/bin/python

import numpy as np
import block_encode_funcs as bef
import qsvt_funcs as qsvt
# import qsvt_funcs_djtil as qsvt
# import math as mt
from scipy.optimize import minimize
from scipy.linalg import logm, cosm, sinm
# from functools import reduce
from scipy.special import erf
# import qiskit as qis
# from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
# from itertools import product
from numpy.polynomial import chebyshev, Chebyshev
# from scipy.interpolate import interp1d
from phase_estimation_crosscheck_implementation import quantum_func_and_grad, min_func_grad
import functools
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
import scipy
from scipy.optimize import minimize
from scipy.special import erf, iv, eval_chebyt
import matplotlib.pyplot as plt


params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

def poly_erf(x, k, n, shift=0):
    front = 2 * (2*k) * np.exp(- (2*k)**2 / 2) / np.sqrt(np.pi)
    first = iv(0, (2*k)**2 / 2) * (x-shift)/2
    series = np.array([iv(j, (2*k)**2 / 2) * (-1.)**j * (eval_chebyt(2*j+1, (x-shift)/2) / (2*j+1) - eval_chebyt(2*j-1, (x-shift)/2) / (2*j-1)) for j in range(1, (n-1)//2 + 1)])
    # print(series)
    return front * (first + np.sum(series, axis=0))

def poly_rect(x, k, n, shift=0):
    return 0.5 * (1. + poly_erf(x, k, n+1, shift=shift) + 1. + poly_erf(-1*x, k, n+1, shift=shift))

def make_polynomial_erf(k, n, shift=0):
    coefs_1 = [0]*(n+1)
    coefs_2 = [0]*(n-1)
    for j in range(1, (n-1)//2+1):
        tmp = iv(j, (2*k)**2 / 2) * (-1.)**j
        coefs_1[2*j+1] = tmp/(2*j+1)
        coefs_2[2*j-1] = tmp/(2*j-1)

    poly_erf = Chebyshev(coefs_1) - Chebyshev(coefs_2)
    poly_erf = Polynomial.cast(poly_erf)
    poly_erf += Polynomial([0, iv(0, (2*k)**2 / 2)])
    poly_erf *= 2 * (2*k) * np.exp(- (2*k)**2 / 2) / np.sqrt(np.pi)
    tmp = Polynomial([-shift/2, 1/2])
    poly_erf = poly_erf(tmp)

    return poly_erf

def make_polynomial_rect(k, n, shift=0):
    poly_erf = make_polynomial_erf(k=k, n=n+1, shift=shift)
    
    plus_x = Polynomial([0, 1])
    minus_x = Polynomial([0, -1])
    poly_rect = 0.5 * (2 + poly_erf(plus_x) + poly_erf(minus_x))

    return poly_rect


def norm_log_mod(x, kappa=2):
    return -(np.sign(x) / np.log(kappa)) * np.log(1 + ((1-kappa) / kappa) * np.abs(x))


# def new_log_mod(x, lambda_min=0.25, yshift=0., xshift=0, scale=1.):
#     -np.log(x-xshift) / np.log(lambda_min)

def loss_with_grad(params, target_func, d, x_arr):
    assert len(params) == d//2 + 1
    
    target_vals = target_func(x_arr)
    func_vals, grad_vals = quantum_func_and_grad(
        params, x_arr,
        is_d_odd = (d%2==1),
        use_fast_alg = True
    )
    
    loss = np.mean((func_vals-target_vals)**2)
    loss_grad = 2*np.mean((func_vals-target_vals) * grad_vals, axis=-1)
    
    return loss, loss_grad


def exact_rect(x, k, shift):
    return 0.5 * (2. + erf(k * (x-shift)) + erf(k * (-x-shift)))


if __name__ == "__main__":
    seed = 0
    rng = np.random.Generator(np.random.PCG64DXSM(seed))

    N_polyfit = 1000
    d = 25
    kappa = 100

    xx = np.linspace(-1, 1, 100)
    x_arr = xx
    y_arr = norm_log_mod(x_arr, kappa=kappa)
    poly_approx = Polynomial.cast(Chebyshev.fit(x_arr, y_arr,
                                                deg=d, domain=[-1, 1],
                                                window=[-1, 1]))

    f, ax = plt.subplots()
    ax.plot(x_arr, y_arr, 'x')
    ax.plot(x_arr, poly_approx(x_arr), '+')
    plt.show()
    del x_arr, y_arr
    # ####################################

    # ## Quantum fit #####################
    djtil = d//2 + 1
    x_arr = np.cos([(2*j-1)*np.pi / (4 * djtil)
                    for j in range(1, djtil+1)])

    target_func = poly_approx

    to_minimize = functools.partial(
        loss_with_grad,
        target_func=target_func,
        d=d,
        x_arr=x_arr,
    )

    start = np.zeros(djtil)
    start[0] = np.pi/4
    if (d % 2) == 1:
        hess_inv = 0.5*np.eye(len(start))
        opt_result = minimize(to_minimize, x0=start,
                              method='BFGS', jac=True,
                              options={'gtol': 1e-24,
                                       'hess_inv0': hess_inv,
                                       'maxiter': 10000})
        print(opt_result)
        phis = np.array(list(opt_result.x) + list(opt_result.x)[::-1])
    else:
        hess_inv = 0.5 * np.ones(len(start))
        hess_inv[-1] = 1
        hess_inv = np.diag(hess_inv)
        opt_result = minimize(to_minimize, x0=start,
                              method='BFGS', jac=True,
                              options={'gtol': 1e-24,
                                       'hess_inv0': hess_inv,
                                       'maxiter': 10000})
        print(opt_result)
        phis = np.array(list(opt_result.x) + list(opt_result.x)[::-1][1:])

    test_x_arr = np.linspace(-1, 1, 2000)
    trained_params = opt_result['x']

    target_vals = norm_log_mod(test_x_arr, kappa=kappa)
    approx_vals, _ = quantum_func_and_grad(params=trained_params,
                                           x_arr=test_x_arr,
                                           is_d_odd=(d % 2 == 1),
                                           use_fast_alg=True)
    poly_vals = target_func(test_x_arr)

    f, ax1 = plt.subplots()
    ax1.plot(test_x_arr, target_vals, linestyle='solid')
    ax1.plot(test_x_arr, approx_vals, linestyle='dashed')
    plt.show()
    exit()
