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
import qiskit_aer
import qiskit

statevector_simulator = qiskit_aer.Aer.get_backend('statevector_simulator')
unitary_simulator = qiskit_aer.Aer.get_backend('unitary_simulator')
aer_simulator = qiskit_aer.Aer.get_backend('aer_simulator')

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

def norm_log_mod(x, lambda_min=0.25, shift=0., scale=1.):
    return scale*(-np.log(np.absolute(x))/np.log(lambda_min) + shift)

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
    # seed = 0
    # rng = np.random.Generator(np.random.PCG64DXSM(seed))

    num_system_qubits = 2
    dimension = 2
    m0 = 0.5
    K = 1.5
    # norm = m0**2 + 16 * K**2
    norm = 32
    
    # N_polyfit = 1000
    # # N_quantfit = 100
    # drect = 100
    # dpoly = 10
    # d = drect+dpoly
    # k = 16
    # epsi = 0.0

    # # lambda_min = 0.23611111
    # lambda_min = 0.1
    # shift = 0.5
    # scale = 1.0

    # ## Polynomial fit ##################
    # # x_arr = np.linspace(lambda_min, 1, N_polyfit)
    # x_arr = (1-lambda_min)*np.random.random(size=N_polyfit) + lambda_min
    # # x_arr = np.random.normal(loc=0.0, scale=0.01, size=N_polyfit)
    # x_arr = np.concatenate([x_arr, -x_arr])
    # # print(x_arr)
    # y_arr = norm_log_mod(x_arr, lambda_min=lambda_min, shift=0.0, scale=scale)
    # poly_approx_no_step = Polynomial.cast(Chebyshev.fit(x_arr, y_arr, deg=list(range(0, dpoly+1, 2)), domain=[-1, 1], window=[-1, 1]))
    # # poly_approx_no_step = Polynomial.cast(Chebyshev.fit(x_arr, y_arr, deg=dpoly, domain=[-1, 1], window=[-1, 1]))
    # del x_arr, y_arr

    # # poly_rect_func = make_polynomial_rect(k=k, n=drect, shift=lambda_min/2)
    # x_arr = np.random.normal(scale=0.5, size=N_polyfit)
    # # x_arr = 2*np.random.random(size=N_polyfit)-1
    # # x_arr = (1-lambda_min)*np.random.random(size=N_polyfit) + lambda_min
    # y_arr = exact_rect(x_arr, k=k, shift=lambda_min/2)
    # f, ax = plt.subplots()
    # ax.plot(x_arr, y_arr, 'x')
    # poly_rect_func = Polynomial.cast(Chebyshev.fit(x_arr, y_arr, deg=list(range(0, drect+1, 2)), domain=[-1, 1], window=[-1, 1]))
    # ax.plot(x_arr, poly_rect_func(x_arr), '+')
    # plt.show()
    # poly_approx_with_step = poly_approx_no_step*poly_rect_func + shift

    # del x_arr, y_arr
    # ####################################

    # ## Quantum fit #####################
    # # x_arr = np.linspace(-1, 1, N_quantfit)
    # djtil = d//2 + 1
    # x_arr = np.cos([(2*j-1)*np.pi / (4 * djtil)
    #                 for j in range(1, djtil+1)])
    # # print(x_arr)
    # # assert False

    # target_func = poly_approx_with_step

    # to_minimize = functools.partial(
    #     loss_with_grad,
    #     target_func=target_func,
    #     d=d,
    #     x_arr=x_arr,
    # )

    # start = np.zeros(djtil)
    # start[0] = np.pi/4
    # hess_inv = 0.5 * np.ones(len(start))
    # hess_inv[-1] = 1
    # hess_inv = np.diag(hess_inv)
    # opt_result = minimize(to_minimize, x0=start,
    #                       method='BFGS', jac=True,
    #                       options={'gtol': 1e-24, 'hess_inv0': hess_inv, 'maxiter':10000})

    # test_x_arr = np.linspace(-1, 1, 2000)
    # # test_x_arr = np.sort(np.concatenate([test_x_arr, -test_x_arr]))
    # print(opt_result)
    # # exit()
    # trained_params = opt_result['x']
    # phis = np.array(list(trained_params) + list(trained_params)[::-1][1:])

    # target_vals = norm_log_mod(test_x_arr, lambda_min=lambda_min, shift=shift, scale=scale)
    # approx_vals, _ = quantum_func_and_grad(params=trained_params, x_arr=test_x_arr, is_d_odd=(d%2==1), use_fast_alg=True)
    # poly_vals = target_func(test_x_arr)

    # err_vals = approx_vals-target_vals
    # # err_vals[np.absolute(test_x_arr) < lambda_min] = 0

    # err2_vals = poly_vals-target_vals
    # # err2_vals[np.absolute(test_x_arr) < lambda_min] = 0
    # f, ax1 = plt.subplots()
    # ax1.plot(test_x_arr, target_vals, linestyle='solid')
    # ax1.plot(test_x_arr, approx_vals, linestyle='dashed')
    # ax1.set_ylim(-1.1, 0.1)
    # ax1.hlines(-1+shift, -1, 1)
    # ax1.vlines(-lambda_min, -5, 1, linewidth=1, linestyle='dashed', color='k')
    # ax1.vlines(lambda_min, -5, 1, linewidth=1, linestyle='dashed', color='k')
    # ax1.set_xlabel("$x$", size=17)
    # ax1.tick_params(which='both', labelsize=15)
    # # ax1.annotate("(a)", xy=(-1, -1), xytext=(-1, -1), fontsize=15)
    # plt.show()
    # exit()

    u1blockencoding = bef.BlockEncodeU1(num_system_qubits, dimension,
                                        m0, K, norm)
    # svec = statevector_simulator.run(
    #     qiskit.transpile(u1blockencoding, backend=statevector_simulator)
    # ).result().get_statevector(u1blockencoding)
    # svec_asarray = np.asarray(svec)
    
    # ## Option 2
    # # Note: `decompose` method is needed to handle the MCInc and MCDec instructions
    # # decomposed_be_circuit = be_circuit.decompose(gates_to_decompose=['MCInc', 'MCDec'], reps=2)
    # # svec = statevector_simulator.run(
    # #     decomposed_be_circuit
    # # ).result().get_statevector(be_circuit)
    # # svec_asarray = np.asarray(svec)

    # print(svec_asarray[:16])

    unitary_simulator = qiskit_aer.Aer.get_backend('unitary_simulator')

    unitary = unitary_simulator.run(
        qiskit.transpile(u1blockencoding, backend=unitary_simulator)
    ).result().get_unitary(u1blockencoding)

    unitary_asarray = np.asarray(unitary)

    print(unitary_asarray[:16, :16])
    exit()
    # n = 31
    # print(Operator(scalar_block_encoding).data[n*16:(n+1)*16, n*16:(n+1)*16])
    # print(Operator(scalar_block_encoding).data.shape)
    # print(n*16)
    # u1W = Operator(u1blockencoding).data[:16, :16]
    # # print(np.sort(np.linalg.eigvals(scalar_lap)))
    # # assert False
    # sign, ans = np.linalg.slogdet(u1W) / np.log(1/lambda_min)
    # print("logdet of lap = ", ans)
    # U, s, V = np.linalg.svd(u1W)
    # poly_mat = U.dot(np.diag(poly_approx_with_step(s)).dot(V))
    # print("trlog", np.trace(poly_mat) - 0.5*np.trace(np.eye(16)))
    # exit()
    # # print(scalar_lap)
    # # assert False
    # qc = qsvt.QuantumSignalProcess(u1blockencoding, phis, num_system_qubits, dimension)
    # # print(qc)
    # # print(phis[::-1])
    # full_pblock = Operator(qc).data
    # # np.save("./log_mat.npy", full_pblock)
    # # print("trace of full pblock = ", np.trace(full_pblock))
    # pblock = full_pblock[:16, :16]
    # print("trace of top block = ", -1*np.abs(np.trace(pblock)) - 0.5*np.trace(np.eye(16)))
    # # # symlog_mat = 0.5 * (logm(np.eye(16) + scalar_lap) + logm(np.eye(16) - scalar_lap))
    # # asymlog_mat = 0.5 * (logm(np.eye(16) + scalar_lap) - logm(np.eye(16) - scalar_lap))
    # # # print("trace of exact log = ", np.trace(symlog_mat))
    # # print("trace of exact log = ", np.trace(asymlog_mat))
    # # print(sinm(scalar_lap))
    # # print(pblock)
    
