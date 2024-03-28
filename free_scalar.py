#!/home/judah/miniconda3/bin/python

import numpy as np
import block_encode_funcs as bef
import qsvt_funcs as qsvt
# import math as mt
from scipy.optimize import minimize
from scipy.linalg import logm, cosm, sinm
# from functools import reduce
# from scipy.special import erf
# import qiskit as qis
# from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
# from itertools import product
from numpy.polynomial import chebyshev, Chebyshev


if __name__ == "__main__":
    d = 38               # The degree of the polynomial
    N = d
    # xx = (np.random.random(size=N) * 2 *
    #       np.sqrt(np.exp(2)-1)/np.exp(1)) - (np.sqrt(np.exp(2)-1)/np.exp(1))
    # sym_xx = (2*np.random.random(size=N)-1) * (np.sqrt(np.exp(2)-1)/np.exp(1))
    # asym_xx = (2*np.random.random(size=N)-1) * (np.exp(2)-1)/(np.exp(2)+1)
    # assert all(np.abs(qsvt.asym_log(asym_xx)) < 1)
    # arr = chebyshev.chebfit(asym_xx, qsvt.asym_log(asym_xx), d)
    xx = np.random.random(size=N)
    arr = chebyshev.chebfit(xx, np.sin(xx), d)
    print(arr)
    # assert False

    def test(x):
        return chebyshev.chebval(x, arr)
    test_val = 0.5
    
    djtil = int(np.ceil((d+1) / 2))
    print("dtil = ", djtil)
    params0 = np.zeros((djtil,))
    params0[0] = np.pi / 4
    print(params0)

    if (d % 2) == 1:
        hess_inv = 0.5*np.eye(len(params0))
        out = minimize(qsvt.min_func, params0, args=(test, d),
                       method='BFGS', jac=qsvt.grad,
                       options={'gtol': 1e-24, 'hess_inv0': hess_inv})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1])
    else:
        hess_inv = 0.5 * np.ones(len(params0))
        hess_inv[-1] = 1
        hess_inv = np.diag(hess_inv)
        out = minimize(qsvt.min_func, params0, args=(test, d),
                       method='BFGS', jac=qsvt.grad,
                       options={'gtol': 1e-24, 'hess_inv0': hess_inv})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1][1:])
    print(np.real(qsvt.output_mat(test_val, out.x, d))[0, 0])
    # print(qsvt.asym_log(test_val))
    # print(np.cos(test_val))
    print(np.sin(test_val))
    # print(qsvt.sym_log(test_val))
    print(test(test_val))

    # phis[0] = phis[0] + np.pi/4
    # phis[-1] = phis[-1] + np.pi/4
    # phis[1:-1] = phis[1:-1] + np.pi/2
    # assert False

    num_system_qubits = 2
    dimension = 2
    scalar_block_encoding = bef.BlockEncodeFreeScalar(num_system_qubits, dimension)
    # n = 31
    # print(Operator(scalar_block_encoding).data[n*16:(n+1)*16, n*16:(n+1)*16])
    # print(Operator(scalar_block_encoding).data.shape)
    # print(n*16)
    scalar_lap = Operator(scalar_block_encoding).data[:16, :16]
    sign, ans = np.linalg.slogdet(np.eye(16) + scalar_lap)
    print("logdet of lap = ", ans)
    # print(scalar_lap)
    # assert False
    qc = qsvt.QuantumSignalProcess(scalar_block_encoding, phis, num_system_qubits, dimension)
    # print(qc)
    # print(phis[::-1])
    full_pblock = Operator(qc).data
    # np.save("./log_mat.npy", full_pblock)
    # print("trace of full pblock = ", np.trace(full_pblock))
    pblock = full_pblock[:16, :16]
    # print("trace of top block = ", np.trace(pblock))
    # # symlog_mat = 0.5 * (logm(np.eye(16) + scalar_lap) + logm(np.eye(16) - scalar_lap))
    # asymlog_mat = 0.5 * (logm(np.eye(16) + scalar_lap) - logm(np.eye(16) - scalar_lap))
    # # print("trace of exact log = ", np.trace(symlog_mat))
    # print("trace of exact log = ", np.trace(asymlog_mat))
    print(sinm(scalar_lap))
    print(pblock)
    
