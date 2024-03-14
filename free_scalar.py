#!/home/judah/miniconda3/bin/python

import numpy as np
import block_encode_funcs as bef
import qsvt_funcs as qsvt
# import math as mt
from scipy.optimize import minimize
from scipy.linalg import logm, cosm
# from functools import reduce
# from scipy.special import erf
# import qiskit as qis
# from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
# from itertools import product
from numpy.polynomial import chebyshev


if __name__ == "__main__":
    d = 40               # The degree of the polynomial
    xx = (np.random.random(size=d) * 2 *
          np.sqrt(np.exp(2)-1)/np.exp(1)) - (np.sqrt(np.exp(2)-1)/np.exp(1))
    # xx = 2*np.random.random(size=d)-1
    print(xx)
    print(qsvt.sym_log(xx))
    arr = chebyshev.chebfit(xx, qsvt.sym_log(xx), d)
    # arr = chebyshev.chebfit(xx, np.cos(xx), d)
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
    print(qsvt.sym_log(test_val))
    print(test(test_val))
    phis[0] = phis[0] + np.pi/4
    phis[-1] = phis[-1] + np.pi/4
    phis[1:-1] = phis[1:-1] + np.pi/2
    # assert False

    num_system_qubits = 2
    dimension = 2
    scalar_block_encoding = bef.BlockEncodeFreeScalar(num_system_qubits, dimension)
    scalar_lap = Operator(scalar_block_encoding).data[:16, :16]
    # print(scalar_lap)
    # assert False
    qc = qsvt.QuantumSignalProcess(scalar_block_encoding, phis, num_system_qubits, dimension)
    # print(qc)
    # print(phis[::-1])
    print((Operator(qc).data[:16, :16]))
    print(0.5 * (logm(np.eye(16) + scalar_lap) + logm(np.eye(16) - scalar_lap)))
    # print(cosm(scalar_lap))
    
