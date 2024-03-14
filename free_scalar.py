#!/home/judah/miniconda3/bin/python

import numpy as np
import block_encode_funcs as bef
import qsvt_funcs as qsvt
# import math as mt
from scipy.optimize import minimize
# from functools import reduce
# from scipy.special import erf
# import qiskit as qis
# from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
# from itertools import product
from numpy.polynomial import chebyshev


if __name__ == "__main__":
    d = 30               # The degree of the polynomial
    xx = (np.random.random(size=d) * 2 *
          np.sqrt(np.exp(2)-1)/np.exp(1)) - (np.sqrt(np.exp(2)-1)/np.exp(1))
    print(xx)
    arr = chebyshev.chebfit(xx, qsvt.asym_log(xx), d)
    print(arr)
    # # assert False

    def test(x):
        return chebyshev.chebval(x, arr)
    test_val = 0.5
    
    djtil = int(np.ceil((d+1) / 2))
    print("dtil = ", djtil)
    params0 = np.zeros((djtil,))
    params0[0] = np.pi / 4
    print(params0)

    if (d % 2) == 1:
        out = minimize(qsvt.min_func, params0, args=(test, d),
                       method='BFGS', jac=qsvt.grad,
                       options={'gtol': 1e-24})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1])
    else:
        out = minimize(qsvt.min_func, params0, args=(test, d),
                       method='BFGS', jac=qsvt.grad,
                       options={'gtol': 1e-24})
        print(out)
        phis = np.array(list(out.x) + list(out.x)[::-1][1:])        
    print(np.real(qsvt.output_mat(test_val, out.x, d))[0, 0])
    print(qsvt.asym_log(test_val))
    print(qsvt.sym_log(test_val))
    print(test(test_val))
    # assert False

    num_system_qubits = 2
    dimension = 2
    scalar_block_encoding = bef.BlockEncodeFreeScalar(num_system_qubits, dimension)
    qc = qsvt.QuantumSignalProcess(scalar_block_encoding, phis, num_system_qubits, dimension)
    print(qc)
    print(Operator(qc).data[:16, :16])
    
