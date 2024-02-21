#!/home/judah/miniconda3/bin/python

import numpy as np
# import math as mt
from scipy.optimize import minimize, HessianUpdateStrategy
from functools import reduce
from scipy.special import erf
import qiskit as qis
from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
from itertools import product
from numpy.polynomial import chebyshev

# test

def min_func(params, targ_func, d):
    djtil = int(np.ceil((d+1) / 2))
    assert len(params) == djtil
    cheby_zeros = np.cos([(2*j-1)*np.pi / (4 * djtil)
                          for j in range(1, djtil+1)])
    if (d % 2) == 1:
        phis = np.array(list(params) + list(params)[::-1])
        # print(len(phis))
    else:
        phis = np.array(list(params) + list(params)[::-1][1:])
        # print(len(phis))
    Upis = [np.array([[np.exp(1j * phis[a]), 0],
                      [0, np.exp(-1j * phis[a])]])
            for a in range(len(phis))]
    want = 0
    start = Upis[0]
    for i in range(djtil):
        W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                      [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
        prod_list = [W.dot(Upis[a]) for a in range(1, len(phis))]
        one_mat = reduce(np.dot, prod_list)
        one_mat = start.dot(one_mat)
        want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
    want /= djtil
    return want


def output_mat(x, params, d):
    if (d % 2) == 1:
        phis = np.array(list(params) + list(params)[::-1])
    else:
        phis = np.array(list(params) + list(params)[::-1][1:])
    Upis = [np.array([[np.exp(1j * phis[a]), 0],
                      [0, np.exp(-1j * phis[a])]])
            for a in range(len(phis))]
    start = Upis[0]
    W = np.array([[x, 1j * np.sqrt(1 - x**2)],
                  [1j * np.sqrt(1 - x**2), x]])
    prod_list = [W.dot(Upis[a]) for a in range(1, len(phis))]
    one_mat = reduce(np.dot, prod_list)
    one_mat = start.dot(one_mat)
    return one_mat




# class Oc(qis.QuantumCircuit):
#     def __init__(self, nsys, s):
#         super().__init__()
#         nblock = int(np.ceil(np.log2(s)))
#         bin_states = product(['0', '1'], repeat=nblock)
#         bin_states = [''.join(x) for x in bin_states]
#         # print(bin_states)
#         sys = qis.QuantumRegister(nsys, name='work')
#         block = qis.QuantumRegister(nblock, name='block')
#         anc = qis.QuantumRegister(1, name='anc')
#         self.add_register(anc)
#         self.add_register(block)
#         self.add_register(sys)
#         for a, bs in enumerate(bin_states[1:(s-1)//2+1]):
#             self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
#                                               ctrl_state=bs),
#                          inplace=True, qubits=(*block, *sys))
#         for bs in bin_states[(s-1)//2+1:s]:
#             self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
#                                                         ctrl_state=bs),
#                          inplace=True, qubits=(*block, *sys))


class Oc(qis.QuantumCircuit):
    def __init__(self, nsys, s):
        super().__init__()
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        bin_states = product(['0', '1'], repeat=nblock)
        bin_states = [''.join(x) for x in bin_states]
        # print(bin_states)
        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        # for a, bs in enumerate(bin_states[1:(s-1)//2+1]):
        self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
                                          ctrl_state='001'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
                                          ctrl_state='010'),
                     inplace=True, qubits=(*block, *yreg))
        # for bs in bin_states[(s-1)//2+1:s]:
        self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
                                                    ctrl_state='011'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
                                                    ctrl_state='100'),
                     inplace=True, qubits=(*block, *yreg))


# class Oc(qis.QuantumCircuit):
#     def __init__(self, nsys, s):
#         super().__init__()
#         if s == 1:
#             nblock = 1
#         else:
#             nblock = int(np.ceil(np.log2(s)))
#         bin_states = product(['0', '1'], repeat=nblock)
#         bin_states = [''.join(x) for x in bin_states]
#         # print(bin_states)
#         xreg = qis.QuantumRegister(nsys, name='x')
#         block = qis.QuantumRegister(nblock, name='block')
#         anc = qis.QuantumRegister(1, name='anc')
#         self.add_register(xreg)
#         self.add_register(block)
#         self.add_register(anc)
#         # self.add_register(yreg)
#         # for a, bs in enumerate(bin_states[1:(s-1)//2+1]):
#         self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
#                                           ctrl_state='10'),
#                      inplace=True, qubits=(*block, *xreg))
#         # self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
#         #                                   ctrl_state='010'),
#         #              inplace=True, qubits=(*block, *yreg))
#         # # for bs in bin_states[(s-1)//2+1:s]:
#         self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
#                                                     ctrl_state='01'),
#                      inplace=True, qubits=(*block, *xreg))
#         # self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
#         #                                             ctrl_state='100'),
#         #              inplace=True, qubits=(*block, *yreg))
        

# class OA(qis.QuantumCircuit):
#     def __init__(self, nsys, s):
#         super().__init__()
#         nblock = int(np.ceil(np.log2(s)))
#         bin_states = product(['0', '1'], repeat=nblock)
#         bin_states = [''.join(x) for x in bin_states]
#         # print(bin_states)
#         sys = qis.QuantumRegister(nsys, name='work')
#         block = qis.QuantumRegister(nblock, name='block')
#         anc = qis.QuantumRegister(1, name='anc')
#         self.add_register(anc)
#         self.add_register(block)
#         self.add_register(sys)
#         self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
#                                          ctrl_state='0'*nblock),
#                      inplace=True, qubits=(*block, anc))
#         for a, bs in enumerate(bin_states[1:s]):
#             self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
#                                              ctrl_state=bs),
#                          inplace=True, qubits=(*block, anc))


class OA(qis.QuantumCircuit):
    def __init__(self, nsys, s):
        super().__init__()
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        bin_states = product(['0', '1'], repeat=nblock)
        bin_states = [''.join(x) for x in bin_states]
        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.compose(RYGate(0.3).control(num_ctrl_qubits=nblock,
                                         ctrl_state='0'*nblock),
                     inplace=True, qubits=(*block, anc))
        for a, bs in enumerate(bin_states[1:s]):
            self.compose(RYGate(-0.1).control(num_ctrl_qubits=nblock,
                                              ctrl_state=bs),
                         inplace=True, qubits=(*block, anc))


# class OA(qis.QuantumCircuit):
#     def __init__(self, nsys, s):
#         super().__init__()
#         if s == 1:
#             nblock = 1
#         else:
#             nblock = int(np.ceil(np.log2(s)))
#         bin_states = product(['0', '1'], repeat=nblock)
#         bin_states = [''.join(x) for x in bin_states]
#         # print(bin_states)
#         xreg = qis.QuantumRegister(nsys, name='x')
#         # yreg = qis.QuantumRegister(nsys, name='y')
#         block = qis.QuantumRegister(nblock, name='block')
#         anc = qis.QuantumRegister(1, name='anc')
#         self.add_register(xreg)
#         self.add_register(yreg)
#         self.add_register(block)
#         self.add_register(anc)
#         self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
#                                          ctrl_state='00'),
#                      inplace=True, qubits=(*block, anc))
#         self.compose(RYGate(0.3).control(num_ctrl_qubits=nblock,
#                                          ctrl_state='10'),
#                      inplace=True, qubits=(*block, anc))
#         self.compose(RYGate(0.5).control(num_ctrl_qubits=nblock,
#                                          ctrl_state='01'),
#                      inplace=True, qubits=(*block, anc))
#         # for a, bs in enumerate(bin_states[1:s]):
#         #     self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
#         #                                      ctrl_state=bs),
#         #                  inplace=True, qubits=(*block, anc))
            

class Diffusion(qis.QuantumCircuit):
    def __init__(self, nsys, s):
        super().__init__()
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.h(block)


class BlockEncode(qis.QuantumCircuit):
    def __init__(self, nsys, s):
        super().__init__()
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.compose(Diffusion(nsys, s), inplace=True)
        self.compose(OA(nsys, s), inplace=True)
        self.compose(Oc(nsys, s), inplace=True)
        self.compose(Diffusion(nsys, s), inplace=True)
            

class Lshift(qis.QuantumCircuit):
    def __init__(self, nsys):
        super().__init__()
        sys = qis.QuantumRegister(nsys, name='work')
        self.add_register(sys)
        for i in range(1, nsys):
            # self.compose(MCXGate(num_ctrl_qubits=nsys-i), inplace=True,
            #              qubits=[*sys][i-1:])
            self.compose(MCXGate(num_ctrl_qubits=nsys-i), inplace=True)
        self.x(sys[0])


class Uproj(qis.QuantumCircuit):
    def __init__(self, angle, nsys, s):
        super().__init__()
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        # self.compose(MCXGate(num_ctrl_qubits=nblock+1,
        #                      ctrl_state='0'*(nblock+1)),
        #              inplace=True, qubits=[*block, *anc, *anc2])
        # self.rz(2*angle, anc2)
        # self.compose(MCXGate(num_ctrl_qubits=nblock+1,
        #                      ctrl_state='0'*(nblock+1)),
        #              inplace=True, qubits=[*block, *anc, *anc2])
        self.cx(anc, anc2, ctrl_state='0')
        self.rz(2*angle, anc2)
        self.cx(anc, anc2, ctrl_state='0')


class QuantumSignalProcess(qis.QuantumCircuit):
    def __init__(self, d, params, nsys, s):
        super().__init__()
        # djtil = int(np.ceil((d+1) / 2))
        # assert len(params) == djtil
        if (d % 2) == 1:
            phis = np.array(list(params) + list(params)[::-1])
        else:
            phis = np.array(list(params) + list(params)[::-1][1:])

        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))

        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        UA = BlockEncode(nsys, s)
        for i in range(len(phis)-1, 0, -1):
            self.compose(Uproj(phis[i], nsys, s), inplace=True)
            self.compose(UA, inplace=True)
        self.compose(Uproj(phis[0], nsys, s), inplace=True)
        
        
if __name__ == "__main__":
    d = 5                       # The degree of the polynomial
    xx = np.linspace(-1, 1, 10000)
    arr = chebyshev.chebfit(xx, xx, d)
    print(arr)

    def test(x):
        return chebyshev.chebval(x, arr)
    print("dtil = ", int(np.ceil((d+1) / 2)))
    params0 = np.zeros((int(np.ceil((d+1) / 2)),))
    params0[0] = np.pi / 4
    print(params0)
    out = minimize(min_func, params0, args=(test, d), method='BFGS')
    print(out)
    print(output_mat(-0.1, out.x, d))
    print(test(-0.1))
    qsp = QuantumSignalProcess(d, out.x, 2, 5)
    print(qsp)
    print(Operator(qsp).data[:16, :16])
    # print(BlockEncode(2, 5))
    print(Operator(BlockEncode(2, 5)).data[:16, :16])
    
