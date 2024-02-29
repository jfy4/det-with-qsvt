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


def min_func(params, targ_func, d):
    """
    The function used for minimization.
    
    Parameters
    ----------
    params    : the angles that parameterize qsvt.
    targ_func : the polynomial being approximated.
    d         : the degree of the polynomial.

    Returns
    -------
    val : the value of the cost function.

    """
    djtil = int(np.ceil((d+1) / 2))
    assert len(params) == djtil
    cheby_zeros = np.cos([(2*j-1)*np.pi / (4 * djtil)
                          for j in range(1, djtil+1)])
    if (d % 2) == 1:
        # phis = np.array(list(params) + list(params)[::-1])
        Upis = [np.array([[np.exp(1j * params[a]), 0],
                          [0, np.exp(-1j * params[a])]])
                for a in range(len(params))]
        want = 0
        start = Upis[0]
        for i in range(djtil):
            W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                          [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
            prod_list = [W.dot(Upis[a]) for a in range(1, len(params))]
            one_mat = reduce(np.dot, prod_list)
            one_mat = start.dot(one_mat)
            one_mat = one_mat.dot(W.dot(one_mat.transpose()))
            # one_mat = one_mat.dot(one_mat.transpose())
            want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
        want /= djtil
        return want
        # print(len(phis))
    else:
        # phis = np.array(list(params) + list(params)[::-1][1:])
        Upis = [np.array([[np.exp(1j * params[a]), 0],
                          [0, np.exp(-1j * params[a])]])
                for a in range(len(params))]
        want = 0
        start = Upis[0]
        for i in range(djtil):
            W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                          [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
            prod_list = [W.dot(Upis[a]) for a in range(1, len(params)-1)]
            one_mat = reduce(np.dot, prod_list)
            one_mat = start.dot(one_mat.dot(W))
            one_mat = one_mat.dot(Upis[-1].dot(one_mat.transpose()))
            # one_mat = one_mat.dot(one_mat.transpose())
            want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
        want /= djtil
        return want

        # print(len(phis))
    # upis = [np.array([[np.exp(1j * phis[a]), 0],
    #                   [0, np.exp(-1j * phis[a])]])
    #         for a in range(len(phis))]
    # want = 0
    # start = Upis[0]
    # for i in range(djtil):
    #     W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
    #                   [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
    #     prod_list = [W.dot(Upis[a]) for a in range(1, len(phis))]
    #     one_mat = reduce(np.dot, prod_list)
    #     one_mat = start.dot(one_mat)
    #     want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
    # want /= djtil
    # return want


def output_mat(x, params, d):
    """
    Creates the little block-encoded matrix from the
    phases.
    
    Parameters
    ----------
    x         : the value to evaluate the matrix at
    params    : the angles that parameterize qsvt.
    d         : the degree of the polynomial.

    Returns
    -------
    matrix : the value of the cost function.

    """
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
        """
        The quantum circuit for the Oc operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
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
        """
        The quantum circuit for the OA operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
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
        self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
                                         ctrl_state='001'),
                     inplace=True, qubits=(*block, anc))
        self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock-1,
                                         ctrl_state='01'),
                     inplace=True, qubits=(*block[1:], anc))
        # self.compose(RYGate(0.3).control(num_ctrl_qubits=nblock-2,
        #                                  ctrl_state='1'),
        #              inplace=True, qubits=(*block[2:], anc))
        self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
                                         ctrl_state='100'),
                     inplace=True, qubits=(*block, anc))
        # for a, bs in enumerate(bin_states[1:s]):
        #     self.compose(RYGate(-0.1).control(num_ctrl_qubits=nblock,
        #                                       ctrl_state=bs),
        #                  inplace=True, qubits=(*block, anc))


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
        """
        The quantum circuit for the Diffusion operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
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
        """
        The full quantum circuit for block-encoding.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
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
        """
        The quantum circuit for modular addition.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'

        """

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
        """
        The quantum circuit for the phase part of qsvt.

        Parameters
        ----------
        angle : The angle of the phase
        nsys  : the number of qubits in the 'system'
        s     : the number of nonzero elements in a row/column

        """
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
    def __init__(self, phis, nsys, s):
        """
        The full quantum circuit for qsvt.

        Parameters
        ----------
        phis : the phases used in the rotations
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
        super().__init__()
        # djtil = int(np.ceil((d+1) / 2))
        # assert len(params) == djtil
        # if (d % 2) == 1:
        #     phis = np.array(list(params) + list(params)[::-1])
        # else:
        #     phis = np.array(list(params) + list(params)[::-1][1:])

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


class RealPart(qis.QuantumCircuit):
    def __init__(self, phis, nsys, s):
        """
        The quantum circuit for the real part of the qsvt.

        Parameters
        ----------
        phis : the phases used in the rotations
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

        """
        super().__init__()
        # if (d % 2) == 1:
        #     phis = np.array(list(params) + list(params)[::-1])
        # else:
        #     phis = np.array(list(params) + list(params)[::-1][1:])
        phis_minus = -1*np.asarray(phis)
        phis_minus[0] = phis_minus[0] + np.pi/2
        phis_minus[-1] = phis_minus[-1] - np.pi/2
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))

        xreg = qis.QuantumRegister(nsys, name='x')
        yreg = qis.QuantumRegister(nsys, name='y')
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        anc3 = qis.QuantumRegister(1, name='anc3')
        self.add_register(xreg)
        self.add_register(yreg)
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.add_register(anc3)

        Uphi = QuantumSignalProcess(phis, nsys, s)
        Uphiminus = QuantumSignalProcess(phis_minus, nsys, s)
        self.h(anc3)
        self.compose(Uphi.control(num_ctrl_qubits=1, ctrl_state='0'),
                     inplace=True, qubits=[*anc3, *xreg, *yreg, *block, *anc, *anc2])
        self.compose(Uphiminus.control(num_ctrl_qubits=1),
                     inplace=True, qubits=[*anc3, *xreg, *yreg, *block, *anc, *anc2])
        self.h(anc3)


def cheby_coeff(func, d):
    """
    Computing the chebyshev coeffcients.

    """
    theta = np.zeros((2*d,))
    for i in range(2*d):
        theta[i] = i*np.pi/d
    f = func(np.cos(theta))
    c = np.fft.fft(f)
    c = np.real(c)
    c = c[:d+1]
    c[1:-1] = c[1:-1]*2
    c = c / (2*d)
    return c

        
if __name__ == "__main__":
    # print(cheby_coeff(np.cos, 4))
    d = 30               # The degree of the polynomial
    arr = cheby_coeff(np.cos, d)
    # xx = np.linspace(-1, 1, 100000)
    # arr = chebyshev.chebfit(xx, np.cos(1 * xx), d)  # using the errorfunction to approximate sign
    print(arr)
    # assert False

    def test(x):
        return chebyshev.chebval(x, arr)
    
    print("dtil = ", int(np.ceil((d+1) / 2)))
    params0 = np.zeros((int(np.ceil((d+1) / 2)),))
    params0[0] = np.pi / 4
    print(params0)
    out = minimize(min_func, params0, args=(test, d),
                   method='BFGS', jac='3-point',
                   options={'gtol':1e-15})
    print(out)
    print(output_mat(-0.1, out.x, d)[0,0])
    print(np.cos(-0.1))
    print(test(-0.1))
    assert False

    if (d % 2) == 1:
        phis = np.array(list(out.x) + list(out.x)[::-1])
    else:
        phis = np.array(list(out.x) + list(out.x)[::-1][1:])
    qsp = RealPart(phis, 2, 5)
    print(qsp)
    # qsp = QuantumSignalProcess(phis, 2, 5)
    # print(qsp)
    print(Operator(qsp).data[:16, :16])
    print(BlockEncode(2, 5))
    print(Operator(BlockEncode(2, 5)).data[:16, :16])
    # lap = Operator(BlockEncode(2, 5)).data[:16, :16]
    # print(erf(lap))
    # U, s, Vh = np.linalg.svd(lap)
    # sign_lap = U.dot(np.diag(erf(s)).dot(Vh))
    # print(sign_lap)
    
