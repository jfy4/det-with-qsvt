#!/home/judah/miniconda3/bin/python

import numpy as np
# import math as mt
from scipy.optimize import minimize, HessianUpdateStrategy
from functools import reduce
from scipy.special import erf
import qiskit as qis
from qiskit.circuit.library import MCXGate, RYGate, XGate
from qiskit.quantum_info import Operator
from itertools import product
from numpy.polynomial import chebyshev
import math as mt
from block_encode_funcs import *


# pauli z matrix
sz = np.array([[1, 0],
               [0, -1]])


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
            want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
        want /= djtil
        return want
    else:
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
            want += np.abs(np.real(one_mat[0, 0]) - targ_func(cheby_zeros[i]))**2
        want /= djtil
        return want


def grad(params, targ_func, d):
    """
    The gradient of the cost function.
    
    Parameters
    ----------
    params    : the angles that parameterize qsvt.
    targ_func : the polynomial being approximated.
    d         : the degree of the polynomial.

    Returns
    -------
    vec : the vector of the gradient.

    """
    djtil = int(np.ceil((d+1) / 2))
    assert len(params) == djtil
    cheby_zeros = np.cos([(2*j-1)*np.pi / (4 * djtil)
                          for j in range(1, djtil+1)])
    Upis = np.array([np.array([[np.exp(1j * params[a]), 0],
                               [0, np.exp(-1j * params[a])]])
                     for a in range(len(params))])
    Thetas = 1j*np.tensordot(sz, Upis, axes=([1], [1])).transpose((1, 0, 2))
    want = np.zeros((len(params),))
    start = Upis[0]
    if (d % 2) == 1:
        val = 0
        for i in range(djtil):
            W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                          [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
            prod_list = [W.dot(Upis[a]) for a in range(1, len(params))]
            one_mat = reduce(np.dot, prod_list)
            two_mat = Thetas[0, :, :].dot(one_mat)
            one_mat = start.dot(one_mat)
            pure_complete = one_mat.dot(W.dot(one_mat.transpose()))
            diff = (np.real(pure_complete[0, 0]) - targ_func(cheby_zeros[i]))
            deriv = (two_mat.dot(W.dot(one_mat.transpose())) +
                     one_mat.dot(W.dot(two_mat.transpose())))
            val += diff * np.real(deriv)[0, 0]
        val *= (2 / djtil)
        want[0] = val
        for j in range(1, djtil):
            val = 0
            for i in range(djtil):
                W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                              [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
                prod_list = [W.dot(Upis[a]) for a in range(1, len(params))]
                one_mat = reduce(np.dot, prod_list)
                one_mat = start.dot(one_mat)
                pure_complete = one_mat.dot(W.dot(one_mat.transpose()))
                diff = (np.real(pure_complete[0, 0]) - targ_func(cheby_zeros[i]))
                try:
                    prod_list_left = reduce(np.dot, [W.dot(Upis[a]) for a in range(1, j)])
                except TypeError:
                    prod_list_left = np.eye(2)
                mid = W.dot(Thetas[j, :, :])
                try:
                    prod_list_right = reduce(np.dot, [W.dot(Upis[a]) for a in range(j+1, len(params))])
                except TypeError:
                    prod_list_right = np.eye(2)
                two_mat = prod_list_left.dot(mid.dot(prod_list_right))
                two_mat = start.dot(two_mat)
                deriv = (two_mat.dot(W.dot(one_mat.transpose())) +
                         one_mat.dot(W.dot(two_mat.transpose())))
                val += diff * np.real(deriv)[0, 0]
            val *= (2 / djtil)
            want[j] = val
    else:
        val = 0
        for i in range(djtil):
            W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                          [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
            prod_list = [W.dot(Upis[a]) for a in range(1, len(params)-1)]
            one_mat = reduce(np.dot, prod_list)
            two_mat = Thetas[0, :, :].dot(one_mat.dot(W))
            one_mat = start.dot(one_mat.dot(W))
            pure_complete = one_mat.dot(Upis[-1].dot(one_mat.transpose()))
            diff = (np.real(pure_complete[0, 0]) - targ_func(cheby_zeros[i]))
            deriv = (two_mat.dot(Upis[-1].dot(one_mat.transpose())) +
                     one_mat.dot(Upis[-1].dot(two_mat.transpose())))
            val += diff * np.real(deriv)[0, 0]
        val *= (2 / djtil)
        want[0] = val
        for j in range(1, djtil-1):
            val = 0
            for i in range(djtil):
                W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                              [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
                prod_list = [W.dot(Upis[a]) for a in range(1, len(params)-1)]
                one_mat = reduce(np.dot, prod_list)
                one_mat = start.dot(one_mat.dot(W))
                pure_complete = one_mat.dot(Upis[-1].dot(one_mat.transpose()))
                diff = (np.real(pure_complete[0, 0]) - targ_func(cheby_zeros[i]))
                try:
                    prod_list_left = reduce(np.dot, [W.dot(Upis[a]) for a in range(1, j)])
                except TypeError:
                    prod_list_left = np.eye(2)
                mid = W.dot(Thetas[j, :, :])
                try:
                    prod_list_right = reduce(np.dot, [W.dot(Upis[a]) for a in range(j+1, len(params)-1)])
                except TypeError:
                    prod_list_right = np.eye(2)
                two_mat = prod_list_left.dot(mid.dot(prod_list_right))
                two_mat = start.dot(two_mat.dot(W))
                deriv = (two_mat.dot(Upis[-1].dot(one_mat.transpose())) +
                         one_mat.dot(Upis[-1].dot(two_mat.transpose())))
                val += diff * np.real(deriv)[0, 0]
            val *= (2 / djtil)
            want[j] = val
        val = 0
        for i in range(djtil):
            W = np.array([[cheby_zeros[i], 1j * np.sqrt(1 - cheby_zeros[i]**2)],
                          [1j * np.sqrt(1 - cheby_zeros[i]**2), cheby_zeros[i]]])
            prod_list = [W.dot(Upis[a]) for a in range(1, len(params)-1)]
            one_mat = reduce(np.dot, prod_list)
            one_mat = start.dot(one_mat.dot(W))
            pure_complete = one_mat.dot(Upis[-1].dot(one_mat.transpose()))
            diff = (np.real(pure_complete[0, 0]) - targ_func(cheby_zeros[i]))
            prod_list_left = reduce(np.dot, [W.dot(Upis[a]) for a in range(1, djtil-1)])
            two_mat = start.dot(prod_list_left.dot(W))
            deriv = two_mat.dot(Thetas[-1, :, :].dot(one_mat.transpose()))
            val += diff * np.real(deriv)[0, 0]
        val *= (2 / djtil)
        want[-1] = val
    return want


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


class Uproj(qis.QuantumCircuit):
    def __init__(self, angle, nsys, dim):
        """
        The quantum circuit for the phase part of qsvt.

        Parameters
        ----------
        angle : The angle of the phase
        nsys  : the number of qubits in the 'system'
        dim  : the spacetime dimensions of the lattice

        """
        super().__init__()
        s = 2*dim+1
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        if dim == 1:
            xreg = qis.QuantumRegister(nsys, name='x')
            # regs = [xreg]
            self.add_register(xreg)
        elif dim == 2:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            # regs = [xreg, yreg]
            self.add_register(xreg)
            self.add_register(yreg)
        elif dim == 3:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            # regs = [xreg, yreg, zreg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
        elif dim == 4:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            treg = qis.QuantumRegister(nsys, name='t')
            # regs = [xreg, yreg, zreg, treg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
            self.add_register(treg)
        else:
            raise ValueError("Dimension must be between 1 and 4")
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        # self.cx(anc, anc2, ctrl_state='0')
        # self.rz(2*angle, anc2)
        # self.cx(anc, anc2, ctrl_state='0')
        self.compose(XGate().control(num_ctrl_qubits=(len(block)+1), ctrl_state='0'*(len(block)+1)), inplace=True, qubits=(*block, *anc, *anc2))
        self.rz(2*angle, anc2)
        self.compose(XGate().control(num_ctrl_qubits=(len(block)+1), ctrl_state='0'*(len(block)+1)), inplace=True, qubits=(*block, *anc, *anc2))


class QuantumSignalProcess(qis.QuantumCircuit):
    def __init__(self, UA, phis, nsys, dim):
        """
        The full quantum circuit for qsvt.

        Parameters
        ----------
        UA   : the block-encoding circuit
        phis : the phases used in the rotations
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimension

        """
        super().__init__()
        s = 2*dim+1
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        if dim == 1:
            xreg = qis.QuantumRegister(nsys, name='x')
            # regs = [xreg]
            self.add_register(xreg)
        elif dim == 2:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            # regs = [xreg, yreg]
            self.add_register(xreg)
            self.add_register(yreg)
        elif dim == 3:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            # regs = [xreg, yreg, zreg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
        elif dim == 4:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            treg = qis.QuantumRegister(nsys, name='t')
            # regs = [xreg, yreg, zreg, treg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
            self.add_register(treg)
        else:
            raise ValueError("Dimension must be between 1 and 4")
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        phis[0] = phis[0] + np.pi/4
        phis[-1] = phis[-1] + np.pi/4
        phis[1:-1] = phis[1:-1] + np.pi/2
        phi_flip = phis[::-1]
        self.h(anc2)
        if len(phi_flip) % 2 == 0:
            for i in range(len(phi_flip)//2-1):
                self.compose(Uproj(phi_flip[2*i], nsys, dim), inplace=True)
                self.compose(UA, inplace=True)
                self.compose(Uproj(phi_flip[2*i+1], nsys, dim), inplace=True)
                self.compose(UA.inverse(), inplace=True)
            self.compose(Uproj(phi_flip[-2], nsys, dim), inplace=True)
            self.compose(UA, inplace=True)
            self.compose(Uproj(phi_flip[-1], nsys, dim), inplace=True)
        else:
            for i in range(len(phi_flip)//2):
                self.compose(Uproj(phi_flip[2*i], nsys, dim), inplace=True)
                self.compose(UA, inplace=True)
                self.compose(Uproj(phi_flip[2*i+1], nsys, dim), inplace=True)
                self.compose(UA.inverse(), inplace=True)
            self.compose(Uproj(phi_flip[-1], nsys, dim), inplace=True)
        self.h(anc2)


class RealPart(qis.QuantumCircuit):
    def __init__(self, UA, phis, nsys, dim):
        """
        The quantum circuit for the real part of the qsvt.

        Parameters
        ----------
        phis : the phases used in the rotations
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimension of the lattice

        """
        super().__init__()
        s = 2*dim+1
        phis_minus = -1*np.asarray(phis)
        phis_minus[0] = phis_minus[0] + np.pi/2
        phis_minus[-1] = phis_minus[-1] - np.pi/2
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
        if dim == 1:
            xreg = qis.QuantumRegister(nsys, name='x')
            regs = [*xreg]
            self.add_register(xreg)
        elif dim == 2:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            regs = [*xreg, *yreg]
            self.add_register(xreg)
            self.add_register(yreg)
        elif dim == 3:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            regs = [*xreg, *yreg, *zreg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
        elif dim == 4:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            treg = qis.QuantumRegister(nsys, name='t')
            regs = [*xreg, *yreg, *zreg, *treg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
            self.add_register(treg)
        else:
            raise ValueError("Dimension must be between 1 and 4")
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        anc3 = qis.QuantumRegister(1, name='anc3')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.add_register(anc3)

        Uphi = QuantumSignalProcess(UA, phis, nsys, dim)
        Uphiminus = QuantumSignalProcess(UA, phis_minus, nsys, dim)
        qus = [*anc3] + regs + [*block, *anc, *anc2]
        self.h(anc3)
        self.compose(Uphi.control(num_ctrl_qubits=1, ctrl_state='0'),
                     inplace=True, qubits=qus)
        self.compose(Uphiminus.control(num_ctrl_qubits=1),
                     inplace=True, qubits=qus)
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


def sym_log(x):
    """ The symmetric part of log(1+x)."""
    return 0.5*(np.log(1+x) + np.log(1-x))


def asym_log(x):
    """ the anti-symmetric part of log(1+x)."""
    return 0.5*(np.log(1+x) - np.log(1-x))


def shift_log(x):
    """ log(1+x)"""
    return np.log(1+x)


# def even_delta(x):
#     if x % 2 == 0:
#         return 1.
#     else:
#         return 0.


# def odd_delta(x):
#     if x % 2 == 1:
#         return 1.
#     else:
#         return 0.

# def log_coeffs(kmax):
#     want = np.array([-np.log(2)] + [-2 * even_delta(n) / n for n in range(1, kmax+1)])
#     return want


# def log_coeffs(kmax):
#     want = [0] + [(-1)**(n+1) * even_delta(n) / n for n in range(1, kmax+1)]
#     return want


# def taylor_eval(x, arr):
#     want = sum([x**n * arr[n] for n in range(len(arr))])
#     return want


# def log_coeffs(kmax):
#     want = np.array([-np.log(2)] + [-2 * (-1.)**n / n for n in range(1, kmax+1)])
#     return want


# def approx_coeffs(func, N):
#     zeros = np.array([np.pi * (k+0.5) / N for k in range(N)])
#     want = [((2 - int(n == 0)) / N) * sum([mt.cos(n * zeros[k])
#                                            * func(mt.cos(zeros[k]))
#                                            for k in range(N)]) for n in range(N)]
#     return want

if __name__ == "__main__":
    ua = BlockEncodeFreeScalar(2, 2)
    test = QuantumSignalProcess(ua, range(2), 2, 2)
    print(test)
