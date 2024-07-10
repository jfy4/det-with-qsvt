#!/home/judah/miniconda3/bin/python

import numpy as np
# import math as mt
# from scipy.optimize import minimize, HessianUpdateStrategy
# from functools import reduce
# from scipy.special import erf
import qiskit as qis
from qiskit_aer import Aer
from scipy.linalg import sqrtm
from scipy.special import binom
from qiskit.circuit.library import MCXGate, RYGate
from qiskit.quantum_info import Operator
from itertools import product
# from numpy.polynomial import chebyshev


# np.random.seed(0)
seed = 0
rng = np.random.Generator(np.random.PCG64DXSM(seed))

one = np.eye(2)
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])


def my_product(*iterables, repeat=1):
    # product('ABCD', 'xy') → Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) → 000 100 010 110 001 101 011 111

    pools = [tuple(pool) for pool in iterables] * repeat

    result = [[]]
    for pool in pools:
        result = [[y]+x for x in result for y in pool]

    for prod in result:
        yield tuple(prod)

        
class NearestNeighborOc(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the Oc operator
        in the case of nearest neighbors.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
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
            regs = [xreg]
            self.add_register(xreg)
        elif dim == 2:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            regs = [xreg, yreg]
            self.add_register(xreg)
            self.add_register(yreg)
        elif dim == 3:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            regs = [xreg, yreg, zreg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
        elif dim == 4:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            treg = qis.QuantumRegister(nsys, name='t')
            regs = [xreg, yreg, zreg, treg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
            self.add_register(treg)
        else:
            raise ValueError("Dimension must be between 1 and 4")
        bin_states = product(['0', '1'], repeat=nblock)
        bin_states = [''.join(x) for x in bin_states]
        # print(bin_states)
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        for reg, bs in zip(regs, bin_states[:(s-1)//2]):
            self.compose(Lshift(nsys).control(num_ctrl_qubits=nblock,
                                              ctrl_state=bs),
                         inplace=True, qubits=(*block, *reg))
        for reg, bs in zip(regs, bin_states[(s-1)//2:]):
            self.compose(Lshift(nsys).inverse().control(num_ctrl_qubits=nblock,
                                                        ctrl_state=bs),
                         inplace=True, qubits=(*block, *reg))


class NtNNOc(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the Oc operator
        in the case of next-to-nearest neighbors.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimensions of the lattice

        """
        super().__init__()
        s = 4*binom(dim, 2) + 2*dim + 1
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
            # ln = 2*nsys
        if dim == 1:
            xreg = qis.QuantumRegister(nsys, name='x')
            regs = [xreg]
            self.add_register(xreg)
        elif dim == 2:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            regs = [xreg, yreg]
            self.add_register(xreg)
            self.add_register(yreg)
        elif dim == 3:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            regs = [xreg, yreg, zreg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
        elif dim == 4:
            xreg = qis.QuantumRegister(nsys, name='x')
            yreg = qis.QuantumRegister(nsys, name='y')
            zreg = qis.QuantumRegister(nsys, name='z')
            treg = qis.QuantumRegister(nsys, name='t')
            regs = [xreg, yreg, zreg, treg]
            self.add_register(xreg)
            self.add_register(yreg)
            self.add_register(zreg)
            self.add_register(treg)
        else:
            raise ValueError("Dimension must be between 1 and 4")
        # bin_states = product(['0', '1'], repeat=nblock)
        # bin_states = [''.join(x) for x in bin_states]
        # print(bin_states)
        block = qis.QuantumRegister(nblock, name='block')
        # lnreg = qis.QuantumRegister(ln, name='ln')
        anc = qis.QuantumRegister(1, name='anc')
        # anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        # self.add_register(lnreg)
        self.add_register(anc)
        # self.add_register(anc2)
        # plus2 = Lshift(nsys).compose(Lshift(nsys))
        # minus2 = Lshift(nsys).inverse().compose(Lshift(nsys).inverse())
        pshift = Lshift(nsys)
        mshift = Lshift(nsys).inverse()

        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0000'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0000'),
                     inplace=True, qubits=(*block, *xreg))
        
        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0001'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0001'),
                     inplace=True, qubits=(*block, *yreg))

        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0010'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0010'),
                     inplace=True, qubits=(*block, *yreg))

        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0011'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0011'),
                     inplace=True, qubits=(*block, *yreg))

        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0100'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0100'),
                     inplace=True, qubits=(*block, *yreg))

        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0101'),
                     inplace=True, qubits=(*block, *yreg))
        self.compose(pshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0101'),
                     inplace=True, qubits=(*block, *yreg))

        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0110'),
                     inplace=True, qubits=(*block, *xreg))
        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0110'),
                     inplace=True, qubits=(*block, *xreg))

        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0111'),
                     inplace=True, qubits=(*block, *yreg))
        self.compose(mshift.control(num_ctrl_qubits=nblock,
                                    ctrl_state='0111'),
                     inplace=True, qubits=(*block, *yreg))


class FreeScalarOA(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the OA operator
        in the case of a free scalar field.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
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
        bin_states = product(['0', '1'], repeat=nblock)
        bin_states = [''.join(x) for x in bin_states]
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        # below I think only works for dim=1, dim=2 and dim=4
        # self.compose(RYGate(2*np.arccos(-4/9)).control(num_ctrl_qubits=1, ctrl_state='0'), inplace=True, qubits=(block[-1], anc))
        self.compose(RYGate(2*np.arccos(-0.1)).control(num_ctrl_qubits=1, ctrl_state='0'), inplace=True, qubits=(block[-1], anc))
        # self.compose(RYGate(2*np.arccos(0.8)).control(num_ctrl_qubits=nblock, ctrl_state=("{0:b}".format(s-1))), inplace=True, qubits=(*block, anc))
        self.compose(RYGate(2*np.arccos(0.8)).control(num_ctrl_qubits=1, ctrl_state="1"), inplace=True, qubits=(block[-1], anc))
        # self.compose(RYGate(2*np.arccos((4 + bare_mass**2)-3)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], anc))


class FreeFermionOA(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the OA operator
        for the case of free staggered fermions.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
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
        bin_states = product(['0', '1'], repeat=nblock)
        bin_states = [''.join(x) for x in bin_states]
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        # below I think only works for dim=1, dim=2 and dim=4
        self.compose(RYGate(2*np.arccos(-4/9)).control(num_ctrl_qubits=2,
                                                       ctrl_state='00'),
                     inplace=True, qubits=(*block[-2:], anc))
        self.compose(RYGate(2*np.arccos(-4/9)).control(num_ctrl_qubits=2,
                                                       ctrl_state='01'),
                     inplace=True, qubits=(*block[-2:], anc))
        self.compose(RYGate(2*np.arccos((4./9)*(8+0.5**2)-3)).control(num_ctrl_qubits=nblock,
                                                                      ctrl_state=("{0:b}".format(s-1))),
                     inplace=True, qubits=(*block, anc))
        if dim == 2:
            stag_phase = StaggeredPhase(nsys, 1)
            self.compose(stag_phase.control(num_ctrl_qubits=3,
                                            ctrl_state='001'),
                         inplace=True, qubits=(block[0], block[-1], *anc, *xreg))
        elif dim == 4:
            stag_phase = StaggeredPhase(nsys, 1)
            self.compose(stag_phase.control(num_ctrl_qubits=4,
                                            ctrl_state='0001'),
                         inplace=True, qubits=(block[0], block[1], block[-1], *anc, *xreg))
            stag_phase = StaggeredPhase(nsys, 2)
            self.compose(stag_phase.control(num_ctrl_qubits=4,
                                            ctrl_state='0010'),
                         inplace=True, qubits=(block[0], block[1], block[-1], *anc, *xreg, *yreg))
            stag_phase = StaggeredPhase(nsys, 3)
            self.compose(stag_phase.control(num_ctrl_qubits=4,
                                            ctrl_state='0011'),
                         inplace=True, qubits=(block[0], block[1], block[-1], *anc, *xreg, *yreg, *zreg))
        elif dim == 1:
            pass
        else:
            raise ValueError("dimesion must be 2 or 4")


class GaugeU1OA(qis.QuantumCircuit):
    def __init__(self, nsys, dim, m0, K):
        """
        The quantum circuit for the OA operator
        for the case of free staggered fermions.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimensions of the lattice
        m0   : the bare fermion mass
        K    : the hopping coupling
        norm : the overall normalization of the W matrix

        """
        super().__init__()
        s = 4*binom(dim, 2) + 2*dim + 1
        lambda_lb = m0**2
        lambda_ub = m0**2 + 16 * K**2
        norm = lambda_lb - lambda_ub
        # V = 2**(2*nsys)
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
            ln = 2*nsys
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
        ntnn_states = product(['0', '1'], repeat=nblock)
        ntnn_states = [''.join(x) for x in ntnn_states]
        vol_states = product(['0', '1'], repeat=ln)
        vol_states = [''.join(x) for x in vol_states]
        block = qis.QuantumRegister(nblock, name='block')
        # lnreg = qis.QuantumRegister(ln, name='ln')
        anc = qis.QuantumRegister(1, name='anc')
        # anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        # self.add_register(lnreg)
        self.add_register(anc)
        # self.add_register(anc2)
        # load in gauge config file
        # gf = np.exp(1j * np.random.random(size=(4,4,2))*2*np.pi)
        gf = np.exp(1j * np.array([i*2*np.pi / (2*4*4) for i in range(2*4*4)])).reshape((2,4,4))
        gf = gf.transpose((1,2,0))
        eta = np.ones(shape=(4,4,2))
        for x, y in product(range(4), repeat=2):
            eta[x][y][1] = (-1)**(x)
        # gf has shape (V, 2)
        for N in zip(product(range(4), range(4)), vol_states):
            p, bs = N[0], N[1]
            y, x = p[0], p[1]
            # print(x,y)
            # print(bs)
            alpha = -(K**2 / 4) * np.conjugate(gf[x][y][0] * gf[(x+1)%4][y][0]) * 16 / norm
            # print(alpha)
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            # print(matrix[0,0])
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0000'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

            alpha = -(K**2 / 4) * (eta[x][y][0] * eta[(x+1)%4][(y+1)%4][1] * np.conjugate(gf[x][y][0] * gf[(x+1)%4][y][1]) +
                                   eta[x][y][1] * eta[(x+1)%4][(y+1)%4][0] * np.conjugate(gf[x][y][1] * gf[x][(y+1)%4][0])) * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0001'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

            alpha = (K**2 / 4) * (eta[x][y][1] * eta[(x+1)%4][y-1][0] * gf[x][y-1][1] * np.conjugate(gf[x][y-1][0]) +
                                  eta[x][y][0] * eta[(x+1)%4][y-1][1] * np.conjugate(gf[x][y][0]) * gf[(x+1)%4][y-1][1]) * 16 / norm
            # print(gf[x][y-1][1] * np.conjugate(gf[x][y-1][0]), np.conjugate(gf[x][y][0]) * gf[(x+1)%4][y-1][1])
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            # print(matrix)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            # print(Operator(qc).data)
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0010'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))
            # break
            alpha = (K**2 / 4) * (eta[x][y][0] * eta[x-1][(y+1)%4][1] * gf[x-1][y][0] * np.conjugate(gf[x-1][y][1]) +
                                  eta[x][y][1] * eta[x-1][(y+1)%4][0] * np.conjugate(gf[x][y][1]) * gf[x-1][(y+1)%4][0]) * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0011'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

            alpha = -(K**2 / 4) * (eta[x][y][0] * eta[x-1][y-1][1] * gf[x-1][y][0] * gf[x-1][y-1][1] +
                                   eta[x][y][1] * eta[x-1][y-1][0] * gf[x][y-1][1] * gf[x-1][y-1][0]) * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0100'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))
            
            alpha = -(K**2 / 4) * np.conjugate(gf[x][y][1] * gf[x][(y+1)%4][1]) * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0101'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

            alpha = -(K**2 / 4) * gf[x-1][y][0] * gf[x-2][y][0] * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0110'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

            alpha = -(K**2 / 4) * gf[x][y-1][1] * gf[x][y-2][1] * 16 / norm
            matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
            qc = qis.QuantumCircuit(1)
            qc.unitary(matrix, [0])
            self.compose(qc.control(num_ctrl_qubits=nblock+ln,
                                    ctrl_state=bs + '0111'),
                         inplace=True, qubits=(*block, *xreg, *yreg, *anc))

        # alpha = ((m0**2 + K**2) - lambda_ub) / norm
        # matrix = (np.real(alpha) * one) + (1j*np.imag(alpha) * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
        # qc = qis.QuantumCircuit(1)
        # qc.unitary(matrix, [0])
        # self.compose(qc.control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))

        # self.compose(RYGate(2*np.arccos(16 * (m0**2 + 2 * K**2) / norm - 7)).control(num_ctrl_qubits=nblock, ctrl_state='1000'), inplace=True, qubits=(*block, *anc))
        # self.compose(RYGate(2*np.arccos(2 * ((m0**2 + K**2) - lambda_ub) / norm)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))
        self.compose(RYGate(2*np.arccos(2 * (m0**2 + K**2) / norm)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))
        # self.compose(RYGate(2*np.arccos(((m0**2 + K**2) - lambda_ub) / (8*norm))).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))
        # self.compose(RYGate(2*np.arccos(0.8)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))


# class GaugeU1OA(qis.QuantumCircuit):
#     def __init__(self, nsys, dim, m0, K, norm):
#         """
#         The quantum circuit for the OA operator
#         for the case of free staggered fermions.

#         Parameters
#         ----------
#         nsys : the number of qubits in the 'system'
#         dim  : the spacetime dimensions of the lattice

#         """
#         super().__init__()
#         s = 4*binom(dim, 2) + 2*dim + 1
#         # V = 2**(2*nsys)
#         if s == 1:
#             nblock = 1
#         else:
#             nblock = int(np.ceil(np.log2(s)))
#             ln = 2*nsys
#         if dim == 1:
#             xreg = qis.QuantumRegister(nsys, name='x')
#             # regs = [xreg]
#             self.add_register(xreg)
#         elif dim == 2:
#             xreg = qis.QuantumRegister(nsys, name='x')
#             yreg = qis.QuantumRegister(nsys, name='y')
#             # regs = [xreg, yreg]
#             self.add_register(xreg)
#             self.add_register(yreg)
#         elif dim == 3:
#             xreg = qis.QuantumRegister(nsys, name='x')
#             yreg = qis.QuantumRegister(nsys, name='y')
#             zreg = qis.QuantumRegister(nsys, name='z')
#             # regs = [xreg, yreg, zreg]
#             self.add_register(xreg)
#             self.add_register(yreg)
#             self.add_register(zreg)
#         elif dim == 4:
#             xreg = qis.QuantumRegister(nsys, name='x')
#             yreg = qis.QuantumRegister(nsys, name='y')
#             zreg = qis.QuantumRegister(nsys, name='z')
#             treg = qis.QuantumRegister(nsys, name='t')
#             # regs = [xreg, yreg, zreg, treg]
#             self.add_register(xreg)
#             self.add_register(yreg)
#             self.add_register(zreg)
#             self.add_register(treg)
#         else:
#             raise ValueError("Dimension must be between 1 and 4")
#         ntnn_states = product(['0', '1'], repeat=nblock)
#         ntnn_states = [''.join(x) for x in ntnn_states]
#         vol_states = product(['0', '1'], repeat=ln)
#         vol_states = [''.join(x) for x in vol_states]
#         block = qis.QuantumRegister(nblock, name='block')
#         lnreg = qis.QuantumRegister(ln, name='ln')
#         anc = qis.QuantumRegister(1, name='anc')
#         anc2 = qis.QuantumRegister(1, name='anc2')
#         self.add_register(block)
#         self.add_register(lnreg)
#         self.add_register(anc)
#         # self.add_register(anc2)
#         # load in gauge config file
#         gf = np.exp(1j * np.random.random(size=(4,4,2))*2*np.pi)
#         eta = np.ones(shape=(4,4,2))
#         for x, y in product(range(4), repeat=2):
#             eta[x][y][1] = (-1)**(x)
#         # gf has shape (V, 2)
#         for N in zip(product(range(4), range(4)), vol_states):
#             p, bs = N[0], N[1]
#             x, y = p[0], p[1]
#             alpha = -(K**2 / 4) * np.conjugate(gf[x][y][0] * gf[(x+1)%4][y][0])
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             # print(matrix)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0000'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = -(K**2 / 4) * (eta[x][y][0] * eta[(x+1)%4][(y+1)%4][1] * np.conjugate(gf[x][y][0] * gf[(x+1)%4][y][1]) +
#                                    eta[x][y][1] * eta[(x+1)%4][(y+1)%4][0] * np.conjugate(gf[x][y][1] * gf[x][(y+1)%4][0])) / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0001'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = (K**2 / 4) * (eta[x][y][1] * eta[(x+1)%4][y-1][0] * gf[x][y-1][1] * np.conjugate(gf[x][y-1][0]) +
#                                    eta[x][y][0] * eta[(x+1)%4][y-1][1] * np.conjugate(gf[x][y][0]) * gf[(x+1)%4][y-1][1]) / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0010'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = (K**2 / 4) * (eta[x][y][0] * eta[x-1][(y+1)%4][1] * gf[x-1][y][0] * np.conjugate(gf[x-1][y][1]) +
#                                    eta[x][y][1] * eta[x-1][(y+1)%4][0] * np.conjugate(gf[x][y][1]) * gf[x-1][(y+1)%4][0]) / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0011'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = -(K**2 / 4) * (eta[x][y][0] * eta[x-1][y-1][1] * gf[x-1][y][0] * gf[x-1][y-1][1] +
#                                    eta[x][y][1] * eta[x-1][y-1][0] * gf[x][y-1][1] * gf[x-1][y-1][0]) / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0100'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))
            
#             alpha = -(K**2 / 4) * np.conjugate(gf[x][y][1] * gf[x][(y+1)%4][1]) / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0101'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = -(K**2 / 4) * gf[x-1][y][0] * gf[x-2][y][0] / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0110'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))

#             alpha = -(K**2 / 4) * gf[x][y-1][1] * gf[x][y-2][1] / norm
#             matrix = ((alpha + np.conjugate(alpha))/2 * one) + ((alpha - np.conjugate(alpha))/2 * Z) - 1j*Y*np.sqrt(1-np.abs(alpha)**2)
#             qc = qis.QuantumCircuit(1)
#             qc.unitary(matrix, [0])
#             self.compose(qc.control(num_ctrl_qubits=nblock+ln,
#                                      ctrl_state=bs + '0111'),
#                          inplace=True, qubits=(*block, *lnreg, *anc))
#         # self.compose(RYGate(2*np.arccos(16 * (m0**2 + 2 * K**2) / norm - 7)).control(num_ctrl_qubits=nblock, ctrl_state='1000'), inplace=True, qubits=(*block, *anc))
#         # self.compose(RYGate(2*np.arccos(2 * (m0**2 + 2 * K**2) / norm)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))
#         self.compose(RYGate(2*np.arccos(0.8)).control(num_ctrl_qubits=1, ctrl_state='1'), inplace=True, qubits=(block[-1], *anc))


class Diffusion(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the Diffusion operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
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
        # self.add_register(anc2)
        self.h(block)


class BigDiffusion(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the Diffusion operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimensions of the lattice

        """
        super().__init__()
        s = 4*binom(dim, 2) + 2*dim + 1
        # V = 2**(2*nsys)
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
            ln = 2*nsys
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
        # lnreg = qis.QuantumRegister(ln, name='ln')
        anc = qis.QuantumRegister(1, name='anc')
        # anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        # self.add_register(lnreg)
        self.add_register(anc)
        # self.add_register(anc2)
        # block = qis.QuantumRegister(nblock, name='block')
        # anc = qis.QuantumRegister(1, name='anc')
        # anc2 = qis.QuantumRegister(1, name='anc2')
        # self.add_register(block)
        # self.add_register(anc)
        # self.add_register(anc2)
        self.h(block)
        # self.h(lnreg)
        

class BlockEncodeFreeScalar(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The full quantum circuit for block-encoding
        the free scalar laplacian.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
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
        # self.add_register(anc2)
        self.compose(Diffusion(nsys, dim), inplace=True)
        self.compose(FreeScalarOA(nsys, dim), inplace=True)
        self.compose(NearestNeighborOc(nsys, dim), inplace=True)
        self.compose(Diffusion(nsys, dim), inplace=True)


class BlockEncodeU1(qis.QuantumCircuit):
    def __init__(self, nsys, dim, m0, K):
        """
        The full quantum circuit for block-encoding
        the free scalar laplacian.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        dim  : the spacetime dimensions of the lattice

        """
        super().__init__()
        s = 4*binom(dim, 2) + 2*dim + 1
        # V = 2**(2*nsys)
        if s == 1:
            nblock = 1
        else:
            nblock = int(np.ceil(np.log2(s)))
            ln = 2*nsys
        # s = 2*dim+1
        # if s == 1:
        #     nblock = 1
        # else:
        #     nblock = int(np.ceil(np.log2(s)))
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
        # lnreg = qis.QuantumRegister(ln, name='ln')
        anc = qis.QuantumRegister(1, name='anc')
        # anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        # self.add_register(lnreg)
        self.add_register(anc)
        # self.add_register(anc2)
        self.compose(BigDiffusion(nsys, dim), inplace=True)
        # self.compose(Diffusion(nsys, dim), inplace=True)
        self.compose(GaugeU1OA(nsys, dim, m0, K), inplace=True)
        self.compose(NtNNOc(nsys, dim), inplace=True)
        # self.compose(Diffusion(nsys, dim), inplace=True)
        self.compose(BigDiffusion(nsys, dim), inplace=True)


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
            self.compose(MCXGate(num_ctrl_qubits=nsys-i), inplace=True)
        self.x(sys[0])


class StaggeredPhase(qis.QuantumCircuit):
    def __init__(self, nsys, mu):
        """
        Computes the staggered phase for a spacetime point.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        mu   : the spactime direction under consideration

        """
        super().__init__()
        if 1 <= mu:
            xreg = qis.QuantumRegister(nsys, name='x')
            self.add_register(xreg)
            for n, q in enumerate(xreg):
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.rz(np.pi * 2**n, q)
        if 2 <= mu:
            yreg = qis.QuantumRegister(nsys, name='y')
            self.add_register(yreg)
            for n, q in enumerate(yreg):
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.rz(np.pi * 2**n, q)
        if 3 <= mu:
            zreg = qis.QuantumRegister(nsys, name='z')
            self.add_register(zreg)
            for n, q in enumerate(zreg):
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.p(2**n * np.pi / 2, q)
                self.x(q)
                self.rz(np.pi * 2**n, q)


def make_ferm_w(m0, K, norm, L):
    # gf = np.exp(1j * np.random.random(size=(4,4,2))*2*np.pi)
    gf = np.exp(1j * np.array([i*2*np.pi / (2*L*L) for i in range(2*L*L)])).reshape((2,L,L))
    gf = gf.transpose((1,2,0))
    # print(gf[:,:,0])
    eta = np.ones(shape=(L,L,2))
    fmatrix = np.zeros((L**2, L**2), dtype=np.complex128)
    for x, y in product(range(L), repeat=2):
        eta[x][y][1] = (-1)**(x)
    
    for x, y in product(range(L), repeat=2):
        x1 = x
        y1 = y
        idx1 = L*x1+y1
        # print(x1, y1)
        x2 = (x-1)%L
        y2 = (y+1)%L
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] = (K**2 / 4) * (eta[x][y][0] * eta[x-1][(y+1)%L][1] * gf[x-1][y][0] * np.conjugate(gf[x-1][y][1]) + eta[x][y][1] * eta[x-1][(y+1)%L][0] * np.conjugate(gf[x][y][1]) * gf[x-1][(y+1)%L][0]) / norm
        x2 = (x-1)%L
        y2 = (y-1)%L
        # print(x2, y2)
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] = -(K**2 / 4) * (eta[x][y][0] * eta[x-1][y-1][1] * gf[x-1][y][0] * gf[x-1][y-1][1] +
                                   eta[x][y][1] * eta[x-1][y-1][0] * gf[x][y-1][1] * gf[x-1][y-1][0]) / norm
        x2 = (x+1)%L
        y2 = (y+1)%L
        # print(x2, y2)
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] = -(K**2 / 4) * (eta[x][y][0] * eta[(x+1)%L][(y+1)%L][1] * np.conjugate(gf[x][y][0] * gf[(x+1)%L][y][1]) +
                                   eta[x][y][1] * eta[(x+1)%L][(y+1)%L][0] * np.conjugate(gf[x][y][1] * gf[x][(y+1)%L][0])) / norm
        x2 = (x+1)%L
        y2 = (y-1)%L
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] = (K**2 / 4) * (eta[x][y][1] * eta[(x+1)%L][y-1][0] * gf[x][y-1][1] * np.conjugate(gf[x][y-1][0]) +
                                   eta[x][y][0] * eta[(x+1)%4][y-1][1] * np.conjugate(gf[x][y][0]) * gf[(x+1)%L][y-1][1]) / norm
        # print(gf[x][y-1][1] * np.conjugate(gf[x][y-1][0]), np.conjugate(gf[x][y][0]) * gf[(x+1)%L][y-1][1])
        # break
        x2 = (x+2)%L
        y2 = y
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] += -(K**2 / 4) * np.conjugate(gf[x][y][0] * gf[(x+1)%L][y][0]) / norm
        # print(fmatrix[idx1, idx2])
        # # fmatrix[idx1, idx2] = 1
        x2 = (x-2)%L
        y2 = y
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] += -(K**2 / 4) * gf[x-1][y][0] * gf[x-2][y][0] / norm
        # # fmatrix[idx1, idx2] = 1
        x2 = x
        y2 = (y+2)%L
        # print(x2, y2)
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] += -(K**2 / 4) * np.conjugate(gf[x][y][1] * gf[x][(y+1)%L][1]) / norm
        # # fmatrix[idx1, idx2] = 1
        x2 = x
        y2 = (y-2)%L
        # print(x2, y2)
        idx2 = L*x2+y2
        fmatrix[idx1, idx2] += -(K**2 / 4) * gf[x][y-1][1] * gf[x][y-2][1] / norm
        # fmatrix[idx1, idx2] = 1
        fmatrix[idx1, idx1] = (m0**2 + K**2) / norm

    return fmatrix


def block_w(matrix):
    mat = np.eye(matrix.shape[0])-matrix.dot(matrix.conjugate().transpose())
    L = sqrtm(mat)
    assert np.allclose(mat, L.dot(L))
    return np.kron(one, matrix) + np.kron(-1j*Y, L)


# class BlockEncodeU1External(qis.QuantumCircuit):
    


if __name__ == "__main__":
    pass
    # test = FreeFermionOA(2, 2)
    
    # print(test)
    # print(np.allclose(arr, arr.transpose().conjugate()))
    # test = NtNNOc(2, 2)
    # test = GaugeU1OA(2, 2, 0.5, 1.5, 32)
    test = BlockEncodeU1(2, 2, 0.5, 1.5, 32)
    # # test = Lshift(2)
    print(test)
    # arr = Operator(test).data
    # print(arr)
    # backend = Aer.get_backend('qasm_simulator')
    # backend_options = {"method": "statevector"}
    # aersim = AerSimulator()

    # results_ideal = qis.execute(test, aersim).result()
    # print(results_ideal)
    # # Circuit execution
    # job = execute(test, backend, backend_options=backend_options)
    # pass
    # qc = qis.transpile(test, basis_gates=['cx', 'u'])
    # arr = Operator(qc).data
    # print(arr[:16, :16])
    # test = BigDiffusion(2, 2)
    # test = Lshift(2)
    # arr = Operator(test).data
    # print(arr[:16, :16])
    # print(test)
    # for i in my_product(range(2), repeat=3):
    #     print(i)
    # test = make_ferm_w(0.5, 1.5, 32, 4)
    # print(test[:,0])

    # # exit()
    # eigvals = np.linalg.eigvals(test)
    # eigvals = np.sort(eigvals)
    # test /= eigvals[-1]
    # print(test)
    # u = block_w(test)
    # print(u[:16, :16].dot(u[:16, :16].conjugate().transpose())
    #       + u[:16, 16:].dot(u[:16, 16:]))
    # print(u[:16, :16])
    # print(u.dot(u.conjugate().transpose()))
    # print(np.allclose(u.dot(u.conjugate().transpose()), np.eye(u.shape[0])))
    # print(test[0,2], test[2,0])
    # print(test)
    # print(np.allclose(test, test.conjugate().transpose()))
    
