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
        self.compose(RYGate(2*np.arccos(-4/9)).control(num_ctrl_qubits=1,
                                                       ctrl_state='0'),
                     inplace=True, qubits=(block[-1], anc))
        self.compose(RYGate(2*np.arccos((4./9)*(8+0.5**2)-3)).control(num_ctrl_qubits=nblock,
                                                                      ctrl_state=("{0:b}".format(s-1))),
                     inplace=True, qubits=(*block, anc))


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
        self.add_register(anc2)
        self.h(block)


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
            regs = [xreg, yreg, zreg, treg]
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
        self.compose(Diffusion(nsys, dim), inplace=True)
        self.compose(FreeScalarOA(nsys, dim), inplace=True)
        self.compose(NearestNeighborOc(nsys, dim), inplace=True)
        self.compose(Diffusion(nsys, dim), inplace=True)
            

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

        
if __name__ == "__main__":
    test = FreeFermionOA(2, 1)
    print(test)
    
