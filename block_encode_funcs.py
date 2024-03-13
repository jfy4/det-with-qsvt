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
        The quantum circuit for the Oc operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

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


class FreeScalarOA(qis.QuantumCircuit):
    def __init__(self, nsys, dim):
        """
        The quantum circuit for the OA operator.

        Parameters
        ----------
        nsys : the number of qubits in the 'system'
        s    : the number of nonzero elements in a row/column

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
        block = qis.QuantumRegister(nblock, name='block')
        anc = qis.QuantumRegister(1, name='anc')
        anc2 = qis.QuantumRegister(1, name='anc2')
        self.add_register(block)
        self.add_register(anc)
        self.add_register(anc2)
        self.compose(RYGate(2*np.arccos(-4/9)).control(num_ctrl_qubits=1,
                                                       ctrl_state='0'),
                     inplace=True, qubits=(block[-1], anc))
        self.compose(RYGate(2*np.arccos((4./9)*(8+0.5**2)-3)).control(num_ctrl_qubits=nblock,
                                                                      ctrl_state=("{0:b}".format(s-1))),
                     inplace=True, qubits=(*block, anc))
        # self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock-1,
        #                                  ctrl_state='01'),
        #              inplace=True, qubits=(*block[1:], anc))
        # self.compose(RYGate(0.3).control(num_ctrl_qubits=nblock-2,
        #                                  ctrl_state='1'),
        #              inplace=True, qubits=(*block[2:], anc))
        # self.compose(RYGate(0.1).control(num_ctrl_qubits=nblock,
        #                                  ctrl_state='100'),
        #              inplace=True, qubits=(*block, anc))
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

if __name__ == "__main__":
    test = FreeScalarOA(2, 2)
    print(test)
    
