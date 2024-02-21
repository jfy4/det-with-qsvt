from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Operator
# from numpy import pi
import numpy as np

# qreg_q = QuantumRegister(3, 'q')
# # creg_c = ClassicalRegister(4, 'c')
# circuit = QuantumCircuit(qreg_q)

# circuit.h(qreg_q[1])
# circuit.cry(2*np.arccos(-0.4), qreg_q[1], qreg_q[2], ctrl_state='0')
# # circuit.cx(qreg_q[1], qreg_q[2])
# circuit.cry(2*np.arccos(0.3), qreg_q[1], qreg_q[2])
# circuit.cx(qreg_q[1], qreg_q[0])
# circuit.h(qreg_q[1])

# print(circuit)

# op = Operator(circuit)
# print(2*op.data[:4, :4])
# print(2*op.data)

qreg_q = QuantumRegister(3, 'q')
# creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q)

circuit.h(qreg_q[1])
circuit.cry(2*np.arccos(-0.4), qreg_q[1], qreg_q[2], ctrl_state='0')
# circuit.cx(qreg_q[1], qreg_q[2])
circuit.cry(2*np.arccos(0.3), qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[0])
circuit.h(qreg_q[1])

print(circuit)

op = Operator(circuit)
print(2*op.data[:4, :4])
print(2*op.data)

