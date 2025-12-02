import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
nn = 100
ans = []
T0 = 100
T = 200
N = 2
PMS1 = 0.00
PMS2 = 0.0
PMS0 = (1 - np.exp(-1)) / 25
#PMS0 = 1 / 50

n_t = 2

zZ = np.array([[1,0]]).T
eE = np.array([[0,1]]).T

A = [zZ, eE]

B = []

def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                for i5 in range(2):
                    B.append(m(A[i1], A[i2], A[i3], A[i4], A[i5]))

def partial_trace(rho_ab):
    tr = np.eye(2) - np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2**5):
                tr = tr + np.kron(A[i].T, B[k].T) @ rho_ab @ np.kron(A[j], B[k]) * A[i] @ A[j].T
    return tr

def R(fi, hi, i=0, j=1):
    I = np.array([[1, 0], [0, 1]])
    x01_for_ms = np.array([[0, 1],
                           [1, 0]])
    y01_for_ms = np.array([[0, complex(0, -1)],
                           [complex(0, 1), 0]])

    if (i, j) == (0, 1):
        x_for_ms = x01_for_ms
        y_for_ms = y01_for_ms
    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(complex(0, -1) * m * hi / 2)


class U(cirq.Gate):
    def __init__(self, mat, diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i


    def _qid_shape_(self):
        return (2,)

    def _unitary_(self):
        return self.mat

    def _circuit_diagram_info_(self, args):
        return self.diag_info

ans = []
for t in np.linspace(0, T, nn):
    p = 1 - np.exp(-t/T0)
    alf1 = 0
    alf2 = 0
    sim = cirq.Simulator()
    circuit1 = cirq.Circuit()
    qutrits1 = []
    for j in range(6):
        qutrits1.append(cirq.LineQid(j, dimension=2))

    povorot = R(alf1, alf2, 0, 1)
    pg = U(povorot)
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q2)], strategy=InsertStrategy.INLINE)


    circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q2)], strategy=InsertStrategy.INLINE)
    #circuit1.append([cirq.CCNOT(q1, q2, q0)])
    #circuit1.append([cirq.CCNOT(q2, q0, q1)])
    #circuit1.append([cirq.CCNOT(q0, q1, q2)])
    #circuit1.append([cirq.CCNOT(q1, q2, q0)])
    #circuit1.append([cirq.CCNOT(q2, q0, q1)])
    #circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)



    q0 = qutrits1[3]
    q1 = qutrits1[4]
    q2 = qutrits1[5]

    circuit1.append([cirq.H.on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q2)], strategy=InsertStrategy.INLINE)

    circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.phase_flip(p).on(q2)], strategy=InsertStrategy.INLINE)
    # circuit1.append([cirq.CCNOT(q1, q2, q0)])
    # circuit1.append([cirq.CCNOT(q2, q0, q1)])


    circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CCNOT(q1, q2, q0)])
    circuit1.append([cirq.H.on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q2)], strategy=InsertStrategy.INLINE)

    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q0_ = qutrits1[3]
    q1_ = qutrits1[4]
    q2_ = qutrits1[5]
    circuit1.append([cirq.CNOT(q0_, q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0_, q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0_, q2)], strategy=InsertStrategy.INLINE)

    circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.CCNOT(q1, q2, q0)])

    circuit1.append([cirq.H.on(q0)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q1)], strategy=InsertStrategy.INLINE)
    circuit1.append([cirq.H.on(q2)], strategy=InsertStrategy.INLINE)
    povorot_r = R(alf1, -alf2, 0, 1)
    pg_r = U(povorot_r)
    circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
    ro_ab = cirq.final_density_matrix(circuit1, qubit_order=[q0, q1, q2, q0_, q1_, q2_])
    fid = partial_trace(ro_ab)[0][0]
    ans.append(fid)


print(circuit1)




ans1 = []
for t in np.linspace(0, T, nn):
    p = 1 - np.exp(-t/T0)

    summ = 0
    sch = 0
    alf1 = 0
    alf2 = 0
    sim = cirq.Simulator()
    circuit1 = cirq.Circuit()
    qutrits1 = []
    for j in range(1):
        qutrits1.append(cirq.LineQid(j, dimension=2))

    povorot = R(alf1, alf2, 0, 1)
    pg = U(povorot)
    circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

    q0 = qutrits1[0]

    circuit1.append([cirq.bit_flip(p).on(q0)], strategy=InsertStrategy.INLINE)
    povorot_r = R(alf1, -alf2, 0, 1)
    pg_r = U(povorot_r)
    circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
    ro_ab = cirq.final_density_matrix(circuit1)

    fid = ro_ab[0][0]
    ans1.append(fid)



def graph(c,s,t):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.scatter(t, s, color='b', s=5, label='без коррекции')
    ax.scatter(t, c, color='r', s=5, label='c коррекции')
    #ax.plot(t, line, color='g', label='теор')
    #ax.plot(t, line2, color='black', label='теор2')
    print(c)
    print(s)
    print(t)
    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('fidelity')
    #plt.title('P1 = 0.999, P2 = 0.99, bit-flip')
    plt.legend()
    plt.grid()

    plt.show()

graph(ans,ans1,np.linspace(0, T, nn))



