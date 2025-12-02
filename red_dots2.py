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
ans = []
T0 = 25
T = 25
N = 4

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


for t in range(0, T):
    p = 1 - np.exp(-t/T0)
    sim = cirq.Simulator()
    circuit1 = cirq.Circuit()
    qutrits1 = []
    summ = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf2 = alf2 + np.pi / N

            sch+=1
            for j in range(3):
                qutrits1.append(cirq.LineQid(j, dimension=2))


            q0 = qutrits1[0]
            q1 = qutrits1[1]
            q2 = qutrits1[2]
            q0 = qutrits1[0]
            #circuit1.append([cirq.amplitude_damp(p).on(q0)], strategy=InsertStrategy.INLINE)
            '''
            povorot = R(alf1, alf2, 0, 1)
            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
            '''
            circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
            circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)
            circuit1.append([cirq.bit_flip(p)(q0)], strategy=InsertStrategy.INLINE)
            #circuit1.append([cirq.phase_flip(p).on(q1)], strategy=InsertStrategy.INLINE)
            #circuit1.append([cirq.phase_flip(p).on(q2)], strategy=InsertStrategy.INLINE)

            circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)
            circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)

            #circuit1.append([cirq.CCNOT(q1,q2,q0)])
            '''
            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            '''
            res1 = sim.simulate(circuit1, qubit_order = [q0,q1,q2])
            # print(circuit1)
            print(res1)
            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    ans.append(summ / sch)

ans1 = []
for t in range(0, T):
    p = 1 - np.exp(-t/T0)
    sim = cirq.Simulator()
    circuit1 = cirq.Circuit()
    qutrits1 = []
    summ = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf2 = alf2 + np.pi / N
            alf1 = 0
            alf2 = 0
            sch += 1
            for j in range(1):
                qutrits1.append(cirq.LineQid(j, dimension=2))

            povorot = R(alf1, alf2, 0, 1)
            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

            q0 = qutrits1[0]
            circuit1.append([cirq.amplitude_damp(p).on(q0)], strategy=InsertStrategy.INLINE)
            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            ro_ab = cirq.final_density_matrix(circuit1)

            fid = ro_ab[0][0]
            summ += fid
    ans1.append(summ/sch)



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
    plt.title('P1 = 1, P2 = 1, amplitude damping (two-qutrit error)')
    plt.legend()
    plt.grid()

    plt.show()

graph(ans,ans1,range(0, T))


