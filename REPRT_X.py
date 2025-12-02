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
gvot = 50
n_t = 4
N_X = n_t * gvot
nn = 10
ans = []
T0 = 100
T = 100
N = 2
PMS1 = 0.005
PMS2 = 0.04
PMS0 = 0.04
#PMS0 = 1 / 50
rrange = range(gvot, N_X, gvot)
#n_t = 4

zZ = np.array([[1,0]]).T
eE = np.array([[0,1]]).T

A = [zZ, eE]

B = []
for i1 in range(2):
    for i2 in range(2):
        B.append(np.kron(A[i1], A[i2]))

def partial_trace(rho_ab):
    tr = np.eye(2) - np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(4):
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


for n_x in rrange:
    #p = 1 - np.exp(-t/T0)

    summ = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf2 = alf2 + np.pi / N
            #alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
            #alf2 = random.randint(0, 1000) / 1000 * 2 * np.pi
            sch+=1
            sim = cirq.Simulator()
            circuit1 = cirq.Circuit()
            qutrits1 = []
            for j in range(3):
                qutrits1.append(cirq.LineQid(j, dimension=2))

            povorot = R(alf1, alf2, 0, 1)
            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

            q0 = qutrits1[0]
            q1 = qutrits1[1]
            q2 = qutrits1[2]

            for __ in np.arange(n_x // gvot):


                circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)

                for _ in range(5):
                    circuit1.append([cirq.bit_flip(PMS1).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS1).on(q1)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q1)], strategy=InsertStrategy.INLINE)

                circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)

                for _ in range(5):
                    circuit1.append([cirq.bit_flip(PMS1).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS1).on(q2)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q2)], strategy=InsertStrategy.INLINE)

                #circuit1.append([cirq.H(q0)], strategy=InsertStrategy.INLINE)
                #circuit1.append([cirq.H(q1)], strategy=InsertStrategy.INLINE)
                #circuit1.append([cirq.H(q2)], strategy=InsertStrategy.INLINE)
                for _ in range(gvot):

                    circuit1.append([cirq.X.on(q0)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.bit_flip(PMS0).on(q0)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.X.on(q1)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.bit_flip(PMS0).on(q1)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.X.on(q2)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.bit_flip(PMS0).on(q2)], strategy=InsertStrategy.INLINE)




                if n_x > 10000000000000 * (1 - np.exp(-1)) * T0:
                    circuit1.append([cirq.X(q0)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.X(q1)], strategy=InsertStrategy.INLINE)
                    circuit1.append([cirq.X(q2)], strategy=InsertStrategy.INLINE)

                #circuit1.append([cirq.H(q0)], strategy=InsertStrategy.INLINE)
                #circuit1.append([cirq.H(q1)], strategy=InsertStrategy.INLINE)
                #circuit1.append([cirq.H(q2)], strategy=InsertStrategy.INLINE)

                circuit1.append([cirq.CNOT(q0, q1)], strategy=InsertStrategy.INLINE)

                for _ in range(5):
                    circuit1.append([cirq.bit_flip(PMS1).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS1).on(q1)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q1)], strategy=InsertStrategy.INLINE)


                circuit1.append([cirq.CNOT(q0, q2)], strategy=InsertStrategy.INLINE)

                for _ in range(5):
                    circuit1.append([cirq.bit_flip(PMS1).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS1).on(q2)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q2)], strategy=InsertStrategy.INLINE)


                circuit1.append([cirq.CCNOT(q1,q2,q0)])

                for _ in range(4):
                    circuit1.append([cirq.bit_flip(PMS1).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(5):
                    circuit1.append([cirq.bit_flip(PMS1).on(q2)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS1).on(q1)], strategy=InsertStrategy.INLINE)
                for _ in range(2):
                    circuit1.append([cirq.bit_flip(PMS2).on(q0)], strategy=InsertStrategy.INLINE)
                for _ in range(3):
                    circuit1.append([cirq.bit_flip(PMS2).on(q2)], strategy=InsertStrategy.INLINE)
                for _ in range(1):
                    circuit1.append([cirq.bit_flip(PMS2).on(q1)], strategy=InsertStrategy.INLINE)

            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            ro_ab = cirq.final_density_matrix(circuit1, qubit_order = [q0,q1,q2])

            fid = partial_trace(ro_ab)[0][0]
            summ += fid
    ans.append(summ / sch)

ans1 = []
for n_x in rrange:
    #p = 1 - np.exp(-t/T0)

    summ = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf2 = alf2 + np.pi / N
            sim = cirq.Simulator()
            circuit1 = cirq.Circuit()
            qutrits1 = []
            #alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
            #alf2 = random.randint(0, 1000) / 1000 * 2 * np.pi
            sch += 1
            for j in range(1):
                qutrits1.append(cirq.LineQid(j, dimension=2))

            povorot = R(alf1, alf2, 0, 1)
            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
            q0 = qutrits1[0]
            for _ in range(n_x):
                circuit1.append([cirq.X.on(q0)], strategy=InsertStrategy.INLINE)
                circuit1.append([cirq.bit_flip(PMS0).on(q0)], strategy=InsertStrategy.INLINE)
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
    ax.scatter(rrange,1- np.array(s), color='b', s=10, label='без коррекции')
    ax.scatter(rrange,1- np.array(c), color='r', s=5, label='c коррекции')
    #ax.plot(t, line, color='g', label='теор')
    #ax.plot(t, line2, color='black', label='теор2')
    print(c)
    print(s)
    print(t)
    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('error_prob')
    plt.title('P1 = 0.999, P2 = 0.99, bit-flip')
    plt.legend()
    plt.grid()

    plt.show()

graph(ans,ans1,np.linspace(0, T, nn))


