
def printm(m):
    import numpy as np
    import cirq
    import networkx as nx
    print('np.array([', end='')
    for line in m:
        print('[', end='')
        for i in range(len(line)-1):
            print(line[i],',', end='')
        print(line[i], end='')
        print('],')
    print('])')

def get_minor(m, i, j):
    X = np.ones(len(m)).astype(bool)
    Y = np.ones(len(m[0])).astype(bool)
    X[i], Y[j] = False, False
    return m[np.ix_(X, Y)]
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random

def m3(g1,g2,g3):
    return(np.kron(g1,np.kron(g2,g3)))

def m2(g1, g2):
    return(np.kron(g1,g2))

def dag(matrix):
    return np.conj(matrix.T)


def R(fi, hi, i=0, j=1):
    N = 3
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms = np.zeros((N, N))
    x_for_ms[i][j] = 1
    x_for_ms[j][i] = 1
    y_for_ms = np.zeros((N, N))
    y_for_ms[i][j] = -1
    y_for_ms[j][i] = 1
    y_for_ms = y_for_ms * 1j

    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(-1j * m * hi / 2)

def make_ms_matrix(N, fi, hi, i, j, k, l):
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms1 = np.zeros((N, N))
    x_for_ms1[i][j] = 1
    x_for_ms1[j][i] = 1
    y_for_ms1 = np.zeros((N, N))
    y_for_ms1[i][j] = -1
    y_for_ms1[j][i] = 1
    y_for_ms1 = 1j * y_for_ms1
    if k == l:
        return
    if k > l:
        k, l = l, k
    x_for_ms2 = np.zeros((N, N))
    x_for_ms2[k][l] = 1
    x_for_ms2[l][k] = 1
    y_for_ms2 = np.zeros((N, N))
    y_for_ms2[k][l] = -1
    y_for_ms2[l][k] = 1
    y_for_ms1 = 1j * y_for_ms1

    m = np.kron((np.cos(fi) * x_for_ms1 + np.sin(fi) * y_for_ms1), (np.cos(fi) * x_for_ms2 + np.sin(fi) * y_for_ms2))
    m = -1j * m * hi
    return linalg.expm(m)
I = np.eye(3)

u1 = np.kron(R(0, -np.pi, 1, 2), I)
u2 = np.kron(R(np.pi / 2, np.pi / 2, 0, 1), I)
u3 = np.kron(R(0, -np.pi, 0, 1), R(0, -np.pi, 0, 1))
u4 = np.kron(R(np.pi / 2, -np.pi / 2, 0, 1), I)
u5 = np.kron(R(0, np.pi, 1, 2), I)
u1r = np.kron(I,R(0, -np.pi, 1, 2))
u2r = np.kron(I,R(np.pi / 2, np.pi / 2, 0, 1))
u3r = np.kron(R(0, -np.pi, 0, 1), R(0, -np.pi, 0, 1))

u4r = np.kron(I,R(np.pi / 2, -np.pi / 2, 0, 1))
u5r = np.kron(I,R(0, np.pi, 1, 2))
xx01 = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
xx01r = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
cx = u1 @ u2 @ xx01 @ u3 @ u4 @ u5
cx01r = u1r @ u2r @ xx01r @ u3r @ u4r @ u5r
def comp_m(m):
    real = (m + np.conj(m)) / 2
    im = (m - np.conj(m)) / 2 * -1j
    for i in range(len(real)):
        for j in range(len(real)):
            real[i][j] = np.round(real[i][j],2)
    for i in range(len(im)):
        for j in range(len(im)):
            im[i][j] = np.round(im[i][j],2)
    return real + 1j * im
mat = np.eye(9)
mat[3][3], mat[3][4] = mat[3][4], mat[3][3]
mat[4][4], mat[4][3] = mat[4][3], mat[4][4]
#print(comp_m(-1j * (cx01)))
#dd01 = np.kron(R(0, np.pi,0,1), R(0, np.pi,0,1))
h = np.kron(I,R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1))
#u1 = np.kron(R(np.pi / 2, -np.pi, 1, 2), I)
#u2 = np.kron(R(0, np.pi / 2, 0, 1), I)
#u3 = np.kron(R(np.pi / 2, -np.pi, 0, 1), R(0, -np.pi, 0, 1))
#u4 = np.kron(R(0, -np.pi / 2, 0, 1), I)
#u5 = np.kron(R(np.pi / 2, np.pi, 1, 2), I)
h = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1)
#dd01 = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
#dx01 = h @ cx01
#print()
#print(comp_m(cx01 @ cx01r @ cx01))
#print()
T = h @ R(np.pi/2, np.pi/8, 0,1) @ h
#print(comp_m(T))
#print(np.linalg.det(make_ms_matrix(3, 0, np.pi / 2,0,1,0,2)))

T = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, -1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, -1j, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1]]
) * -1j

T = T / np.linalg.det(T)
L = 1
BASIS = np.array([[[1,0,0]],[[0,1,0]]])
ro = np.zeros((27,27))
for vec1 in BASIS:
    for vec2 in BASIS:
        for vec3 in BASIS:
            ro += 1 / 8 **0.5 * m3(vec1.T,vec2.T,vec3.T)@ m3(vec1,vec2,vec3)

cx = u1 @ u2 @ xx01 @ u3 @ u4 @ u5
ilonmask = np.array([[-1j, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, -1j, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, -1j, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, -1j, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1]])
#print(ro)
def func(xx):

    rs = np.eye(9)


    z1 = np.kron(R(0,xx[0], 0, 1) @ R(0,xx[0], 0, 1), R(0,xx[0], 0, 1) @ R(0,xx[0], 0, 1))
    z2 = np.kron(R(0,xx[1], 0, 2) @ R(0,xx[1], 0, 2), R(0,xx[1], 0, 2) @ R(0,xx[1], 0, 2))
    z3 = np.kron(R(0,xx[2], 1, 2) @ R(0,xx[2], 1, 2), R(0,xx[2], 1, 2) @ R(0,xx[2], 1, 2))



    Ts = z1 @ z3 @ z2

    return(abs(Ts - ilonmask).sum())




def sh_func(x):

    rs = np.eye(9)

    r1 = m2(R(0, x[0], 0, 1), I)
    r2 = m2(R(np.pi / 2, x[1], 0, 1), I)
    r3 = m2(R(0, x[2], 0, 2), I)
    r4 = m2(R(np.pi / 2, x[3], 0, 2), I)
    r5 = m2(R(0, x[4], 1, 2), I)
    r6 = m2(R(np.pi / 2, x[5], 1, 2), I)

    TT = r1 @ r2 @ r3 @ r4 @ r5 @ r6
    return TT
for _ in range(0):
    bnds = []
    for i in range(3):
        bnds.append((-np.pi, np.pi))
    bnds = np.array(bnds)
    guess = []
    for i in range(3):
        guess.append(random.randint(-1000, 1000) / 1000 * np.pi)
        #guess.append(2)
    guess = np.array(guess)
    res1 = scipy.optimize.minimize(func, guess, bounds=bnds)
    if res1.fun < 10:
        print(res1)
        print(list(res1.x))

xx01 = make_ms_matrix(3, 0, np.pi / 2,0,1,0,1)
theta =  - 0
z1 = np.kron(R(0,theta , 0, 1) @ R(0,theta , 0, 1), R(0,theta, 0, 1) @ R(0,theta , 0, 1))
z2 = np.kron(R(0,theta /2, 0, 2) @ R(0,theta / 2, 0, 2), R(0,theta / 2, 0, 2) @ R(0,theta / 2, 0, 2))
z3 = np.kron(R(0,theta, 1, 2) @ R(0,theta, 1, 2), R(0,theta, 1, 2) @ R(0,theta, 1, 2))

R1 = np.kron(R(np.pi, np.pi/2,1,2), I)
R2 = np.kron(R(np.pi, np.pi/2,1,2), I)
#print(make_ms_matrix(3,0,np.pi/2, 0, 1, 0, 1))
swap = np.eye(9)
swap[]
xx = make_ms_matrix(3,0,np.pi/2, 0,1,0,2)
xx = TwoQS([0,1,0,2])
xx1 = TwoQS([0,2,0,1])
xx2 = TwoQS([0,1,1,2])
xx3 = TwoQS([0,2,0,1])
xx4 = TwoQS([0,1,0,2])
circuit1.append([xx(q1,q3)], strategy=InsertStrategy.INLINE)
circuit1.append([xx1(q3,q1)], strategy=InsertStrategy.INLINE)
#circuit1.append([xx2(q2,q3)], strategy=InsertStrategy.INLINE)
circuit1.append([xx3(q3,q2)], strategy=InsertStrategy.INLINE)
circuit1.append([xx4(q1,q2)], strategy=InsertStrategy.INLINE)
