import numpy as np
import cirq
import networkx as nx
def printm(m):
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

m1 = np.eye(10)
m2 = np.eye(10)
m1[1][1], m1[1][8] =  m1[1][8], m1[1][1]
m1[8][8], m1[8][1] =  m1[8][1], m1[8][8]
m2[2][2], m2[2][7] =  m2[2][7], m2[2][2]
m2[7][7], m2[7][2] =  m2[7][2], m2[7][7]
#print(m1@m2)



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
L = 4
BASIS = np.array([[[1,0,0]],[[0,1,0]]])
ro = np.zeros((27,27))
for vec1 in BASIS:
    for vec2 in BASIS:
        for vec3 in BASIS:
            ro += 1 / 8 **0.5 * m3(vec1.T,vec2.T,vec3.T)@ m3(vec1,vec2,vec3)

#print(ro)
def func(x):
    rs = np.eye(27)
    for i in range(0,L, 5):
        r1 = R(x[12 * i +0], x[12 * i +1], 0, 1)
        r2 = R(x[12 * i +2], x[12 * i +3], 0, 1)
        r3 = R(x[12 * i +4], x[12 * i +5], 1, 2)
        r4 = R(x[12 * i +6], x[12 * i +7], 1, 2)
        r5 = R(x[12 * i +8], x[12 * i +9], 0, 2)
        r6 = R(x[12 * i +10], x[12 * i +11], 0, 2)
        i += 1
        r13 = np.kron(r1, R(x[12 * i + 0], x[12 * i + 1], 0, 1))
        r14 = np.kron(r2, R(x[12 * i + 2], x[12 * i + 3], 0, 1))
        r15 = np.kron(r3, R(x[12 * i + 4], x[12 * i + 5], 1, 2))
        r16 = np.kron(r4, R(x[12 * i + 6], x[12 * i + 7], 1, 2))
        r17 = np.kron(r5, R(x[12 * i + 8], x[12 * i + 9], 0, 2))
        r18 = np.kron(r6, R(x[12 * i + 10], x[12 * i + 11], 0, 2))
        i += 1
        r7 = np.kron(r13, R(x[12 * i + 0], x[12 * i + 1], 0, 1))
        r8 = np.kron(r14, R(x[12 * i + 2], x[12 * i + 3], 0, 1))
        r9 = np.kron(r15, R(x[12 * i + 4], x[12 * i + 5], 1, 2))
        r10 = np.kron(r16, R(x[12 * i + 6], x[12 * i + 7], 1, 2))
        r11 = np.kron(r17, R(x[12 * i + 8], x[12 * i + 9], 0, 2))
        r12 = np.kron(r18, R(x[12 * i + 10], x[12 * i + 11], 0, 2))
        i += 1
        ms1 = np.kron(I, make_ms_matrix(3, x[12 * i + 0], x[12 * i + 1], 0, 1, 0, 1))
        ms2 = np.kron(I, make_ms_matrix(3, x[12 * i + 2], x[12 * i + 3], 0, 1, 0, 1))
        ms3 = np.kron(I, make_ms_matrix(3, x[12 * i + 4], x[12 * i + 5], 0, 1, 0, 1))
        ms4 = np.kron(I, make_ms_matrix(3, x[12 * i + 6], x[12 * i + 7], 0, 1, 0, 1))
        ms5 = np.kron(I, make_ms_matrix(3, x[12 * i + 8], x[12 * i + 9], 0, 1, 0, 1))
        ms6 = np.kron(I, make_ms_matrix(3, x[12 * i + 10], x[12 * i + 11], 0, 1, 0, 1))
        i += 1
        ms7 = np.kron(make_ms_matrix(3, x[12 * i + 0], x[12 * i + 1], 0, 2, 0, 2), I)
        ms8 = np.kron(make_ms_matrix(3, x[12 * i + 2], x[12 * i + 3], 0, 2, 0, 2), I)
        ms9 = np.kron(make_ms_matrix(3, x[12 * i + 4], x[12 * i + 5], 0, 2, 0, 2), I)
        ms10 = np.kron(make_ms_matrix(3, x[12 * i + 6], x[12 * i + 7], 0, 2, 0, 2), I)
        ms11 = np.kron(make_ms_matrix(3, x[12 * i + 8], x[12 * i + 9], 0, 2, 0, 2), I)
        ms12 = np.kron(make_ms_matrix(3, x[12 * i + 10], x[12 * i + 11], 0, 2, 0, 2), I)

        rs = rs @ r7 @ ms1 @ ms7 @ r8 @ ms2 @ ms8  @ r9 @ ms3 @ ms9  @ r10 @ ms4 @ ms10  @ r11 @ ms5 @ ms11  @ r12 @ ms6 @ ms12
    matrix = rs @ ro @ dag(rs)
    total = abs(matrix).sum()
    minor = abs(matrix[np.array([0,1,2,3,4,5,6,7,8])[:,np.newaxis],np.array([0,1,2,3,4,5,6,7,8])]).sum()
    return(total - minor)

def sh_func(x):
    rs = np.eye(27)
    for i in range(0, L, 5):
        r1 = R(x[12 * i + 0], x[12 * i + 1], 0, 1)
        r2 = R(x[12 * i + 2], x[12 * i + 3], 0, 1)
        r3 = R(x[12 * i + 4], x[12 * i + 5], 1, 2)
        r4 = R(x[12 * i + 6], x[12 * i + 7], 1, 2)
        r5 = R(x[12 * i + 8], x[12 * i + 9], 0, 2)
        r6 = R(x[12 * i + 10], x[12 * i + 11], 0, 2)
        i += 1
        r13 = np.kron(r1, R(x[12 * i + 0], x[12 * i + 1], 0, 1))
        r14 = np.kron(r2, R(x[12 * i + 2], x[12 * i + 3], 0, 1))
        r15 = np.kron(r3, R(x[12 * i + 4], x[12 * i + 5], 1, 2))
        r16 = np.kron(r4, R(x[12 * i + 6], x[12 * i + 7], 1, 2))
        r17 = np.kron(r5, R(x[12 * i + 8], x[12 * i + 9], 0, 2))
        r18 = np.kron(r6, R(x[12 * i + 10], x[12 * i + 11], 0, 2))
        i += 1
        r7 = np.kron(r13, R(x[12 * i + 0], x[12 * i + 1], 0, 1))
        r8 = np.kron(r14, R(x[12 * i + 2], x[12 * i + 3], 0, 1))
        r9 = np.kron(r15, R(x[12 * i + 4], x[12 * i + 5], 1, 2))
        r10 = np.kron(r16, R(x[12 * i + 6], x[12 * i + 7], 1, 2))
        r11 = np.kron(r17, R(x[12 * i + 8], x[12 * i + 9], 0, 2))
        r12 = np.kron(r18, R(x[12 * i + 10], x[12 * i + 11], 0, 2))
        i += 1
        ms1 = np.kron(I, make_ms_matrix(3, x[12 * i + 0], x[12 * i + 1], 0, 1, 0, 1))
        ms2 = np.kron(I, make_ms_matrix(3, x[12 * i + 2], x[12 * i + 3], 0, 1, 0, 1))
        ms3 = np.kron(I, make_ms_matrix(3, x[12 * i + 4], x[12 * i + 5], 0, 1, 0, 1))
        ms4 = np.kron(I, make_ms_matrix(3, x[12 * i + 6], x[12 * i + 7], 0, 1, 0, 1))
        ms5 = np.kron(I, make_ms_matrix(3, x[12 * i + 8], x[12 * i + 9], 0, 1, 0, 1))
        ms6 = np.kron(I, make_ms_matrix(3, x[12 * i + 10], x[12 * i + 11], 0, 1, 0, 1))
        i += 1
        ms7 = np.kron(make_ms_matrix(3, x[12 * i + 0], x[12 * i + 1], 0, 1, 0, 1), I)
        ms8 = np.kron(make_ms_matrix(3, x[12 * i + 2], x[12 * i + 3], 0, 1, 0, 1), I)
        ms9 = np.kron(make_ms_matrix(3, x[12 * i + 4], x[12 * i + 5], 0, 1, 0, 1), I)
        ms10 = np.kron(make_ms_matrix(3, x[12 * i + 6], x[12 * i + 7], 0, 1, 0, 1), I)
        ms11 = np.kron(make_ms_matrix(3, x[12 * i + 8], x[12 * i + 9], 0, 1, 0, 1), I)
        ms12 = np.kron(make_ms_matrix(3, x[12 * i + 10], x[12 * i + 11], 0, 1, 0, 1), I)

        rs = rs @ r7 @ ms1 @ ms7 @ r8 @ ms2 @ ms8 @ r9 @ ms3 @ ms9 @ r10 @ ms4 @ ms10 @ r11 @ ms5 @ ms11 @ r12 @ ms6 @ ms12

    return rs
bnds = []
for i in range(L*60):
    bnds.append((-np.pi, np.pi))
bnds = np.array(bnds)
guess = []
for i in range(L * 60):
    guess.append(random.randint(0, 1000) / 1000 * 2 * np.pi)
guess = np.array(guess)
#res1 = scipy.optimize.minimize(func, guess, bounds=bnds)
#print(res1)
#print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
#print(list(res1.x))

print(comp_m(cx))