import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math


from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
nn = 100
ans = []
T0 = 25
T = 25
N = 4
PMS1 = 0.005
PMS2 = 0
PMS0 = PMS1
n = np.arange(0, 400, 2)

def C(n_, k_):
    if k_ == n_ or k_ == 0:
        return 1
    if k_ != 1:
        return C(n_-1, k_) + C(n_-1, k_-1)
    else:
        return n_

def P(nt):
    ssum = 0
    for i in range(1, nt - 1, 2):
        ssum += math.comb(nt, i) * PMS0 ** i * (1 - PMS0) ** (nt - i)
    return ssum

p = []
for nn in n:
    p.append(P(nn))
p = np.array(p)

zero = (1-p)**3
one = 3 * p * (1-p)**2
two = 3 * p**2 * (1-p)
three = p**3

gvot = np.arange(5,1000,1)
wanted = 1000
PMS1 = 0.005
PMS2 = 0.04
PMS0 = PMS2

cl = []
for i in gvot:
    t = wanted // i
    cl.append(((1 - P(i)) ** 3 + 3 * P(i) * (1-P(i))**2)**t * (1 - 20 * PMS1)**t)


p = []
for nn in gvot:
    #3 * p ** 2 * (1 - p) + p**3
    ppp = P(wanted)
    p.append(1 - 3 * ppp ** 2 * (1 - ppp) + ppp**3)
p = np.array(p)

sl = p
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(gvot, 1 -np.array(cl), color='b', s=5, label='0')
#ax.scatter(n, one, color='r', s=5, label='1')
#ax.scatter(n, two, color='g', s=5, label='2')
ax.scatter(gvot, 1 - np.array(sl), color='purple', s=5, label='3')
#ax.scatter(n, three + one + zero + two, color='orange', s=5, label='sum')
plt.legend()
plt.grid()
plt.show()
