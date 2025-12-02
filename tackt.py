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
T0 = 25
T = 25
N = 4
PMS1 = 0.99
PMS2 = 0.9
t = np.linspace(0, T, nn)

p = 1 - np.exp(-t/T0)

zero = (1-p)**3
n = np.arange(1, 50)
one = 3 * p * (1-p)**2
two = 3 * p**2 * (1-p)
three = p**3
def P(t_):
    return 1-np.exp(-t_ / T0)мс
tackt_fid = (1-(P(T / n)**3 + 3 * P(T / n) ** 2 * (1 - P(T / n)) + 0.2)) ** n
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
ax.scatter(n, tackt_fid, color='purple', s=5, label='3')
#ax.scatter(t, three + one + zero + two, color='orange', s=5, label='sum')
plt.legend()
plt.grid()
plt.show()
