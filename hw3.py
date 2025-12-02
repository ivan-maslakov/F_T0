import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy


# Создать графики
def sred(x,y, par):
    y = y[:len(y) - len(y) % par]
    return x[:len(x) - par:par], (np.reshape(y, (len(y) // par, par))).sum(axis=1) / par

def integral(X,Y):
    iy =[]
    X = np.array(X)
    #print(X[2])
    dx = (X[len(X) - 1] - X[0]) / len(X)
    int_sum = 0
    for y in Y:
        int_sum += y * dx
        iy.append(int_sum)
    return int_sum

E = 1000000
ans = []
E_ = np.linspace(-125/16, 4,100)
for E in E_:
    print(E)
    print()
    print()
    print()
    d = 0
    u = 0
    ar = []
    sch = 0
    delt = 0.001
    prev = -100
    for x_ in np.linspace(-5, 5, 100000):
        if abs(3 * x_**4 - 2 * x_**3 - 9 * x_**2 + 4 - E) < delt and abs(prev - x_) > 0.1:

            ar.append(x_)
            prev = x_
    print(len(ar))
    d, u = ar[0], ar[len(ar) - 1]
    print(d,u, len(ar))
    x = np.linspace(d + 0.001, u - 0.001, 10000)
    y = 2/(2*(E- 3 * x**4 + 2 * x**3 + 9 * x**2 - 4))**0.5
    int_s = integral(x,y)
    ans.append(int_s)
    print(int_s)
plt.plot(E_, ans)
plt.show()