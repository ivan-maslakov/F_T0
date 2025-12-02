import matplotlib.pyplot as plt
import numpy as np

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


#a = 1
T_a = []
pars = np.linspace(0,2, 100)
for par in pars:

    #gamma = 0.1
    omega = 1
    C1 = 1 - par
    fis = np.linspace(0,2*np.pi, 232432)
    kor = (1 + (C1 - 1) * np.sin(2 * fis)**2)**0.5
    fi_dot1 = -omega / C1 * (-0.5 - kor / 2) * (1-(1 - kor) / 2 / np.cos(fis) ** 2)


    kor = (1 + (C1 - 1) * np.sin(2 * fis)**2)**0.5
    fi_dot2 = omega * (1-par * (1 + np.sin(fis)**4))
    print((((fi_dot1 - fi_dot2) ** 2).sum() / len(fis))**0.5)
    #plt.scatter(fis, fi_dot1, s = 5, color = 'b')
    #plt.scatter(fis, np.zeros_like(fis), s = 5, color = 'b', alpha=0)
    T_a.append(integral(fis, 1 / fi_dot1))
#plt.plot(fis, fi_dot2, color = 'r')
plt.scatter(pars, T_a, s = 5, color = 'b')
plt.scatter(pars, 2 * np.pi * np.ones_like(pars), s = 5, color = 'r', alpha=0.5)
plt.scatter(pars, 2 * np.pi * (1-pars *3/8), s = 5, color = 'g')
plt.show()