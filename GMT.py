import matplotlib.pyplot as plt
import numpy as np
lim = 2
x1 = np.linspace(-lim,lim,1000)
x2 = np.linspace(-lim,lim,1000)
a1p = []
a1m = []
a2p = []
a2m = []
om = 1
g = 1
c = 0.1
for x1_ in x1:
    print(x1_)
    for x2_ in x2:
        if abs(c - (x1_**2-1/g) * (x2_**2-om**2/g)) < 0.01:

            if len(a2m) > len(a2p):
                a1p.append(x1_)
                a2p.append(x2_)
            else:
                a1m.append(x1_)
                a2m.append(x2_)
plt.scatter(a1m, a2m, color = 'r', s = 3)
plt.scatter(a1p, a2p, color = 'b', s = 3)
a = 0.5
gamma = -0.1
omega = 1
C = 2 - gamma * a**2
fis = np.linspace(0,2*np.pi, 1000)
d_ts = (1 - 2 * np.cos(fis)**2 * ((1-(1-C * np.sin(2 * fis)**2)**0.5) / np.sin(2 * fis))) / omega / (1 - C * np.sin(2 * fis)**2) ** 0.5
plt.show()
