import matplotlib.pyplot as plt
import numpy as np
fi = np.pi * 0.3
x = np.linspace(0,0.01,100)
kor = 1/2 + (1- x * np.sin(2 * fi)**2)**0.5 / 2
kort = 1 - x * np.sin(2 * fi) ** 2 / 4
kor = 1 - (1 - (1- x * np.sin(2 * fi)**2)**0.5) / 2 / np.cos(fi)**2
kort = 1 - x * np.sin(fi) ** 2
f1 = -1 / (1-x) * (-0.5 - (1-x*np.sin(2*fi)**2)**0.5 / 2) * (1 - (1-(1-x*np.sin(2*fi)**2)**0.5) / 2 / np.cos(fi)**2)
f2 = 1 + x*(1 - np.sin(fi)**2 * (np.cos(fi)**2-1))
f2 = 1 + x*(1 - np.sin(2*fi)**2 / 4 + np.sin(fi)**2)
#f2 = 1 * (1+x) * (1 - x * np.sin(2*fi)**2 / 4 ) * (1 - x * np.sin(2*fi)**2 / 4 / np.cos(fi)**2)
#f2 = 1 / (1-x) * (1 - x * np.sin(2*fi)**2 / 4 ) * (1 - x * np.sin(2*fi)**2 / 4 / np.cos(fi)**2)

#plt.plot(x,f1)
plt.plot(x,f1 - f2, color = 'r')
plt.show()