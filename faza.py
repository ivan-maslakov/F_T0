import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Определение системы дифференциальных уравнений
def system(y, t):
    """
    Определяет систему дифференциальных уравнений.

    Args:
      y: Список из двух элементов, представляющий текущее состояние системы (y[0], y[1]).
      t: Время.

    Returns:
      Список из двух элементов, представляющий производные y[0] и y[1] по времени.
    """
    a = -10
    gamma = 1
    omega = 1
    y1, y2 = y
    dy1dt = y2
    dy2dt = y1*(gamma*y2**2 - omega**2) / (1- gamma * y1 **2)# Замените на ваше уравнение для dy2/dt
    #dy2dt = -12 * y1**3 + 6*y1**2 + 18 *y1
    return [dy1dt, dy2dt]

par = 3
# Создание сетки точек на фазовой плоскости
y1 = np.linspace(-par, par, 40)
y2 = np.linspace(-par, par, 40)
Y1, Y2 = np.meshgrid(y1, y2)

# Вычисление направления вектора в каждой точке
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = system([x, y], 0)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]

# Построение фазового портрета
plt.figure()
Q = plt.quiver(Y1, Y2, u, v, color='r')
plt.xlabel('y1')
plt.ylabel('y2')
plt.title('Фазовый портрет')
plt.xlim([-par,par])
plt.ylim([-par, par])

# Построение нескольких траекторий
for y0 in [[-0.3, 0.3], [0.3, -0.3], [0.5, 0.5]]:
    t = np.linspace(0, 10, 100)
    ys = odeint(system, y0, t)
    plt.plot(ys[:, 0], ys[:, 1], 'b-')  # Траектория синим цветом

plt.show()
