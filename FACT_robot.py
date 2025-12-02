import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('imu_test (1).csv')

# Вывести первые несколько строк данных
#print(data.head())

# Изучить статистические характеристики данных
#print(data.describe())

# Проверить наличие пропущенных значений
#print(data.isnull().sum())

# Создать подзаголовки графиков
subplots_titles = ['координата по оси X', 'координата по оси Y', 'координата по оси Z',
                   'Гироскопический угол по оси X', 'Гироскопический угол по оси Y', 'Гироскопический угол по оси Z']

# Создать графики
def sred(x,y, par):
    y = y[:len(y) - len(y) % par]
    return x[:len(x) - par:par], (np.reshape(y, (len(y) // par, par))).sum(axis=1) / par

def integral(X,Y):
    iy =[]
    X = np.array(X)
    print(X[2])
    dx = (X[len(X) - 1] - X[0]) / len(X)
    int_sum = 0
    for y in Y:
        int_sum += y * dx
        iy.append(int_sum)
    return X, iy

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

for i, ax in enumerate(axes.flatten()):
    #fig = plt.figure(figsize=(7, 4))
    #ax = fig.add_subplot()
    column_name = data.columns[i + 2]
    t = data['ts']
    d = data[column_name]
    t, d = sred(t, d, 100)
    id = []
    t, id = integral(t, d)
    if i < 3:
        t, id = integral(t, id)
    #d = d - d.mean()
    #print(len(d))
    print(len(t), len(d))

    ax.scatter(t, d, s = 3)
    ax.set_title(subplots_titles[i])
    #ax.set_xlabel('Время, сек')
    #ax.set_ylabel('Значение')

    #n = ax.hist(d, bins=100)
    #print(n)

subplots_titles = ['Ускорение по оси X', 'Ускорение по оси Y', 'Ускорение по оси Z',
                   'Гироскопическая скорость по оси X', 'Гироскопическая скорость по оси Y', 'Гироскопическая скорость по оси Z']


fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
AR = []
for ii, ax in enumerate(axes.flatten()):
    #fig = plt.figure(figsize=(7, 4))
    #ax = fig.add_subplot()
    column_name = data.columns[ii + 2]
    t = data['ts']
    d = data[column_name]
    #t,d = sred(t,d,200)

    max_xx = []
    for i in range(5, len(d) - 5):
        h = 0
        for tt in range(5):
            tt = tt + 1
            if (d[i]>=d[i-tt])and(d[i] >= d[i+tt]):
                h = h+1
            elif (d[i]<=d[i+tt])and(d[i] <= d[i-tt]):
                h = h-1
        if abs(h) == 5:
            max_xx.append(d[i])

    d = max_xx
    print('len_max',len(d))


    #ax.scatter(t, d, s = 3)
    ax.set_title(subplots_titles[ii])
    #ax.set_xlabel('Время, сек')
    #ax.set_ylabel('Значение')

    ar = ax.hist(d, bins=100)
    AR.append(ar)
    #print(n)
    

array = np.random.randn(100000)
ar = plt.hist(array, bins=1000)
fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

for i, ax in enumerate(axes.flatten()):

    n, amp = ar[0], ar[1][:-1]
    n_, amp_ = AR[i][0], AR[i][1][:-1]
    print(len(n), len(amp))
    #fig = plt.figure(figsize=(7, 4))
    #ax = fig.add_subplot()
    column_name = data.columns[i + 2]
    t = data['ts']
    d = data[column_name]
    n = []
    amp = []
    for iii in range(len(n_)):
        if n_[iii] > 0:
            n.append(n_[iii])
            amp.append(amp_[iii])
        if n_[iii] > 20:
            for j in range(10000):
                n.append(n_[iii])
                amp.append(amp_[iii])
    degree = 2
    coefficients = np.polyfit(amp, np.log(n), degree)
    amp = np.array(amp)
    y_ = coefficients[2] + coefficients[1] * amp + amp**2 * coefficients[0]
    sigma = - 1 / (2 * coefficients[0])
    mu = (2*sigma*(np.log(1/((2*np.pi*sigma)**0.5)) - coefficients[2]))**0.5
    mu = coefficients[1] * sigma
    print('coafficients1 ', coefficients[0], ' ', coefficients[2])
    print('coafficients ', sigma**0.5/9.8, ' ',mu )
    print(i)



    # Визуализация результатов

    ax.scatter(amp, y_, color = 'r', s = 3)
    ax.scatter(amp, np.log(n),color = 'b', s = 3)
    ax.set_title(subplots_titles[i])
    #ax.set_xlabel('Время, сек')
    #ax.set_ylabel('Значение')

    #print(n)

plt.show()

