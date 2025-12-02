import matplotlib.pyplot as plt
import numpy as np
#lim = 2
#x1 = np.linspace(-lim,lim,1000)
#x2 = np.linspace(-lim,lim,1000)
x1n = 10
x2n = 0

gamma = 1
omega = 1
par = 1000000 * 1000
#par =100
def dx2dt(x1, x2):
    return x1*(gamma*x2**2 - omega**2) / (1- gamma * x1 **2)

def dx1dt(x1,x2):
    return x2


fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
def trag(x1n, x2n, clr, lim):

    x1nn = x1n
    x2nn = x2n
    time = np.linspace(0, 2 * np.pi / omega * 2, lim)
    dt = time[len(time)-1] / len(time)
    ans1 = []
    ans2 = []
    ans_c = []
    bdot1 = []
    bdot2 = []
    Ts = []
    for t in time:
        #print((t // dt) % (lim // 1000))
        '''
        if (t // dt) % (lim // 1000) == 0:
            #print((t // dt) % (lim // 1000))
            bdot1.append(x1n)
            bdot2.append(x2n)
        '''
        ans1.append(x1n)
        ans2.append(x2n)
        x1n = x1n + dt * dx1dt(x1n, x2n)
        x2n = x2n + dt * dx2dt(x1n, x2n)
        if abs(x1n - x1nn) < x1nn / par and abs(t - 6.28) < 1:
            Ts.append(t)
        ans_c.append(abs(x2n**2-omega**2/gamma) * abs(x1n**2 - 1/gamma))
        #print(abs(x2n**2-omega**2/gamma) * abs(x1n**2 - 1/gamma))
    T0 = 2 * np.pi / omega
    if len(Ts) > 0:
        T = Ts[len(Ts) // 2]
        print(T, Ts[0], Ts[len(Ts) - 1], (1 - 3 / 8 * gamma * x1nn ** 2) * T0)
        print((T - T0) / gamma / x1nn ** 2 /T0)
    else:
        T = 1






    ax1.scatter(ans1,ans2, color = clr, s = 3)
    ax1.scatter(bdot1, bdot2, color='black', s=5)
    #ax2.scatter(time, ans_c, color = clr, s = 3)
    return
#trag(1.1,0, 'r', 100000)
trag(0.01,0,'b',10000)
#trag(-1.1,0,'r',10000)
#trag(-1.2,0,'g',10000)
#trag(0,-1.1, 'purple', 10000)
#trag(-0.9,1.1, 'orange', 10000)
#trag(0,0.09, 'g', 100000)
# #trag(1.1,0, 'b', 100000)
plt.show()
