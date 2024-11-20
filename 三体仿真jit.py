from numba import jit
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
T = time.time()

@jit(nopython=True)
def compute_forces():
    N = 1000000 # 仿真总步数
    dt = 0.003
    G = 1 # 引力常量

    m = np.array([1, 0.9, 1.1], dtype=np.float64)

    x = np.array([0, 0.3, 1], dtype=np.float64)
    y = np.array([-0.3, 0, -1], dtype=np.float64)
    z = np.array([-0.1, 0.6, -1], dtype=np.float64)

    vx = 0.1 * np.array([1, -1, 0], dtype=np.float64)/m
    vy = 0.1* np.array([0, -0.5, 0.5], dtype=np.float64)/m
    vz = 0.1 * np.array([0.1, -0.4, 0.3], dtype=np.float64) / m

    num = 3 # 有几个个星球
    epsilon = 0.1 # 避免r太小导致F太大

    xl = np.zeros((num, N), dtype=np.float64)
    yl = np.zeros((num, N), dtype=np.float64)
    zl = np.zeros((num, N), dtype=np.float64)

    for t in range(N):
        for i in range(num):
            for j in range(num):
                if i != j:
                    r = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2 )
                    r = r if r>epsilon else epsilon
                    F = G*m[i]*m[j] / r
                    vx[i] += (x[j] - x[i]) / r * F / m[i] * dt
                    vy[i] += (y[j] - y[i]) / r * F / m[i] * dt
                    vz[i] += (z[j] - z[i]) / r * F / m[i] * dt
        for i in range(num):
            x[i] += dt * vx[i]
            y[i] += dt * vy[i]
            z[i] += dt * vz[i]

            xl[i][t] = x[i]
            yl[i][t] = y[i]
            zl[i][t] = z[i]

    return xl, yl, zl

xl,yl,zl = compute_forces()
print('用时：', time.time()-T)

df = pd.DataFrame({
    'x1' : xl[0]-np.mean(xl[0]),
    'y1' : yl[0]-np.mean(yl[0]),
    'z1' : zl[0]-np.mean(zl[0]),

    'x2' : xl[1]-np.mean(xl[1]),
    'y2' : yl[1]-np.mean(yl[1]),
    'z2' : zl[0]-np.mean(zl[0]),

    'x3' : xl[2]-np.mean(xl[2]),
    'y3' : yl[2]-np.mean(yl[2]),
    'z3' : zl[0]-np.mean(zl[0]),
})

df = df.iloc[::10]
print(len(df))
df.to_csv('santi_距离反比.csv', index=False)
df = df.iloc[:1000:10]
df.plot()
plt.show()


