import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
import time

T = time.time()

df = pd.read_csv('santi_距离反比.csv')
df = df.iloc[:len(df)//30]
df = df.iloc[::3]
x1 = df['x1'].to_numpy()
y1 = df['y1'].to_numpy()
x2 = df['x2'].to_numpy()
y2 = df['y2'].to_numpy()
x3 = df['x3'].to_numpy()
y3 = df['y3'].to_numpy()

# 创建图形和坐标轴
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# 创建三个点的初始位置
point1, = ax.plot(x1[0], y1[0], 'bo')
point2, = ax.plot(x2[0], y2[0], 'ro')
point3, = ax.plot(x3[0], y3[0], 'go')

# 初始化函数，设置点的初始位置
def init():
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    return point1, point2, point3

# 更新函数，用于动画的每一帧
def update(frame):
    point1.set_data(x1[frame], y1[frame])
    point2.set_data(x2[frame], y2[frame])
    point3.set_data(x3[frame], y3[frame])
    return point1, point2, point3



# 创建动画
ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True, interval=100)

# 保存动画为 GIF
images = []
for i in range(len(x1)):
    ani._init_draw()
    ani._draw_frame(i)  # 绘制当前帧
    buf = io.BytesIO()  # 创建内存文件
    fig.savefig(buf, format='png')  # 保存当前帧到内存文件
    buf.seek(0)
    img = Image.open(buf)  # 使用Pillow打开内存文件
    img = img.convert('P', palette=Image.ADAPTIVE, colors=10)  # 转换颜色
    images.append(img)

# 保存 GIF
images[0].save('santi_animation.gif', save_all=True, append_images=images[1:], optimize=False, duration=30, loop=0)
print('用时:',time.time()-T)

# 显示动画
plt.show()
