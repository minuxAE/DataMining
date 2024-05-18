"""
典型超平面
"""
from p1_hyperplane import h, label
import numpy as np
# 设分割超平面为：5x1+2x2-20=0
w = np.array([5, 2]) # 权重向量
b = -20

# 支持向量为(2, 2)
xp = np.array([2, 2])
yp = label(xp)

def getS(x):
    return 1.0 / (h(x) * label(x))

s = getS(xp) # s = 1/6

# 缩放后的权向量
ws = s*w # [5/6, 2/6]

# 缩放后的偏置
bs = s*b # -10/3

# 典型超平面方程
def hc(x):
    return ws@x + bs

# 支持向量到典型超平面的距离（间隔）
d = label(xp) * hc(xp) / np.linalg.norm(ws) # 1.114



