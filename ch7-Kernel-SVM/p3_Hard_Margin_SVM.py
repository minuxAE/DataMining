"""
计算案例：Hard Margin SVM
"""
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

with open("data/svm-data.txt") as f:
    for line in f.readlines():
        x1, x2, y = line.strip().split(' ')
        X.append(np.array([float(x1), float(x2)]))
        Y.append(int(y))


X = np.array(X)
y = np.array(y)

# 求解对应的Lagrangian问题可以获得w和b的值
w = np.array([0.833, 0.334])
b = -3.332

## 转换为点斜式
_k = -w[0] / w[1]
_b = -b / w[1]

## 提取部分支持向量
### y=+1
p1 = X[0]
p2 = X[1]
k1 = (p2[1]-p1[1]) / (p2[0] - p1[0])
b1 = p1[1] - k1*p1[0]

### y=-1
n1 = X[-2]
n2 = X[-1]
k2 = (n2[1]-n1[1]) / (n2[0] - n1[0])
b2 = n1[1] - k2*n1[0]

xs = np.linspace(0, 5, 100)
ys = _k*xs + _b
plt.scatter(X[:, 0], X[:, 1], color='b')
plt.plot(xs, ys, lw=1, color='r')
plt.plot(xs, k1*xs+b1, lw=1, color='k', ls='--')
plt.plot(xs, k2*xs+b2, lw=1, color='k', ls='--')
plt.show()
