"""
实验A：绘制决策边界（典型超平面）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(2222)

# 样本点的特征向量
X = np.array([[3, 3], [4, 3], [1, 1]])
# 样本点的标签
Y = np.array([1, 1, -1])

# 使用SVM线性分类器进行拟合
clf = svm.SVC(kernel='linear').fit(X, Y)

# 绘制决策边界（典型超平面）h(x)=w'x+b=1/2x1+1/2x2-2=0
w = clf.coef_[0]
# 将标准方程转换为点斜式方程 
k = -w[0] / w[1] # 斜率信息
b = -clf.intercept_[0] / w[1] # 偏置信息

xx = np.linspace(-5, 5)
yy = k*xx + b

# 绘制上下边界（穿过支持向量点的直线）x1和x3是支持向量
b1 = clf.support_vectors_[0] # 提取下边界直线信息（根据x1得到）
yy_down = k * xx + (b1[1] - k * b1[0]) # 下边界
b2 = clf.support_vectors_[-1] # 提取上边界直线信息（根据x3得到）
yy_up = k * xx + (b2[1] - k * b2[0]) # 上边界

plt.plot(xx, yy, 'k-', lw=1) # 典型超平面
plt.plot(xx, yy_down, 'b--', lw=1) # 下边界
plt.plot(xx, yy_up, 'b--', lw=1) # 上边界

# 绘制数据点
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='red') # 支持向量
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired) # 所有样本点
plt.axis('tight')
plt.show()

