"""
实验B：绘制分类界面
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 数据集
data = np.array([
    [0.1, 0.7], [0.3, 0.6], [0.4, 0.1], [0.5, 0.4], [0.8, 0.04], [0.42, 0.6], # 正类样本
    [0.9, 0.4], [0.6, 0.5], [0.7, 0.2], [0.7, 0.67], [0.27, 0.8], [0.5, 0.72] # 负类样本
])

labels = [1] * 6 + [0] * 6 # 正类标注为1，负类标注为0

# 尝试使用线性SVM进行分类
C = 0.0001
# svc_linear = svm.SVC(kernel='linear', C=C).fit(data, labels)
# print(svc_linear.score(data, labels)) # 线性分类准确率92%

# 使用Gaussian Kernel进行非线性分类
## 绘制绘图网格
h = 0.002
x_min, x_max = data[:, 0].min()-0.2, data[:, 0].max()+0.2
y_min, y_max = data[:, 1].min()-0.2, data[:, 1].max()+0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.figure(figsize=(15, 10))

# 拟合rbf-SVC（高斯核SVM）
for i, gamma in enumerate([1, 5, 15, 35, 45, 55]):
    # gamma值越大对应的rbf核的非线性程度越高
    svc_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(data, labels)
    z = svc_rbf.predict(np.c_[xx.ravel(), yy.ravel()]) # rbf-svc进行判定
    z = z.reshape(xx.shape)

    plt.subplot(2, 3, i+1)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.contourf(xx, yy, z, cmap=plt.cm.ocean, alpha=0.4) # 绘制等高线，即分类界面
    
    # 样本点图像绘制
    plt.scatter(data[:6, 0], data[:6, 1], marker='o', s=50)
    plt.scatter(data[6:, 0], data[6:, 1], marker='x', s=50)
    plt.title('RBF with $\gamma=$' + str(gamma))

plt.show()