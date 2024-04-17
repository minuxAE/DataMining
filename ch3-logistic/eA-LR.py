import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt

np.random.seed(2024) # 设置随机数种子

filename='data.txt'

# 读取数据
def loadData():
    X, y = [], []

    with open(filename, 'r') as f:
        for line in f.readlines():
            content = line.strip().split() # 去除两端空白后，分割字段
            X.append([1.0, content[0], content[1]]) # 1表示常数项
            y.append(content[2])

        return np.array(X, dtype='float'), np.array(y, dtype='int')
    
# sigmoid函数
def sig(x):
    return 1.0/(1.0+np.exp(-x))

# 随机梯度下降算法
def stoGradient(X, y):
    m, d = X.shape # m个样本，d个特征
    weights = np.ones((d, 1)).ravel() # 初始化权重向量为一个全1向量
    max_iter = 1000 # 最多进行1000轮迭代
    for j in range(max_iter):
        sample_index = [i for i in range(m)]
        for i in range(m):
            alpha = 4/(1+i+j)+0.0001 # 更新速率，先快后慢
            rand_index = int(np.random.uniform(0, len(sample_index))) # 均匀随机采样，抽取计算loss的样本编号
            h = sig(X[rand_index, :]@weights.ravel()) # 计算这些随机样本的预测标签
            err = y[rand_index] - h # 计算预测标签和真实标签之间的误差
            weights = weights + alpha * err * X[rand_index] # 更新权重

            del sample_index[rand_index] # 被选中的样本在下一轮中剔除，进行不放回采样
    
    return weights

# 绘图函数
def plotFunc(X, y, weights, title='LR+Stochastic Gradient'):
    x1, y1 = [], [] # 正例坐标(x,y)
    x2, y2 = [], [] # 负例坐标(x,y)
    m, d = X.shape
    for i in range(m):
        if y[i] == 1:
            x1.append(X[i, 1]), y1.append(X[i, 2])
        else:
            x2.append(X[i, 1]), y2.append(X[i, 2])
    
    xpts = np.arange(-4, 4, 0.1)
    ypts = (-weights[0])/weights[2] - weights[1]*xpts / weights[2]

    plt.scatter(x1, y1, s=30, color='red', marker='s')
    plt.scatter(x2, y2, s=30, color='green', marker='o')
    plt.plot(xpts, ypts, 'b-')
    plt.xlabel('X1'), plt.ylabel('X2')
    plt.title(title)
    plt.show()


def callSklearn(X, y):
    lr = LR(fit_intercept=True).fit(X[:, 1:], y) # 调包时不用加入全1列, 因为拟合截距参数被设置为True
    print(np.c_[lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]].ravel())
    return np.c_[lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1]].ravel()

# 全样本梯度下降算法
def batchGradient(X, y):
    m, d = X.shape
    weights = np.ones((d, 1)).ravel()
    max_iter = 1000
    for _ in range(max_iter):
        for _ in range(m):
            alpha = 0.0001 # 固定速率
            h = sig(X@weights.ravel())
            err = y-h
            weights = weights + alpha * X.T@err
    
    return weights

def main():
    X, y = loadData()
    weights = stoGradient(X, y)
    print(weights) # [14.53, 0.87, -1.88]
    weights2 = callSklearn(X, y)
    # plotFunc(X, y, weights)
    print(weights2) # [11.38607726  0.85767013 -1.54232428]
    # plotFunc(X, y, weights2, 'LR + Sklearn')
    weights3 = batchGradient(X, y)
    print(weights3) # [11.23511046  1.00901157 -1.53604833] 结果和sklearn包中的相近
    plotFunc(X, y, weights3, 'LR + batch Gradient')

if __name__ == '__main__':
    main()