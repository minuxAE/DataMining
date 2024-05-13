import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, (2, 1)] # 特征矩阵
X = X - np.mean(X, axis=0) # 特征矩阵居中
y = iris.target # 标签向量
N = X.shape[0] # 样本数量
d = X.shape[1] # 特征数量

"""
多元正态pdf
"""
def f(x, mu, sig, d):
    A = 1 / ((2*np.pi)**(d/2) * np.linalg.det(sig)**(1/2))
    B = np.exp(-(x-mu).T@np.linalg.inv(sig)@(x-mu)/2)
    return A*B

"""
初始参数
"""
k = 3 # 簇的数量
Sig = np.array([np.eye(d) for _ in range(k)]) # 协方差矩阵
Mu = np.array([
    [-3.59, 0.25],
    [-1.09, -0.46],
    [0.75, 1.07]
])

Pc = np.array([1/k for _ in range(k)])

def EM():

    def E_Step(_mu, _sig, _P):
        # 计算后验概率（样本点的权重）
        w = np.zeros((N, k))
        for j in range(N):
            for i in range(k):
                w[j, i] = f(X[j,:], _mu[i], _sig[i], d) * _P[i]
        
        w_norm = np.zeros((k, N))
        for i in range(k):
            w_norm[i, :] = w[:, i] / np.sum(w, axis=1)

        return w_norm

    def M_Step(_mu, _sig, _P):
        w = E_Step(_mu, _sig, _P)
        ## 更新均值
        mu = np.zeros((k, d))
        for i in range(k):
            mu_sum = np.zeros(d)
            for j in range(N):
                mu_sum += w[i,j] * X[j,:]
            mu[i] = mu_sum / np.sum(w[i,:])
        
        ## 更新协方差矩阵
        sig = np.array([np.zeros((d, d)) for _ in range(k)])
        for i in range(k):
            sig_sum = np.zeros((d, d))
            for j in range(N):
                sig_sum += w[i,j]*(X[j,:]-mu[i]).reshape(-1, 1)@(X[j,:]-mu[i]).reshape(-1, 1).T
            sig[i] = sig_sum / np.sum(w[i,:])

        ## 更新先验概率
        P = np.zeros(k)
        for i in range(k):
            P[i] = np.sum(w[i,:]) / N

        return mu, sig, P

    def work():
        eps = 1e-3
        mu_pre = Mu
        mu, sig, P = M_Step(Mu, Sig, Pc)
        i = 0
        while np.linalg.norm(mu-mu_pre) > eps:
            print('------------------------Iteration: ', i+1, '--------------------------')
            print(P)
            i+=1
            mu_pre = mu
            mu, sig, P = M_Step(mu, sig, P)

        return mu, sig, P
    return work()

def plot_func():
    mu, sig, P = EM()
    mu0 = np.mean(X[:, 0])
    mu1 = np.mean(X[:, 1])
    for i in range(N):
        if y[i] == 0:
            plt.scatter(X[i, 0] - mu0, X[i, 1] - mu1, marker='s', color='blue', alpha=0.5)
        elif y[i] == 1:
            plt.scatter(X[i, 0] - mu0, X[i, 1] - mu1, marker='^', color='red', alpha=0.5)
        else:
            plt.scatter(X[i, 0] - mu0, X[i, 1] - mu1, marker='o', color='gray', alpha=0.5)
    
    plt.plot(mu[0, 0], mu[0, 1], '*', color='k', markersize=20)
    plt.plot(mu[1, 0], mu[1, 1], '.', color='k', markersize=20)
    plt.plot(mu[2, 0], mu[2, 1], 'd', color='k', markersize=20)
    plt.show()

plot_func()