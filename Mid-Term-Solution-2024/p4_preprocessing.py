import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2024)

"""
产生随机数样本
"""
D1 = np.random.gamma(shape=1, scale=2, size=50).reshape(-1, 1) # Gamma分布的样本
D2 = np.random.normal(loc=1, scale=3, size=50).reshape(-1, 1) # Gaussian分布的样本
data = np.concatenate((D1, D2), axis=1) # data.shape = (50, 2)


"""
归一化处理
"""
from sklearn.preprocessing import MinMaxScaler as MM

def func_MM():
    dataMM = MM().fit_transform(data)
    plot_samples_scale(dataMM)

"""
标准化处理：StandardScaler
"""
from sklearn.preprocessing import StandardScaler as SS

def func_SS():
    dataSS = SS().fit_transform(data)
    plot_samples_scale(dataSS)

"""
标准化处理：RobustScaler
"""
from sklearn.preprocessing import RobustScaler as RS

def func_RS():
    dataRS = RS(quantile_range=(10, 90)).fit_transform(data)
    plot_samples_scale(dataRS)

"""
白化
"""
def func_Whiten(X):
    def _center(X):
        return X - np.mean(X, axis=0)
    
    def _whiten(X, correct=True):
        Xc = _center(X)
        U, Lam, V = np.linalg.svd(Xc)
        A = V.T@np.diag(1.0 / Lam)
        return Xc@A*np.sqrt(X.shape[0]) if correct else 1.0
    
    return _whiten(X)

"""
绘图接口
"""
def plot_samples_scale(data):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8)
    plt.xlabel('X'), plt.ylabel('Y')
    plt.show()


def main():
    # plot_samples_scale(data)
    # func_MM()
    # func_SS()
    # func_RS()
    dataW = func_Whiten(data)
    print('白化前数据的协方差矩阵：')
    print(np.cov(data.T))
    print('白化后的数据协方差矩阵：')
    print(np.cov(dataW.T))
    plot_samples_scale(dataW)

if __name__ == '__main__':
    main()