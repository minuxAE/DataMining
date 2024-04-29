"""
Step1 : 生成不平衡的类别样本
"""
from sklearn.datasets import make_classification as MC
Nsamples = 1000
weights = (0.95, 0.05)

X, Y = MC(n_samples=Nsamples, n_features=2, n_redundant=0, weights=weights, random_state=2024)
# 检查样本分布的状态
print(X[Y==0].shape) # (944, 2)
print(X[Y==1].shape) # (56, 2)

"""
Step2 : 重采样
"""
import numpy as np
from sklearn.utils import resample

def func_resample():
    X1s = resample(X[Y==1], n_samples=X[Y==0].shape[0], random_state=2024) # 对少数类进行重采样
    # 合并样本
    Xu = np.concatenate((X[Y==0], X1s))
    Yu = np.concatenate((Y[Y==0], np.ones(shape=(X1s.shape[0], ), dtype=np.int32)))

    # 检查样本的分布状态
    print(Xu[Yu==0].shape) # (944, 2)
    print(Xu[Yu==1].shape) # (944, 2)
    return Xu, Yu

"""
Step3: SMOTE
"""
from imblearn.over_sampling import SMOTE
def func_SMOTE():
    # 设定SMOTE的采样策略为少数类过采样，每次考虑5个邻居点
    smote = SMOTE(random_state=2024, sampling_strategy='minority', k_neighbors=5)
    Xsmo, Ysmo = smote.fit_resample(X, Y)
    
    # 检查样本的分布
    print(Xsmo[Ysmo==0].shape)
    print(Xsmo[Ysmo==1].shape)

    return Xsmo, Ysmo

"""
二分类绘图函数
"""
import matplotlib.pyplot as plt
def plot_samples(X, Y, Nsamples):
    for i in range(Nsamples):
        if Y[i]==0:
            plt.plot(X[i, 0], X[i, 1], marker='.', color='r', markersize=2)
        else:
            plt.plot(X[i, 0], X[i, 1], marker='.', color='b', markersize=2)
    plt.show()

def main():
    # plot_samples(X, Y, Nsamples)
    # Xu, Yu = func_resample()
    # plot_samples(Xu, Yu, Xu.shape[0])
    Xsmo, Ysmo = func_SMOTE()
    plot_samples(Xsmo, Ysmo, Xsmo.shape[0])


if __name__ == '__main__':
    main()


