import numpy as np

# 一维数据集上的数据的坐标点
N = 11
x = np.array([1.0, 1.3, 2.2, 2.6, 2.8, 5.0, 7.3, 7.4, 7.5, 7.7, 7.9])

# 簇的数量, 希望将数据点聚类为两个簇
k = 2
# 初始化随机簇参数
Mu = [6.63, 7.57]
Sig = [1, 1]
Pc = [1/k, 1/k]

# 一元正态分布
def f(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mu)**2 / (2*sig**2))

def EM():
    
    def E_step(_mu, _sig, _w):
        # 计算后验概率
        w1, w2 = [], []
        for j in range(N):
            w1.append(f(x[j], _mu[0], _sig[0]) * _w[0])
            w2.append(f(x[j], _mu[1], _sig[1]) * _w[1])

        wa = [0 for _ in range(N)] # 所有点对簇1的权向量
        wb = [0 for _ in range(N)] # 所有点对簇2的权向量
        for i in range(N):
            wa[i] = w1[i] / (w1[i] + w2[i])
            wb[i] = w2[i] / (w1[i] + w2[i])

        return np.array(wa), np.array(wb) 
    
    def M_step(mu, sig, w):
        wa, wb = E_step(mu, sig, w) # 先调用E-step
    
        mu = [wa@x / np.sum(wa), wb@x / np.sum(wb)] # 估计均值参数
        z = np.array([x-np.array(mu[0]), x-np.array(mu[1])]).T
        z2 = np.array([z[:, 0]**2, z[:, 1]**2])

        sig = [np.sqrt(wa@z2[0] / np.sum(wa)), np.sqrt(wb@z2[1] / np.sum(wb))] # 估计方差参数

        w = [np.sum(wa)/N, np.sum(wb)/N] # 估计先验概率

        return mu, sig, w, wa, wb
    
    def work():
        mu, sig, w  = Mu, Sig, Pc
        for i in range(5):
            mu, sig, w, wa, wb = M_step(mu, sig, w)
            print('------------------Iteration: ', i+1, '--------------------')
            print(*zip(wa.round(2), wb.round(2)))
            print('mu1 = ', mu[0], 'mu2 = ', mu[1])
            print('sig1 = ', sig[0], 'sig2 = ', sig[1])
            print('w1 = ', w[0], 'w2 = ', w[1])

    work()

EM()
