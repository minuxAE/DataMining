"""
寻找第一主成分，进行一维最优近似
"""
import numpy as np

Sigma = np.array([
    [0.681, -0.039, 1.265],
    [-0.039, 0.187, -0.320],
    [1.265, -0.320, 3.092]
])

"""
三个维度上的方差可以从主对角线的元素中直接读取
"""
sigma1 = 0.681
sigma2 = 0.187
sigma3 = 3.092

# 进行特征值分解
eig_values, eig_vectors = np.linalg.eig(Sigma)
# 获取最大特征值
lam1 = eig_values[0] # 3.6615
# 获取对应的特征向量
u1 = eig_vectors[:, 0] # [-0.3901, 0.0887, -0.9165]
# 计算数据的总方差(数据点已经居中，该值也等于数据方差)
V = np.trace(Sigma) # 3.96
# 计算MSE
mse1 = V - lam1 # 0.298
mse2 = V - u1.T@Sigma@u1 # 0.298
