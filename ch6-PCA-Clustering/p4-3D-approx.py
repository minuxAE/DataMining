"""
考虑iris数据的前三个主成分
"""
import numpy as np
Sigma = np.array([
    [0.681, -0.039, 1.265],
    [-0.039, 0.187, -0.320],
    [1.265, -0.32, 3.092]
])

# 矩阵分解得到特征值和特征向量
e_vals, U = np.linalg.eig(Sigma)
## 特征值：lam1 = 3.662, lam2 = 0.239, lam3 = 0.059
lam1, lam2, lam3 = e_vals[0], e_vals[1], e_vals[2]
## 总方差
vD = lam1 + lam2 + lam3 # 3.96

## 提取特征向量中的u1, u2, u3
u1 = U[:, 0] # (-0.390, 0.089, -0.916)
u2 = U[:, 1] # (-0.639, -0.742, 0.200)
u3 = U[:, 2] # (-0.663, 0.664, 0.346)

# 计算使用r=1,r=2,r=3时可解释方差的比率
## r=1
f1 = lam1 / vD # 92.46%
f2 = (lam1 + lam2) / vD # 93.95%
f3 = (lam1 + lam2 + lam3) / vD # 100%

