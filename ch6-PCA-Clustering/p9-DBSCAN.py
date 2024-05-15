"""
DBSCAN聚类案例
"""
from sklearn.datasets import make_blobs as MB
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

## 数据集1 环形
X1, y1 = datasets.make_circles(n_samples=1000, factor=0.6, noise=0.03)
## 数据集2 球形
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=0.1, random_state=0)

from sklearn.cluster import DBSCAN
X = np.concatenate((X1, X2))
yD = DBSCAN(eps=0.11, min_samples=10).fit_predict(X) # 使用DBSCAN进行聚类并给出预测结果

import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
yK = KMeans(n_clusters=3, random_state=2024).fit_predict(X) # 使用KMeans进行聚类并给出预测结果

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_figwidth(10)
axes[0].scatter(X[:, 0], X[:, 1], c=yD, s=10)
axes[1].scatter(X[:, 0], X[:, 1], c=yK, s=10)

plt.show()

