"""
调用sklearn中的KMeans++接口实现聚类
API地址: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html
"""

from sklearn.datasets import make_blobs as MB
from sklearn.cluster import kmeans_plusplus as KPP
import matplotlib.pyplot as plt

# 产生随机样本点
n_samples = 6000
n_components = 4 # 大致形成4个簇

X, y = MB(
    n_samples= n_samples, centers = n_components, cluster_std=0.8, random_state=0
)

# 调用KMeans++算法进行聚类
center_pts, indices = KPP(X, n_clusters=4, random_state=2024)

# 绘制图像
plt.figure(1)
colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']

for k, color in enumerate(colors):
    cluster_data = y == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=color, marker='.', s=10)

plt.scatter(center_pts[:, 0], center_pts[:, 1], c='k', s=50)
plt.title('KMeans++ Initialization')
plt.xticks([])
plt.yticks([])
plt.show()
