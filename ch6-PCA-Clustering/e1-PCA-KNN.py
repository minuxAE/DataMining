"""
实验A: PCA+KNN
"""
from sklearn.datasets import load_digits ## 载入digits数据集
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from time import time ## 计时器

digits = load_digits()
X = digits.data # 特征矩阵 (1797, 64) 每个数字使用8x8的点阵表示，有64个特征
y = digits.target # 标签向量 (1797, 1) 数字为0-9

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2024)


from sklearn.neighbors import KNeighborsClassifier as KC # KNN分类器
def KNN():
    st = time()
    knn_clf = KC().fit(X_train, y_train)
    et = time()
    print('KNN time used {:.4f}, accuracy is {:.4f}'.format(et-st, knn_clf.score(X_test, y_test)))


def PCA_KNN():
    pca = PCA(n_components=4).fit(X_train) # 保留前4个主成分
    # 特征矩阵降维(测试集、训练集都要降维)
    X_train_tr = pca.transform(X_train) 
    X_test_tr = pca.transform(X_test)

    st = time()
    knn_clf = KC().fit(X_train_tr, y_train)
    et = time()
    print('PCA+KNN time used {:.4f}, accuracy is {:.4f}'.format(et-st, knn_clf.score(X_test_tr, y_test)))

    # 输出4个主成分保留信息的比率
    print(pca.explained_variance_ratio_) # 14%, 13%, 11.88% 8%


# 降维+白化 可视化
import matplotlib.pyplot as plt
import numpy as np
def PCA_Whiten_Vis():
    def plot_images(Xi): # 展示前5个图像
        _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
        for ax, image, label in zip(axes, Xi.reshape(-1, 8, 8)[:5], y[:5]): # digits.images 等效于 X.reshape(-1, 8, 8)
            ax.set_axis_off() # 关闭显示坐标轴
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Digit: {}'.format(label))
        plt.show()

    

    # 提取前36个主成分，加入whiten
    pca = PCA(n_components=36, whiten=True)
    X_pca = pca.fit_transform(X / 255) # 归一化特征矩阵
    print(pca.explained_variance_ratio_) # 查看特征保留信息的比率

    # 可视化保留多个主成分可以保留的信息比率
    def plot_comp():
        plt.bar(np.arange(1, 1+len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_))
        plt.show()

    # 使用降维后的特征重新可视化数据
    def Re_PCA():
        X_re = pca.inverse_transform(X_pca) # 使用主成分降维后的特征还原到原始特征矩阵
        plot_images(X) # 原始
        plot_images(X_re) # 降维后

    # 保留95%的信息需要主成分的数量
    def keep95():
        pca = PCA(n_components=0.95)
        pca.fit(X_train)
        print(pca.components_.shape) # (28, 64) 需要保留前28个特征才可以达到95%的信息保留
        print(np.cumsum(pca.explained_variance_ratio_))

    keep95()

def work():
    PCA_Whiten_Vis()
    # KNN()
    # PCA_KNN()
    

if __name__ == '__main__':
    work()
