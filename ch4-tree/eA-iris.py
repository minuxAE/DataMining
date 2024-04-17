from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np

# 从远程uci中读取iris数据集(有延迟)
# 使用pandas的绘图功能绘制特征的直方图
def feature_hist():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # 特征和类别
    datasets = pd.read_csv(url, names=names)
    print(datasets.describe())
    datasets.hist()
    plt.show()

# 建立决策树分类器进行拟合
def func_dtc_1():
    iris = load_iris() # 从sklearn中读取iris数据集
    # print(iris.data) # 特征
    # print(iris.target) # 标签表示iris是按顺序排列不同种类的
    dtc = DTC() # 使用默认参数
    dtc.fit(iris.data, iris.target) # iris.data是特征矩阵， iris.target 是标签矩阵
    # print(dtc)
    iris.pred = dtc.predict(iris.data)
    # print(iris.pred) # 决策树预测结果

    # 绘制分布图像
    # 共有150个样本，每50个表示一种类别
    X = iris.data
    # 选用前2个特征进行二维绘图
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='s', label='virginica')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best')
    plt.show()

# 使用knn数据集进行决策树实验
def func_dtc_2():
    # X, Y = [], [] # 存储特征和标签
    # knn = open('knn.txt')
    # for line in knn.readlines():
    #     line = line.strip().split()
    #     X.append([int(line[0]), int(line[1])]) # 特征读入
    #     Y.append(int(line[-1])) # 标签读入

    # X = np.array(X)
    # y = np.array(Y)

    # 使用iris数据集进行测试
    iris = load_iris() # 从sklearn中读取iris数据集
    X = iris.data[:, [0, 1]] # 在特征中选择两个进行决策树分类
    y = iris.target

    dtc = DTC(criterion='entropy', max_depth=4) # 决策树，分类准则为entropy, 最大深度为4
    dtc.fit(X, y)

    # 绘制分类界面
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) # 制作网格
    z = dtc.predict(np.c_[xx.ravel(), yy.ravel()]) # 决策树在每个网格点上的预测结果
    z = z.reshape(xx.shape) # 统一矩阵维度
    plt.contour(xx, yy, z, alpha=0.3) # 绘制等高线
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=1)
    plt.show()


# 主函数
def main():
    func_dtc_2()
    # func_dtc_1()
    # feature_hist()

if __name__ == '__main__':
    main()