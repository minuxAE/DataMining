"""
实验C：不同参数组合测试
"""
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 读取数据
data = load_breast_cancer()
X = data.data # (569, 30)
y = data.target

# 1. 测试不同核函数的svm的accuracy
# Using kernel : linear. Accuracy is 0.9532
# Using kernel : poly. Accuracy is 0.9649
# Using kernel : rbf. Accuracy is 0.6433
# Using kernel : sigmoid. Accuracy is 0.6433
def func1():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2222)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='auto', degree=1).fit(X_train, y_train)
        print('Using kernel : {}. Accuracy is {:.4f}'.format(kernel, clf.score(X_test, y_test)))    

    # 绘制数据点分布图像: 两类样本具有明显的交错现象 --- soft margin SVM
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

# 2.测试标准化步骤对SVM的影响
# Using kernel : linear. Accuracy is 0.9708
# Using kernel : poly. Accuracy is 0.9591
# Using kernel : rbf. Accuracy is 0.9591
# Using kernel : sigmoid. Accuracy is 0.9415
import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
def func2():
    # data = pd.DataFrame(X)
    # print(data.describe([0.05, 0.20, 0.50, 0.75, 0.90]).T) # 查看特征的分位数分布，发现量纲影响明显
    global X # 使用全局变量X
    X = SS().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2222)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='auto', degree=1).fit(X_train, y_train)
        print('Using kernel : {}. Accuracy is {:.4f}'.format(kernel, clf.score(X_test, y_test)))

# 3. 测试核函数（RBF、POLY）的参数调节
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import GridSearchCV as GSV
def func3():
    def rbf_gamma():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2222)
        score = []
        gammas = np.logspace(-10, 1, 50) # 生成50个测试点 10^(-10) ~ 10^(1)
        for gamma in gammas:
            clf = SVC(kernel='rbf', gamma=gamma).fit(X_train, y_train)
            score.append(clf.score(X_test, y_test))

        # 输出最高accuracy
        ## gamma = 2.4*10-5的时候rbf-svc达到最好的分类效果准确率为94.7%
        print(max(score), gammas[score.index(max(score))])

        # 绘制不同gamma对应的accuracy
        plt.plot(gammas, score)
        plt.show()

    def poly_grid(): # 程序运行（网格搜索）需要一定的时间
        gammas = np.logspace(-10, 1, 20)
        c0s = np.linspace(0, 5, 10)
        param_grid = dict(gamma=gammas, coef0 = c0s)
        cv = SSS(n_splits=5, test_size=0.3, random_state=2222) # 分层抽样得到5个不同的测试集
        grid = GSV(SVC(kernel='poly', degree=1), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        print('The best parameters are {} with accuracy {:.4f}'.format(grid.best_params_, grid.best_score_))
        # 搜索范围内的最优参数组合：gamma=10, coef0=1.11, accuracy=95.44%

    poly_grid()


# 4. 测试不同惩罚项C对accuracy的影响
def func4(): # 程序需要一定的运行时间
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2222)
    score = []

    Cs = np.linspace(0.01, 30, 50)
    for C in Cs:
        # 测试linear核: 最优状态C=25.71, acc=0.97
        # clf = SVC(kernel='linear', C=C).fit(X_train, y_train)
        # 测试rbf核：最优状态C=3.07, acc=0.935
        clf = SVC(kernel='rbf', C=C).fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        score.append(acc)
        print('C={:.2f}. Accuracy={:.4f}'.format(C, acc))
    
    print(max(score), Cs[score.index(max(score))])

    plt.plot(Cs, score)
    plt.show()

func4()