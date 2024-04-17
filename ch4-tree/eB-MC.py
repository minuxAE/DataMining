from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np

from sklearn.model_selection import cross_val_score as CVS

# 导入MC数据集
from sklearn.datasets import make_classification as MC
Nsamples = 500
X, Y = MC(n_samples=Nsamples, n_features=3, n_informative=3, 
          n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=2024)

# 考察不同深度参数下决策树的准确率
def param_depth_adj():
    ## 决策树拟合
    dtc = DTC().fit(X, Y)
    print(CVS(dtc, X, Y, scoring='accuracy', cv=10).mean()) # 十折交叉验证的平均准确率， 94%
    ## 输出特征重要性
    print(dtc.feature_importances_) # [0.14629012 0.38433736 0.46937252]
    xpts = [i for i in range(1, 25)]
    ypts = []
    for depth in range(1, 25):
        dtc = DTC(max_depth=depth).fit(X, Y)
        acc = CVS(dtc, X, Y, scoring='accuracy', cv=10).mean()
        ypts.append(acc)

    plt.xlabel('Max Depth of Decision Tree')
    plt.ylabel('Accuracy')
    plt.plot(xpts, ypts, 'b-o', lw=1)
    plt.show()

# 对比：使用Logisitc Regression
def func_LR():
    from sklearn.linear_model import LogisticRegression as LR
    lr = LR()
    print(CVS(lr, X, Y, scoring='accuracy', cv=10).mean()) # 准确率也达到了94%左右

# 网格搜索：对多种参数组合进行测试
from sklearn.model_selection import GridSearchCV as GSCV
def func_grid():
    param_grid = [{
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [3, 5, 7, 9, 11, 15, 20],
        'max_depth': [3, 5, 10, 15, 20, 25]
    }]

    gs = GSCV(estimator=DTC(), param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=4) # 4线程并行
    gs.fit(X, Y)
    print(gs.best_estimator_) # 最好的参数组合下的决策树
    print(gs.best_score_) # 该决策树的准确率
    """
    DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='log2',
                       min_samples_split=3)
    0.9419999999999998
    """
    

def main():
    func_grid()
    # func_LR()
    # param_depth_adj()


if __name__ == '__main__':
    main()
