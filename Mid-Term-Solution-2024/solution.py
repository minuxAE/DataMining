from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore') # 屏蔽警告

# 生成随机数据点
x1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.3, random_state=2024) # factor表示内外圈点之比
x2, y2 = make_moons(n_samples=1000, noise=0.3, random_state=2024)
x3, y3 = make_blobs(n_samples=1000, n_features=6, centers=4, cluster_std=8.0) # 6个特征，4个中心, 聚类的标准差为8

# 绘图函数
def plot_func():
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.scatter(x2[:, 0], x2[:, 1], marker='o', c=y2)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.scatter(x3[:, 0], x3[:, 1], c=y3)
    plt.xticks([])
    plt.yticks([])
    plt.show()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split

# 拆分训练集和测试集，建立模型进行评估算法效果
def clf_func(X, y, dataset='circles'):
    print('-'*10, dataset, '-'*10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Logistic Regression
    lr = LR().fit(X_train, y_train)
    print('Logistic Regression:')
    print('On training set. The accuracy of Logistic Regression is {:.4f}%'.format(100*lr.score(X_train, y_train)))
    print('On testing set. The accuracy of Logistic Regression is {:.4f}%'.format(100*lr.score(X_test, y_test)))

    # Decision Tree
    dtc = DTC().fit(X_train, y_train)
    print('Decision Tree:')
    print('On training set. The accuracy of Decision Tree is {:.4f}%'.format(100*dtc.score(X_train, y_train)))
    print('On testing set. The accuracy of Decision Tree is {:.4f}%'.format(100*dtc.score(X_test, y_test)))

    # Random Forest
    rfc = RFC().fit(X_train, y_train)
    print('Random Forest:')
    print('On training set. The accuracy of Random Forest is {:.4f}%'.format(100*rfc.score(X_train, y_train)))
    print('On testing set. The accuracy of Random Forest is {:.4f}%'.format(100*rfc.score(X_test, y_test)))


from sklearn.model_selection import GridSearchCV as GSCV
# 参数fine-tunning
def fine_tune(X, y, dataset='circles'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('-'*10, dataset, '-'*10)
    ## 对逻辑回归进行调参
    lr_grid = [{
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [1, 2, 3, 4, 5],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky']
    }]

    lr_gs = GSCV(estimator=LR(), param_grid=lr_grid, scoring='accuracy', cv=10, n_jobs=4)
    lr_gs.fit(X_train, y_train)
    print('Logistic Regression:')
    print('Best Estimator is ', lr_gs.best_estimator_)
    print('On training set. Best Estimator accuracy is {:.4f}%'.format(100*lr_gs.best_score_))
    print('On testing set. Best Estimator is {:.4f}%'.format(100*lr_gs.best_estimator_.score(X_test, y_test)))

    ## 对Decision Tree进行调参
    tr_grid = [{
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'splitter' : ['best', 'random'],
        'max_depth' : [3, 5, 7, 9, 11, 13],
        'min_samples_split' : [3, 5, 7, 9, 11, 13],
        'max_features': ['sqrt', 'log2']
    }]

    dtc_gs = GSCV(estimator=DTC(), param_grid=tr_grid, scoring='accuracy', cv=10, n_jobs=4)
    dtc_gs.fit(X_train, y_train)
    print('Decision Tree Classifier:')
    print('Best Estimator is ', dtc_gs.best_estimator_)
    print('On training set. Best Estimator accuracy is {:.4f}%'.format(100*dtc_gs.best_score_))
    print('On testing set. Best Estimator is {:.4f}%'.format(100*dtc_gs.best_estimator_.score(X_test, y_test)))

    ## 对随机森林进行调参
    rf_grid = [{
        'max_depth': [5, 10, 15, 20, 25],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [5, 10, 15, 20],
        'n_estimators': [10, 20, 30]
    }]

    rfc_gs = GSCV(estimator=RFC(), param_grid=rf_grid, scoring='accuracy', cv=10, n_jobs=-1)
    rfc_gs.fit(X_train, y_train)
    print('Random Forest Classifier:')
    print('Best Estimator is ', rfc_gs.best_estimator_)
    print('On training set. Best Estimator accuracy is {:.4f}%'.format(100*rfc_gs.best_score_))
    print('On testing set. Best Estimator is {:.4f}%'.format(100*rfc_gs.best_estimator_.score(X_test, y_test)))

# 混淆矩阵计算函数
def conf_matrix(X, y, mdl, mdl_name='Logistic Regression', dataset='circles'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
    mdl = mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    # 定义混淆矩阵中的4个变量
    def TN(y_true, y_pred):
        return np.sum((y_true == 0) & (y_pred == 0))
    
    def FP(y_true, y_pred):
        return np.sum((y_true == 0) & (y_pred == 1))
    
    def FN(y_true, y_pred):
        return np.sum((y_true == 1) & (y_pred == 0))
    
    def TP(y_true, y_pred):
        return np.sum((y_true == 1) & (y_pred == 1))
    
    def precision(y_true, y_pred):
        return TP(y_true, y_pred) / (TP(y_true, y_pred) + FP(y_true, y_pred))
    
    def recall(y_true, y_pred):
        return TP(y_true, y_pred) / (TP(y_true, y_pred) + FN(y_true, y_pred))
    
    def confusion_matrix(y_true, y_pred):
        return np.array([
            [TN(y_true, y_pred), FN(y_true, y_pred)],
            [FP(y_true, y_pred), TP(y_true, y_pred)]
        ])
    
    print('On the dataset ', dataset, 'For the model ', mdl_name)
    print('The Confusion Matrix is:')
    print(confusion_matrix(y_test, y_pred))
    print('Precision is {:.4f}%'.format(100*precision(y_test, y_pred)))
    print('Recall is {:.4f}%'.format(100*recall(y_test, y_pred)))

    return y_test, y_pred, X_test

def main():
    ## 实验A
    ## 算法评估，将三个数据集传递给clf_func函数
    # clf_func(x1, y1, 'circles')
    # clf_func(x2, y2, 'moons')
    # clf_func(x3, y3, 'blobs')
    # plot_func()

    # fine_tune(x1, y1, 'circles')
    # fine_tune(x2, y2, 'moons')
    # fine_tune(x3, y3, 'blobs')

    ## 实验B
    # lr = LR()
    # conf_matrix(x1, y1, lr, 'Logistic Regression', 'circles')
    # conf_matrix(x2, y2, lr, 'Logistic Regression', 'moons')

    # dtc = DTC()
    # conf_matrix(x1, y1, dtc, 'Decision Tree Classifier', 'circles')
    # conf_matrix(x2, y2, dtc, 'Decision Tree Classifier', 'moons')

    rfc = RFC()
    conf_matrix(x1, y1, rfc, 'Random Forest Classifier', 'circles')
    conf_matrix(x2, y2, rfc, 'Random Forest Classifier', 'moons')

if __name__ == '__main__':
    main()