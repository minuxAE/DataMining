import numpy as np

"""
S1: 实现train_test_split函数
"""
def my_train_test_split(X, y, test_size=0.2, random_state=2024):
    assert X.shape[0] == y.shape[0], 'size should be matched'
    assert 0 <= test_size <= 1.0, 'size should be valid'

    if random_state: np.random.seed(random_state)

    shuffle_index = np.random.permutation(len(X)) # 将数据集索引随机排列

    t_size = int(len(X) * test_size)
    index_test = shuffle_index[:t_size] # 测试集数据的索引
    index_train = shuffle_index[t_size:] # 训练集数据的索引

    X_train= X[index_train]
    y_train = y[index_train]

    X_test = X[index_test]
    y_test = y[index_test]

    return X_train, X_test, y_train, y_test

"""
S2:ROC曲线
"""
from sklearn.datasets import make_classification as MC
from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import auc as AUC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LogReg
import matplotlib.pyplot as plt

def myROC():
    X, Y = MC(n_samples=1000, n_features=10, n_classes=2, random_state=2024)
    X_train, X_test, y_train, y_test = my_train_test_split(X, Y, test_size=0.3, random_state=2024)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # 测试自定义的train_test_split函数

    # 模型：决策树
    dtc = DTC(random_state=2024).fit(X_train, y_train)
    y_pred = dtc.predict_proba(X_test)[:, 1]# 输出标签的概率, 0列对应负例（0）的概率，1列对应正例（1）的概率

    # ROC：决策树
    fpr, tpr, Ts = ROC(y_test, y_pred)
    auc_dtc = AUC(fpr, tpr)
    print('AUC of DTC is {:.4f}'.format(auc_dtc)) # 0.9573

    # 模型：Logistic Regression
    ## 加入了L2正则项参数
    log_reg = LogReg(random_state=2024, max_iter=1024, penalty='l2', solver='liblinear', verbose=0).fit(X_train, y_train)
    y_pred1 = log_reg.decision_function(X_test)

    # ROC: Logistic Regression
    fpr1, tpr1, Ts1 = ROC(y_test, y_pred1)
    auc_log_reg = AUC(fpr1, tpr1)
    print('AUC of Log Reg is {:.4f}'.format(auc_log_reg)) # 0.9597

    # 模型：Linear Regression
    lin_reg = LinReg().fit(X_train, y_train)
    y_pred2 = lin_reg.predict(X_test)

    # ROC: Linear Regression
    fpr2, tpr2, Ts2 = ROC(y_test, y_pred2)
    auc_lin_reg = AUC(fpr2, tpr2)
    print('AUC of Lin Reg is {:.4f}'.format(auc_lin_reg)) # 0.9592

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', label='ROC: DTC (AUC={:.2f})'.format(auc_dtc))
    plt.plot(fpr1, tpr1, color='red', label='ROC: Log Reg (AUC={:.2f})'.format(auc_log_reg))
    plt.plot(fpr2, tpr2, color='darkgreen', label='ROC: Lin Reg (AUC={:.2f})'.format(auc_lin_reg))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='-.') # 随机模型的ROC曲线
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.show()

"""
S3:多分类
"""
from sklearn.datasets import load_digits
from sklearn.metrics import precision_score as PS
from sklearn.metrics import recall_score as RS
from sklearn.metrics import confusion_matrix as CM
def multiClassification():
    digits = load_digits()
    X = digits.data # 1797 * 64
    y = digits.target # 1797

    X_train, X_test, y_train, y_test = my_train_test_split(X, y, random_state=2024)

    log_reg = LogReg(max_iter=5096)
    log_reg.fit(X_train, y_train)
    ## 计算多分类指标的加权策略使用macro: 每一个标签都计算，给出均值，不过这种策略没有考虑样本不平衡的因素
    print(PS(log_reg.predict(X_test), y_test, average='macro')) # 0.9666
    print(RS(log_reg.predict(X_test), y_test, average='macro')) # 0.9671

    # 多分类问题的混淆矩阵 (本例中标签有10个分类，分别代表数字0-9)
    y_pred= log_reg.predict(X_test)
    cfm = CM(y_test, y_pred)
    print(cfm) # 对角线元素表示真实标签和预测标签一致的数量

    # 绘制矩阵
    # plt.matshow(cfm, cmap=plt.cm.gray)
    # plt.show()

    # 绘制分类错误矩阵
    row_err_sum = np.sum(cfm, axis=1)
    err_matrix = cfm / row_err_sum

    # 只需要观测分类错误的情况，将分类正确的数据清空，即矩阵对角线置0
    np.fill_diagonal(err_matrix, 0)
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    plt.show() # 可以发现数字8和9最容易发生预测错误
    

"""
主函数
"""
def main():
    # myROC()
    multiClassification()


if __name__ == '__main__':
    main()