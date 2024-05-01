import numpy as np
from sklearn.datasets import load_digits

# 加载手写数据集
digits = load_digits()
X = digits.data # 1797 * 64
y = digits.target.copy() # 1797

# 转变为二分类数据集，将标签中为9的设置为正例（1），其余的设置为负例（0）
y[digits.target==9] = 1
y[digits.target!=9] = 0

# 使用Logistic Regression进行二分类拟合
from sklearn.linear_model import LogisticRegression as LR
from solution import conf_matrix

lr = LR(max_iter=1024)
y_test, y_pred, X_test = conf_matrix(X, y, lr, dataset='Digits') # 使用自制Confusion Matrix

from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import precision_score as PS
from sklearn.metrics import recall_score as RS
print(CM(y_test, y_pred)) # sklearn接口
print('The precision is {:.4f}%'.format(100*PS(y_test, y_pred)))
print('The recall is {:.4f}%'.format(100*RS(y_test, y_pred)))

# 查看decision_function
# print(lr.decision_function(X_test))

# 设置阈值T=5
T=5
y_pred_T1 = np.array(lr.decision_function(X_test) > T, dtype='int')
print(CM(y_test, y_pred_T1))
print('The precision is {:.4f}%'.format(100*PS(y_test, y_pred_T1)))
print('The recall is {:.4f}%'.format(100*RS(y_test, y_pred_T1)))

# Precision-Recall曲线绘制
import matplotlib.pyplot as plt

def precision_recall():
    decision_scores = lr.decision_function(X_test)
    precisions, recalls = [], []
    Ts = np.arange(np.min(decision_scores), np.max(decision_scores))
    
    for T in Ts:
        y_pred = np.array(decision_scores >= T, dtype='int')
        precisions.append(PS(y_test, y_pred))
        recalls.append(RS(y_test, y_pred))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axes[0].plot(Ts, precisions, 'r-', lw=1, label='precision')
    axes[0].plot(Ts, recalls, 'b-', lw=1, label='recall')
    axes[0].scatter(0, precisions[np.where(abs(Ts)<0.5)[0][0]], marker='s', color='k', s=10, label='T=0')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Indicators')
    axes[0].legend(loc='best')

    axes[1].plot(precisions, recalls, color='b', lw=1)
    axes[1].set_xlabel('Precision')
    axes[1].set_ylabel('Recall')

    plt.show()

from sklearn.metrics import precision_recall_curve as PRC
def sklearn_precision_recall():
    decision_scores = lr.decision_function(X_test)
    # 末端点对应的precision=1.0, recall=0 (即将所有的样本都判为负例)，没有对应的T值
    precisions, recalls, Ts = PRC(y_test, decision_scores)
    precisions = np.delete(precisions, -1)
    recalls = np.delete(recalls, -1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axes[0].plot(Ts, precisions, 'r-', lw=1, label='precision')
    axes[0].plot(Ts, recalls, 'b-', lw=1, label='recall')
    axes[0].scatter(0, precisions[np.where(abs(Ts)<0.5)[0][0]], marker='s', color='k', s=10, label='T=0')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Indicators')

    axes[1].plot(precisions, recalls, color='b', lw=1)
    axes[1].set_xlabel('Precision')
    axes[1].set_ylabel('Recall')
    plt.show()

"""
ROC曲线的绘制
"""
def myROC(y_true, y_pred):
    def TN(y_true, y_pred):
        return np.sum((y_true == 0) & (y_pred == 0))
    
    def FP(y_true, y_pred):
        return np.sum((y_true == 0) & (y_pred == 1))
    
    def FN(y_true, y_pred):
        return np.sum((y_true == 1) & (y_pred == 0))
    
    def TP(y_true, y_pred):
        return np.sum((y_true == 1) & (y_pred == 1))

    def TPR(y_true, y_pred):
        tp = TP(y_true, y_pred)
        fn = FN(y_true, y_pred)
        try:
            return tp/(tp+fn)
        except:
            return 0.0
        
    def FPR(y_true, y_pred):
        fp = FP(y_true, y_pred)
        fn = TN(y_true, y_pred)
        try:
            return fp/(fp+fn)
        except:
            return 0.0

    decision_scores = lr.decision_function(X_test)

    tprs, fprs = [], []
    Ts = np.arange(min(decision_scores), max(decision_scores))

    for T in Ts:
        y_pred = np.array(decision_scores >= T, dtype='int')
        tprs.append(TPR(y_test, y_pred))
        fprs.append(FPR(y_test, y_pred))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axes[0].plot(Ts, tprs, 'r-', lw=1, label='TPR')
    axes[0].plot(Ts, fprs, 'b-', lw=1, label='FPR')
    axes[0].legend(loc='best')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Indicators')

    axes[1].plot(fprs, tprs)
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')

    plt.show()


from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import auc
def sklearnROC(y_true, y_pred):
    decision_scores = lr.decision_function(X_test)

    fprs, tprs, Ts = ROC(y_true, decision_scores)
    print('AUC is {:.4f}'.format(auc(fprs, tprs)))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axes[0].plot(Ts, tprs, 'r-', lw=1, label='TPR')
    axes[0].plot(Ts, fprs, 'b-', lw=1, label='FPR')
    axes[0].legend(loc='best')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Indicators')

    axes[1].plot(fprs, tprs)
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')

    plt.show()

def main():
    # precision_recall()
    # sklearn_precision_recall()
    # myROC(y_test, y_pred)
    sklearnROC(y_test, y_pred)

if __name__ == '__main__':
    main()