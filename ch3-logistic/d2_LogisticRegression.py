import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import Lasso

def read_data(path='bankloan.xlsx'):
    try:
        bank_data = pd.read_excel(path, index_col=None) # 不需要指定索引列
        X = bank_data.iloc[:, :8]
        y = bank_data.iloc[:, -1]
        return X, y
    except:
        bank_data = pd.read_csv(path)
        return bank_data
    
def screen(X, y):
    selector = SelectKBest(chi2, k=4) # 选择4个特征
    selector.fit_transform(X, y)
    cols = X.columns[selector.get_support(indices=True)]
    print(cols)
    return cols
    
def test(X, y):
    lr = LR(solver='liblinear')
    lr.fit(X, y) # 全样本进行拟合
    print('Logistic Model Accuracy is {:.2f}%'.format(100*lr.score(X, y)))

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_pred = lr.predict(X_test)

    # 图像绘制
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)
    ## 散点图
    plt.subplot(211)
    plt.scatter(range(len(X_test)), y_test+0.5, c='g', s=2, label='Test')
    plt.scatter(range(len(X_test)), y_pred, c='r', s=2, label='Pred')
    plt.title('Predctive Results')
    plt.yticks([0, 1], ['Not Default', 'Default'])
    plt.legend()
    plt.ylim([-0.5, 2.5])

    ## 小提琴图
    plt.subplot(212)
    data = pd.concat([
        pd.DataFrame(y_pred, columns=['pred']),
        pd.DataFrame(y_test.values, columns=['test'])
    ], axis=1).stack().reset_index() # 数据合并后进行轴旋转（透视图），将列转为行index
    
    data = data.drop(columns = data.columns[0]) # 去除level 0中的数据
    data = data.rename(columns={
        data.columns[0] : 'labels',
        data.columns[1] : 'value'
    })
    data['x-axis']=1
    plt.title('Predictive Results')
    sns.violinplot(data=data, x='x-axis', y='value', split=True, hue='labels')
    plt.yticks([0, 1], ['Not Default', 'Default'])
    plt.legend()
    plt.show()
    return lr

def main():
    X, y = read_data()
    cols = screen(X, y)
    X = X[cols].values
    test(X, y)

if __name__ == '__main__':
    main()