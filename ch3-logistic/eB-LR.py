import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2

def read_data(path='bankloan.xlsx'):
    try:
        bank_data = pd.read_excel(path, index_col=None) # 不需要指定索引列
        X = bank_data.iloc[:, :8] # 特征矩阵
        y = bank_data.iloc[:, -1] # 标签
        return X, y
    except:
        print('Error in reading excel file')
        return -1
    
def screen(X, y): # 使用chi2准则筛选出4个最有效的特征进行建模
    selector = SelectKBest(chi2, k=4)
    selector.fit_transform(X, y)
    cols = X.columns[selector.get_support(indices=True)] # 得到选出的4个特征
    print(cols)
    return cols

def test(X, y):
    lr = LR(solver='liblinear') # 建立LR对象，设定求解器为liblinear
    lr.fit(X, y)
    print('Logistic Regression Accuracy: {:.2f}'.format(lr.score(X, y)))

    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 20%的样本作为测试集
    y_pred = lr.predict(X_test)

    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)

    ## 散点图
    plt.subplot(211)
    plt.scatter(range(len(X_test)), y_test+0.5, color='green', s=2, label='Test') # 测试集的真实标签, 上方的点
    plt.scatter(range(len(X_test)), y_pred, color='red', s=2, label='Pred') # 测试集的预测标签
    plt.title('Predictive Results')
    plt.yticks([0, 1], ['Not Default', 'Default'])
    plt.legend()
    plt.ylim([-0.5, 2.5])

    ## 小提琴图: 对称度越高表示预测越准确
    plt.subplot(212)
    ### 合并数据
    data = pd.concat([
        pd.DataFrame(y_pred, columns=['pred']),
        pd.DataFrame(y_test.values, columns=['test'])
    ], axis=1).stack().reset_index() # 合并后进行轴旋转，列转为行，类似Excel中构建Pivot Table

    data = data.drop(columns=data.columns[0]) # 去除level 0 中的数据
    data = data.rename(columns={
        data.columns[0] : 'labels',
        data.columns[1] : 'values'
    }) # 将列名进行重命名

    data['x-axis']=1 # 对称轴

    plt.title('Predictive Results')
    sns.violinplot(data=data, x='x-axis', y='values', split=True, hue='labels')
    plt.yticks([0, 1], ['Not Default', 'Default'])
    plt.legend()
    plt.show()



def main():
    X, y = read_data()
    cols = screen(X, y) # 筛选出的特征有：工龄、地址、负债率、信用卡负债，数据降维
    X = X[cols].values
    test(X, y) # 准确率为81%

if __name__ == '__main__':
    main()

