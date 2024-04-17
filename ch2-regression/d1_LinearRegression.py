# 线性回归案例
# 数据：AAPL, 苹果公司股价（2013-2023）
import pandas as pd
AAPL = pd.read_csv('AAPL.csv', index_col=0)


# 构建特征
from sklearn import preprocessing as PP
## 特征1: HL
AAPL['HL'] = (AAPL['High'] - AAPL['Low']) / AAPL['Low']
## 特征2：CO
AAPL['CO'] = (AAPL['Close'] - AAPL['Open'])/ AAPL['Open']
# print(AAPL.head())

# 预测时间节点的数量
Npred = 100
y_true = AAPL['Adj Close'][-Npred:] # 取最后100个点作为测试集
AAPL['Adj Close'] = AAPL['Adj Close'].shift(-Npred) # 其余点依次上移100个节点, 使得回归模型具备预测能力

# 分离训练集和测试集，特征标准化处理
X = AAPL[['HL', 'CO', 'Open', 'High', 'Low', 'Close', 'Volume']] # 提取特征矩阵
X = pd.DataFrame(PP.scale(X)) # 标准化

X_train,X_test = X.iloc[:-Npred, :], X.iloc[-Npred:, :]
y_train = AAPL['Adj Close'][:-Npred]

# 建立回归模型
from sklearn.linear_model import LinearRegression as LinReg

lin_reg = LinReg().fit(X_train, y_train) # 拟合

# 查看模型的R2
print('On training set, R squared is {:.4f}'.format(lin_reg.score(X_train, y_train))) # 0.7185
print('On testing set, R squared is {:.4f}'.format(lin_reg.score(X_test, y_true))) # -21.4567 # 说明模型在测试集上处于失效状态

# 计算adjusted R2
Ntrain, Ntest = X_train.shape[0], X_test.shape[0]
TrainR2, TestR2 = lin_reg.score(X_train, y_train), lin_reg.score(X_test, y_true)
d = 7 # 特征的数量

Train_adj_R2 = 1-(1-TrainR2)*(Ntrain-1)/(Ntrain-d-1)
Test_adj_R2 = 1-(1-TestR2)*(Ntest-1)/(Ntest-d-1)

print('On training set, Adjusted R squared is {:.4f}'.format(Train_adj_R2)) # 0.7139
print('On testing set, Adjusted R squared is {:.4f}'.format(Test_adj_R2)) # -23.1654

# 绘制图像
"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=1, nrows=2)

# Training set
axes[0].plot(lin_reg.predict(X_train), 'r-', lw=0.5, label='Predicted Adj. Close')
axes[0].plot(y_train.values, 'b-', lw=0.5, label='True Adj. Close')
axes[0].legend(loc='best')
axes[0].set_title('On Training Set')

axes[1].plot(lin_reg.predict(X_test), 'r-', lw=0.5, label='Predicted Adj. Close')
axes[1].plot(y_true.values, 'b-', lw=0.5, label='True Adj. Close')
axes[1].legend(loc='best')
axes[1].set_title('On Testing Set')

plt.subplots_adjust(bottom=0.01, top=0.99, wspace=0.2)
plt.show()
"""

# 封装评估函数
import numpy as np

def regEvaluation(y_pred, y_true, d=-1):
    assert len(y_pred) > 0 and len(y_true) > 0, 'Length of labels should be positive'
    assert len(y_pred) == len(y_true), 'Length of labels should be equal'
    assert sum(np.isnan(y_pred)) + sum(np.isnan(y_true)) == 0, 'NaN should not in labels'

    if d==-1: d=3 # 默认特征数量为3个

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true) # 转为np.array格式
    res = y_true - y_pred # ei
    RSS = np.sum((y_true - y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    Ns = len(y_pred)

    metrics = {
        'ME': np.sum(res) / Ns,
        'RMSE': np.sqrt(1/Ns*np.sum(res**2)),
        'MAE': np.sum(np.abs(res)) / Ns,
        'MPE': 100 * np.sum(res / y_true) / Ns,
        'MAPE': 100 * np.sum(np.abs(res / y_true))/ Ns,
        'R2': 1-RSS/TSS,
        'Adj R2': 1-(RSS/TSS)*(Ns-1)/(Ns-d-1)
    }

    # 数据显示格式
    fmt = '{{:<{}}}  :   {{:.4f}}'.format(max(len(key) for key in metrics.keys()))

    for key in metrics.keys():
        print(fmt.format(key, metrics[key]))

y_fit = lin_reg.predict(X_train) # 在训练集上的拟合值
print('-------------------------------使用Numpy实现指标计算-------------------------')
regEvaluation(y_fit, y_train, d=7)

print('-------------------------------调用sklearn.metrics模块-------------------------')
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

sk_metrics = {
    'RMSE': np.sqrt(MSE(y_train, y_fit)),
    'MAE': MAE(y_train, y_fit),
    'MAPE': MAPE(y_train, y_fit),
    'R2': R2(y_train, y_fit)
}

sk_fmt = '{{:<{}}}  :   {{:.4f}}'.format(max(len(key) for key in sk_metrics.keys()))
for key in sk_metrics.keys():
    print(sk_fmt.format(key, sk_metrics[key]))
