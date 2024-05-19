"""
多项式回归
"""
import numpy as np
import matplotlib.pyplot as plt

# s1: 生成非线性数据集 y=0.5x^2+x+2+eps
x = np.random.uniform(-3, 3, size=100) # x轴上取点
y = 0.5*x**2+x+2+np.random.normal(0, 1, size=100)

## 绘图开关
s1_plot = False
s2_plot = False
s3_plot = False
s4_plot = False
s5_plot = False
s6_plot = False

if s1_plot:
    plt.scatter(x, y, s=4)
    plt.show()

# s2: 使用线性回归拟合数据集
from sklearn.linear_model import LinearRegression as LinReg
X = x.reshape(-1, 1) # 转为矩阵
lin_reg = LinReg()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X) # 结果为欠拟合（无法描述二阶项）

if s2_plot:
    plt.scatter(x, y, s=4)
    plt.plot(x, y_pred, color='r', lw=1)
    plt.show()

# s3: 添加x^2的特征，再次拟合
X2 = np.hstack([X, X**2])
lin_reg2 = LinReg()
lin_reg2.fit(X2, y)
y_pred2 = lin_reg2.predict(X2)

if s3_plot:
    plt.scatter(x, y, s=4)
    # plt.plot(np.sort(x), y_pred2[np.argsort(x)], color='r', lw=1)
    plt.plot(x, y_pred2)
    plt.show()

# print(lin_reg2.coef_) # [0.9551653  0.47540514] y=0.95x^2+0.475x + 2 

# s4: 构建多项式特征并拟合
from sklearn.preprocessing import PolynomialFeatures as POLYF
poly = POLYF(degree=2) # 构建最高为2阶特征
poly.fit(X)
X2 = poly.transform(X)

lin_reg3 = LinReg()
lin_reg3.fit(X2, y)
y_pred3 = lin_reg3.predict(X2)

if s4_plot:
    plt.scatter(x, y, s=4)
    plt.plot(np.sort(x), y_pred3[np.argsort(x)], color='r', lw=1)
    plt.show()

# s5: 管道拼接操作后拟合+可视化
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SS

poly_reg = Pipeline([
    ('poly', POLYF(degree=2)), # 操作1： 构建多项式特征
    ('standard scaler', SS()), # 操作2： 数据标准化
    ('lin-reg', LinReg()) # 操作3：线性回归
])

poly_reg.fit(X, y)
y_predp = poly_reg.predict(X)

if s5_plot:
    plt.scatter(x, y, s=4)
    plt.plot(np.sort(x), y_predp[np.argsort(x)], color='r', lw=1)
    plt.show()

# s6: 过拟合
## 将多项式特征设定为生成100阶, 再进行拟合
poly_reg100 = Pipeline([
    ('poly', POLYF(degree=100)), # 操作1： 构建多项式特征
    ('standard scaler', SS()), # 操作2： 数据标准化
    ('lin-reg', LinReg()) # 操作3：线性回归
])

poly_reg100.fit(X, y)
y_predp100 = poly_reg100.predict(X)

if s6_plot: # 过拟合状态下，曲线会尽可能地通过每一个样本点
    plt.scatter(x, y, s=4)
    plt.plot(np.sort(x), y_predp100[np.argsort(x)], color='r', lw=1)
    plt.show()

## 对比过拟合模型在训练集和测试上的MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
poly_reg100.fit(X_train, y_train)

print('On the training set, MSE is {:.4f}'.format(MSE(poly_reg100.predict(X_train), y_train))) # 0.335
print('On the testing set, MSE is {:.4f}'.format(MSE(poly_reg100.predict(X_test), y_test))) # 特别大
