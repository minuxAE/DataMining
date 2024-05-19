"""
模型正则化
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures as POLYF
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

import numpy as np
import matplotlib.pyplot as plt

# s1: 生成非线性数据集 y=0.5x^2+x+2+eps
x = np.random.uniform(-3, 3, size=100) # x轴上取点
y = 0.5*x**2+x+2+np.random.normal(0, 1, size=100)
X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# s1: Ridge
def RidgeReg(degree, alpha):
    return Pipeline([
        ('Poly', POLYF(degree)),
        ('standard scaler', SS()),
        ('ridge_reg', Ridge(alpha=alpha))
    ])

def work_ridge():
    ridge_reg = RidgeReg(100, 0.1)
    ridge_reg.fit(X_train, y_train)

    y_pred_r1 = ridge_reg.predict(X_test)

    print('Ridge Reg. MSE is {:.4f}'.format(MSE(y_test, y_pred_r1)))

    ## 拟合曲线绘制
    Xp = np.linspace(-3, 3, 100).reshape(-1, 1)
    yp = ridge_reg.predict(Xp)

    plt.scatter(x, y, s=4)
    plt.plot(Xp, yp, color='r', lw=1)
    plt.axis([-3, 3, -1, 10])
    plt.show()

# s2: LASSO
def LassoReg(degree, alpha):
    return Pipeline([
        ('poly', POLYF(degree=degree)),
        ('standard scaler', SS()),
        ('lasso_reg', Lasso(alpha=alpha))
    ])


def work_lasso():
    lasso_reg = LassoReg(100, 0.1)
    lasso_reg.fit(X_train, y_train)

    y_pred_r2 = lasso_reg.predict(X_test)

    print('Lasso Reg. MSE is {:.4f}'.format(MSE(y_test, y_pred_r2)))

    ## 拟合曲线绘制
    Xp = np.linspace(-3, 3, 100).reshape(-1, 1)
    yp = lasso_reg.predict(Xp)

    plt.scatter(x, y, s=4)
    plt.plot(Xp, yp, color='r', lw=1)
    plt.axis([-3, 3, -1, 10])
    plt.show()


def ESReg(degree, alpha, ratio):
    return Pipeline([
        ('poly', POLYF(degree=degree)),
        ('standard scaler', SS()),
        ('ES_reg', ElasticNet(alpha=alpha, l1_ratio=ratio))
    ])

def work_ES():
    ES_reg = ESReg(100, 0.1, 0.5)
    ES_reg.fit(X_train, y_train)

    y_pred_r3 = ES_reg.predict(X_test)

    print('ES Reg. MSE is {:.4f}'.format(MSE(y_test, y_pred_r3)))

    ## 拟合曲线绘制
    Xp = np.linspace(-3, 3, 100).reshape(-1, 1)
    yp = ES_reg.predict(Xp)

    plt.scatter(x, y, s=4)
    plt.plot(Xp, yp, color='r', lw=1)
    plt.axis([-3, 3, -1, 10])
    plt.show()

work_ES()

