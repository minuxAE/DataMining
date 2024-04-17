from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as DTR
import numpy as np

data = pd.read_excel('Concrete_Data.xlsx', header=0) # 如果提示缺少openpyxl，用pip install openpyxl安装

from sklearn.model_selection import train_test_split
X = data.iloc[:, 0:8] # 特征矩阵
Y = data.iloc[:, -1] # 标签
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=200, random_state=2024)
dtr = DTR(criterion='squared_error', max_depth=11, random_state=2024).fit(X_train, y_train)
print(dtr)
print(dtr.score(X_train, y_train)) # 训练集上的误差(MSE) 0.987
print(dtr.score(X_test, y_test)) # 测试集上的误差(MSE) # 0.914

# 使用柱状图绘制真实值和DTR的预测值
fig = plt.figure(figsize=(8, 7))
N = len(y_test)
plt.subplot(211)
plt.title('True Value')
plt.bar(range(1, 1+N), y_test, color='blue')

plt.subplot(212)
plt.title('Predictive Value')
plt.bar(range(1, 1+N), dtr.predict(X_test), color='orange')

plt.show()