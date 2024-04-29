import numpy as np

np.random.seed(2024)
X = np.random.uniform(0, 2, size=(20, 3)) # 样本数量为20, 特征数量为3
Y = np.random.choice(('Low', 'Mid', 'High'), size=20) # 离散型、字符型标签向量

"""
LabelEncoder
"""
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Yt = le.fit_transform(Y) 

print(Y) # ['High' 'High' 'Low' 'High' 'Low' 'High' 'Mid'...
print(Yt) # [0 0 1 0 1 0 2 1 0 2 2 1 1 0 2 1 1 0 0 0]

## 反向转换
output = [0, 1, 0, 1, 2, 0, 2, 0, 2, 1, 1, 2, 0]
de_output = [le.classes_[int(i)] for i in output]
print(de_output) # ['High', 'Low', 'High', 'Low', 'Mid', 'High', 'Mid'...

"""
OneHotEncoder
"""
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(categories='auto')
# 'High' 'High' 'Low' 'High' 'Low' 'High' 'Mid' 'Low' 'High' 'Mid' 'Mid'
Yo = oh.fit_transform(Y.reshape(-1, 1))
# [1. 0. 0.]
# [1. 0. 0.]
# [0. 1. 0.]
print(Yo.todense())