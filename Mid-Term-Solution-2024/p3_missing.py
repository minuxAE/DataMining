import numpy as np
from sklearn.impute import SimpleImputer as SI

"""
生成带有缺失值的数据表
"""
data = np.array([
    [3, 5, np.nan, 7, 10],
    [2, np.nan, 8, 7, 10],
    [np.nan, np.nan, 9, 9, 8],
    [4, 5, 6, 7, 8],
    [5, 8, 9, 10, np.nan]
])

print(data)

"""
插值策略1: 使用属性均值进行插入
"""
def imp_method():
    imp_mean = SI(strategy='mean')
    data_mean = imp_mean.fit_transform(data)
    print(data_mean)

    """
    插值策略2：使用中位数进行插入
    """
    imp_med = SI(strategy='median')
    data_med = imp_med.fit_transform(data)
    print(data_med)

    """
    插值策略3：众数插入
    """
    imp_mode = SI(strategy='most_frequent')
    data_mode = imp_mode.fit_transform(data)
    print(data_mode)

def interpolate_method():
    pass
