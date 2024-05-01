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

"""
插值策略4：scipy.interpolate
"""

from scipy.interpolate import interp1d
def interpolate_method():
    for j in range(data.shape[1]):
        sub_data = data[:, j]
        indices = np.arange(len(sub_data))
        known_indices = indices[~np.isnan(sub_data)]
        known_values = sub_data[~np.isnan(sub_data)]
        
        # Linear interpolation
        interp_func = interp1d(known_indices, known_values, kind='linear', fill_value='extrapolate')
        filled_values = interp_func(indices)
        data[:, j] = filled_values
    return data


def main():
    filled_data = interpolate_method()
    print(filled_data)

if __name__ == '__main__':
    main()
