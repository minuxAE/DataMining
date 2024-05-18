"""
超平面 计算案例
"""
import numpy as np

# hyper-plane: w1x1+w2x2+b=0
# 考虑超平面上的两个点p1和p2
p1 = np.array([4, 0])
p2 = np.array([2, 5])
# 计算斜率 -w1/w2
k = (p2[1]-p1[1]) / (p2[0] - p1[0]) # -w1 / w2 = -5/2

# 得到权重向量 w = (5, 2)
w = np.array([5, 2])
# 计算偏置 b, 代入超平面上的点进行计算
b = -(p1@w) # b=-20

# 计算原点到超平面的距离
## 通过超平面方程判断点的性质
def h(x):
    return x@w+b

def label(x):
    return 1 if h(x) > 0 else -1

## 距离计算
def dist(x):
    r = h(x) / np.linalg.norm(w)
    return label(x) * r

# 计算原点到超平面的距离
def work():
    o = np.zeros(2)
    print(dist(o)) # 3.71

if __name__ == '__main__':
    work()



