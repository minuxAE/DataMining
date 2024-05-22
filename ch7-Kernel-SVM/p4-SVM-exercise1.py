"""
例题求解
Data Mining and Analysis: Fundamental Concepts and Algorithms

21.7
"""
import numpy as np


# Q1
def Q1():
    ## (a) 确定h1和h2的方程
    p1 = np.array([6, 0])
    p2 = np.array([1, 10])
    k1 = (p2[1] - p1[1]) / (p2[0] - p1[0]) # -2
    _b1 = p1[1] - k1 * p1[0] # 12

    # h1: 2x1 + x2 - 12 = 0

    n1 = np.array([2, 0])
    n2 = np.array([5, 5])
    k2 = (n2[1] - n1[1]) / (n2[0] - n1[0]) # 5/3
    _b2 = n1[1] - k2 * n1[0] # -10/3
    
    # h2: 5x1 - 3x2 - 10 = 0

    ## (c) 
    w1 = np.array([2, 1])
    w2 = np.array([5, -3])
    b1 = -12
    b2 = -10

    def dist(x, w, b):
        return (w@x + b) / np.linalg.norm(w) 
    
    ## 考察h1, 支持向量为(2, 6)和(6, 2)
    a1 = np.array([2, 6])
    a2 = np.array([6, 2])
    
    da1 = dist(a1, w1, b1)
    da2 = dist(a2, w1, b1)

    tot_h1 = np.abs(da1-da2) # 1.7889
    print('Total Margin of h1 is {:.4f}'.format(tot_h1))

    ## 考察h2, 支持向量为(3, 4)和(7, 6)
    e1 = np.array([3, 4])
    e2 = np.array([7, 6])

    de1 = dist(e1, w2, b2)
    de2 = dist(e2, w2, b2)

    tot_h2 = np.abs(de1-de2) # 2.4010
    print('Total Margin of h2 is {:.4f}'.format(tot_h2))

    ## h2的区分度要超过h1



# Q2
def Q2():
    X1 = [4, 4, 1, 2.5, 4.9, 1.9, 3.5, 0.5, 2, 4.5]
    X2 = [2.9, 4, 2.5, 1, 4.5, 1.9, 4, 1.5, 2.1, 2.5]
    y = np.array([1, 1, -1, -1, 1, -1, 1, -1, -1, 1])
    alpha = np.array([0.414, 0, 0, 0.018, 0, 0, 0.018, 0, 0.414, 0])

    x = np.array([*zip(X1, X2)])
    ## (a) 计算SVM的超平面方程
    ## 计算权重
    w = np.zeros(2) # w=(0.846, 0.385)
    for i in range(len(x)):
        if alpha[i]>0:
            w += alpha[i]*y[i]*x[i]


    ## 计算偏置
    b = y[0] - w@x[0] # b=-3.50
    
    ## Hyper Plane: h(x)=0.846x1 + 0.385x2 - 3.50

    ## (b) 计算点x6到超平面的距离，判断它是否在分类器间隔内
    d = w@x[5]+b # -1.16
    print('The distance of x6 to Hyper Plane is {:.4f}'.format(d)) 
    if d > 1 or d < -1: print('The point is outside the margin') 
    else: print('The point is inside the margin')

    ## (c) 对点(3, 3)进行分类
    z = np.array([3, 3])
    hz = w@z+b # 0.19
    if hz > 0: print('The class is +1')
    else: print('The class is -1')

Q1()


