#!/usr/bin/env python
# coding: utf-8

# ### 数学基础(2)
# 计算标准正态分布的概率质量，对于距离均值k个标准差的概率质量，考察$k=1,2,3$.

# In[1]:


import numpy as np
from scipy.special import erf
from scipy.integrate import quad

def PM_Gauss(k):
    def f(x):
        return np.exp(-x**2)*2/np.sqrt(np.pi)
    
    pm1,_ = quad(f, 0, k/np.sqrt(2)) # 积分方法计算概率
    pm2 = erf(k/np.sqrt(2)) # 直接用erf函数得到概率
    
    return pm1, pm2

for k in range(1, 4):
    pm1, pm2 = PM_Gauss(k)
    print('k={}, Integration method is {:.4f}, ERF method is {:.4f}'.format(k, pm1, pm2))


# ### 编程案例：iris数据

# In[6]:


mu = np.array([5.843, 3.054]).T # 样本均值
Sig = np.array([
    [0.681, -0.039],
    [-0.039, 0.187]
]) # 样本协方差矩阵

# 考虑一个样本点x2
x2 = np.array([6.9, 3.1]).T

## 计算 x2 - mu
d = x2-mu
## 计算马氏距离
Md = d.T@np.linalg.inv(Sig)@d
print('Maha distance is {:.4f}'.format(Md))

## 计算欧式距离的平方
Ed = np.linalg.norm(x2-mu)**2
print('Euc distance is {:.4f}'.format(Ed))

## 特征值分解
e_vals, e_vecs = np.linalg.eig(Sig)
print('Eigen Values', e_vals)
print('Eigen Vectors', e_vecs)


# In[12]:


## 在特征向量的新坐标系下的协方差矩阵为
Lam = np.array([
    [0.684, 0],
    [0, 0.184]
])

## 计算新坐标系相对于原始坐标系的旋转角度
e1 = np.array([1, 0])
u1 = np.array([0.99693605, 0.07822095])
c_theta = e1.T@u1
theta = np.arccos(c_theta)*180/np.pi
theta # 表示新的坐标系相对于原始坐标系旋转了4.5度


# ### Apriori算法案例
# 使用mlxtend库，或者手写Apriori算法进行求解

# In[13]:


from itertools import combinations

# 事务 （案例中的数据）
transactions = [
    ['A', 'C', 'D'],
    ['B', 'C', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'E']
]

# 最小支持度为0.5
min_support = 0.5

# 找到所有的不重复项
def get_unique_items(transactions):
    uniq_items = set()
    for trans in transactions:
        uniq_items.update(trans)
    return list(uniq_items) # 转为列表返回

# 产生指定长度的频繁项集
def get_frequent_itemsets(transactions, uniq_items, length, min_support):
    itemsets = list(combinations(uniq_items, length)) # 根据指定长度进行两两组合
    item_cnt = {itemset: 0 for itemset in itemsets} # 初始化项集
    
    # 记录项集在事务中出现的次数
    for trans in transactions:
        for itms in itemsets:
            if set(itms).issubset(trans): # 如果是子集，则计数+1
                item_cnt[itms] += 1
                
    # 计算支持度
    len_trans = len(transactions) # 得到 |D|, 事务的长度
    freq_itms = {} # 记录频繁项集
    for itms, cnt in item_cnt.items():
        support = cnt / len_trans
        if support >= min_support:
            freq_itms[itms] = support
    
    return freq_itms

# Aporiori
def apriori(transactions, min_support):
    uniq_items = get_unique_items(transactions)
    
    # 初始化, 找频繁1-项集
    curr_l = get_frequent_itemsets(transactions, uniq_items, 1, min_support)
    k = 2
    all_l = [] # 记录所有频繁项集
    all_l.append(curr_l)
    
    while curr_l != {}: # 如果上一层的频繁项集不为空，则开始组合并筛选出下一层的频繁项集
        curr_itemsets = list(curr_l.keys())
        curr_uniq_items = set(item for itemset in curr_itemsets for item in itemset) # 上一层的频繁项集
        curr_l = get_frequent_itemsets(transactions, curr_uniq_items, k, min_support)
        if curr_l:
            all_l.append(curr_l)
        k+=1
    return all_l
    
# 执行算法
freq_itemsets = apriori(transactions, min_support)

for k_items in freq_itemsets:
    for itemset, support in k_items.items():
        print('Itemset: {}, Support: {}'.format(itemset, support))


# In[23]:


# 调用mlxtend库实现
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder as TE
from mlxtend.frequent_patterns import apriori

ts = [
    ['A', 'C', 'D'],
    ['B', 'C', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'E']
]

te = TE()
te_array = te.fit(ts).transform(ts)
df = pd.DataFrame(te_array, columns=te.columns_) # 对事务进行编码
df


# In[25]:


apriori(df, min_support=0.5, use_colnames=True)


# ### FP-Growth算法案例

# In[4]:


# 调用mlxtend库，使用FP-Growth算法
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder as TE

ts = [
    ['A', 'C', 'D'],
    ['B', 'C', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'E']
]

te = TE()
te_array = te.fit(ts).transform(ts) # 编码转换
df = pd.DataFrame(te_array, columns=te.columns_)
df


# In[6]:


from mlxtend.frequent_patterns import fpgrowth
fpgrowth(df, min_support=0.5, use_colnames=True)


# In[8]:


# 对比 apriori 算法和 fpgrowth 算法的速度
from mlxtend.frequent_patterns import apriori
get_ipython().run_line_magic('timeit', '-n 100 -r 10 apriori(df, min_support=0.5)')


# In[9]:


get_ipython().run_line_magic('timeit', '-n 100 -r 10 fpgrowth(df, min_support=0.5)')


# In[ ]:





# In[ ]:




