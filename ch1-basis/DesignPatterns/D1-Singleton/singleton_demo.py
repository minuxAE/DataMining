"""
单例设计模式：
- 确保类有且只有一个对象被创建
- 为对象提供一个访问点，以使得程序可以全局访问对象
- 控制共享资源的并行访问

构造函数私有化，创建一个静态方法完成对象初始化
"""

# 单例设计模式
# 只允许Singleton类生成一个实例
# 如果有实例了，会重复提供同一个对象

class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls) # 通过覆盖__new__方法来控制对象的创建
        return cls.instance
    
s = Singleton()
print('Object created', s)

s1 = Singleton()
print('Object created', s1)