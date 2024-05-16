class Borg:
    __shared_state = {"1":"2"}
    def __init__(self) -> None:
        self.x = 1
        self.__dict__ = self.__shared_state # 存储类所有对象的状态
        pass

b = Borg()
b1 = Borg()
b.x = 4

# b和b1不是相同的实例
print('Object b', b)
print('Object b1', b1)

# 但是他们共享相同的状态
print(b.__dict__)
print(b1.__dict__)

# 实现Borg模式: 通过修改__new__方法

class Borg1(object):
    _shared_state = {}

    def __new__(cls, *args, **kwargs):
        obj = super(Borg1, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state # 状态复制
        return obj

bb = Borg1()
bb1 = Borg1()
# b和b1不是相同的实例
print('Object b', bb)
print('Object b1', bb1)

# 但是他们共享相同的状态
print(bb.__dict__)
print(bb1.__dict__)