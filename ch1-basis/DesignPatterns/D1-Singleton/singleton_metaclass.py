from typing import Any


class MyInt(type):
    def __call__(cls, *args: Any, **kwds: Any) -> Any: # 对于已经存在的类，需要创建对象时，将调用特殊方法__call__
        print('This is my int', args)
        print('Now do whatever you want with these objects...')
        return type.__call__(cls, *args, **kwds)

# 元类控制着对象的实例化
class int(metaclass=MyInt):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

i = int(4, 5)

# 通过该控制权，可以用于创建单例
class MetaSingleton(type):
    _instances = {}
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwds)
        return cls._instances[cls]
    
class Logger(metaclass = MetaSingleton):
    pass

logger1 = Logger()
logger2 = Logger()


print(logger1, logger2)
