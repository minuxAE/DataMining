"""
单例模式的用例之一是进行懒汉式实例化，即在实际使用时才创建对象
"""
class Singleton:
    __instance = None
    def __init__(self) -> None:
        if not Singleton.__instance:
            print('__init__ method is called..')
        else:
            print('Instance already created: ', self.getInstance())

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = Singleton()
        return cls.__instance
    

s = Singleton()
print('Object created', Singleton.getInstance())

s1 = Singleton()