"""
观察者模式的主要目标：
1. 定义了对象之间的一对多的依赖关系，从而使得一个对象中的任何改变都会通知到其他依赖对象
2. 封装了主题的核心组件

观察者模式可以用于以下场景:
1. 在分布式系统中实现事件服务
2. 用于新闻机构的框架
3. 股票市场
"""

class Subject:
    def __init__(self) -> None:
        self.__observers = []

    def register(self, observer):
        self.__observers.append(observer)

    def notifyAll(self, *args, **kwargs):
        for observer in self.__observers:
            observer.notify(self, *args, **kwargs)

class Observer1:
    def __init__(self, subject) -> None:
        subject.register(self)

    def notify(self, subject, *args):
        print(type(self).__name__, ':: Got', args, 'From', subject)

class Observer2:
    def __init__(self, subject) -> None:
        subject.register(self)

    def notify(self, subject, *args):
        print(type(self).__name__, ':: Got', args, 'From', subject)

subject = Subject()
obs1 = Observer1(subject)
obs2 = Observer2(subject)
subject.notifyAll('notification')