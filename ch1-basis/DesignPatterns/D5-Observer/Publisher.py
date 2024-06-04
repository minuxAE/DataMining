"""
NewsPublisher提供了一个供订阅用户使用的接口
attach() 方法给观察者来注册NewsPublisherObserver
detach() 用于注销
subscriber() 返回已经使用Subject注册的所有订户的列表
notifySubscriber() 遍历已经向NewsPublisher注册的所有用户
"""

class NewsPublisher:
    def __init__(self) -> None:
        self.__subscribers = []
        self.__latestNews = None

    def attach(self, subscriber):
        self.__subscribers.append(subscriber)

    def detach(self):
        return self.__subscribers.pop()
    
    def subscribers(self):
        return [type(x).__name__ for x in self.__subscribers]
    
    def notifySubscribers(self):
        for sub in self.__subscribers:
            sub.update()

    def addNews(self, news):
        self.__latestNews = news

    def getNews(self):
        return "Got News:", self.__latestNews
    

from abc import ABCMeta, abstractclassmethod

class Subscriber(metaclass=ABCMeta):
    @abstractclassmethod
    def update(self):
        pass

class SMSSSubscriber:
    def __init__(self, publisher) -> None:
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())

class EmailSubscriber:
    def __init__(self, publisher) -> None:
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())

class AnyOtherSubscriber:
    def __init__(self, publisher) -> None:
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.getNews())


"""
观察者模式的优点和缺点
优点：
1. 它使得彼此交互的对象之间保持松耦合
2. 无需对观察者进行任何修改的情况下发送数据到其他对象
3. 随时添加/删除观察者

缺点：
1. 观察者的接口必须由观察者实现
2. 实现不当会增加复杂性
3. 可能会出现竞争条件或者不一致性
"""


if __name__ == '__main__':
    news_publisher = NewsPublisher()
    for SUB in [SMSSSubscriber, EmailSubscriber, AnyOtherSubscriber]:
        SUB(news_publisher)
    print('\nSubscribers:', news_publisher.subscribers())
    
    news_publisher.addNews('Hello World!')
    news_publisher.notifySubscribers()

    print('\nDetached:', type(news_publisher.detach()).__name__)
    print('\nSubscribers:', news_publisher.subscribers())

    news_publisher.addNews('My Second News!')
    news_publisher.notifySubscribers()
    