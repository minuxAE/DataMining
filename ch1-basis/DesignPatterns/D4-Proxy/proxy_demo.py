"""
代理是寻求方和中介方的中介系统

客户端在向网站发出请求，首先链接代理服务器，然后再请求网页之类的信息

代理服务器会在内部评估请求，将其发送到适当的服务器，收到响应后，会将响应发送回客户端

代理服务器可以封装请求、保护隐私、适合在分布式架构中运行

代理模式的主要目的是为其他对象提供一个代理者或者占位符，从而控制实际对象的访问
"""

class Actor(object):
    def __init__(self) -> None:
        self.isBusy = False

    def occupied(self):
        self.isBusy = True
        print(type(self).__name__, 'is occupied with current movie')

    def available(self):
        self.isBusy = False
        print(type(self).__name__, 'is free for the movie')

    def getStatus(self):
        return self.isBusy


class Agent(object):
    def __init__(self) -> None:
        self.principal = None

    def work(self):
        self.actor = Actor()
        if self.actor.getStatus():
            self.actor.occupied()
        else:
            self.actor.available()

if __name__ == '__main__':
    r = Agent()
    r.work()