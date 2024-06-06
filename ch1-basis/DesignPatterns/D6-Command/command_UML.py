"""
Command: 生命执行操作的接口
ConcreteCommand: 将一个Receiver对象和一个操作绑定
Client: 创建ConcreteCommand对象并设定其为接收者
Invoker: 要求改ConcreteCommand执行这个请求
Receiver: 如何实施请求相关操作
"""

from abc import ABCMeta, abstractclassmethod

class Command(metaclass=ABCMeta):
    def __init__(self, recv) -> None:
        self.recv = recv

    def execute(self):
        pass

class ConcreteCommand(Command):
    def __init__(self, recv) -> None:
        self.recv = recv

    def execute(self):
        self.recv.action()

class Receiver:
    def action(self):
        print('Receiver Action')

class Invoker:
    def command(self, cmd):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()

if __name__ == '__main__':
    recv = Receiver()
    cmd = ConcreteCommand(recv)
    invoker = Invoker()
    invoker.command(cmd)
    invoker.execute()