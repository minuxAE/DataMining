"""
状态模式是一种行为设计模式，被称为状态模式对象
一个对象可以基于其内部状态封装多个行为

状态设计模式在3个主要参与者的协助下工作

State: 封装对象行为的接口
ConcreteState: 实现State接口的子类
Context: 客户感兴趣的接口
"""
from abc import abstractmethod, ABCMeta

class State(metaclass=ABCMeta):
    @abstractmethod
    def Handle(self):
        pass

class ConcreteStateB(State):
    def Handle(self):
        print('ConcreteStateB')

class ConcreteStateA(State):
    def Handle(self):
        print('ConcreteStateA')

class Context(State):
    def __init__(self) -> None:
        self.state=None

    def getState(self):
        return self.state
    
    def setState(self, state):
        self.state = state

    def Handle(self):
        self.state.Handle()

context = Context()
stateA = ConcreteStateA()
stateB = ConcreteStateB()

context.setState(stateA)
context.Handle()