from abc import ABCMeta, abstractmethod

class AbstractClass(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def operation1(self):
        pass

    @abstractmethod
    def operation2(self):
        pass

    def template_method(self):
        print('Defining the Algorithm. Operationa1 follows Operation2')
        self.operation1()
        self.operation2()

class ConcreteClass(AbstractClass):
    def operation1(self):
        print('My Concrete Operation1')

    def operation2(self):
        print('Operation2 remains same')

class Client:
    def main(self):
        self.concrete = ConcreteClass()
        self.concrete.template_method()

client = Client()
client.main()