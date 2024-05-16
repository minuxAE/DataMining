"""
Factory有三种变体模式
简单工厂模式：允许接口创建对象，但是不会暴露对象的创建逻辑
工厂方法模式：允许接口创建对象，但是使用哪个类创建对象，由接口确定
抽象工厂模式：抽象工厂是一个能创建一系列相关对象而无序指定/公开具体类的接口。
"""

from abc import ABCMeta, abstractclassmethod

class Animal(metaclass=ABCMeta):
    @abstractclassmethod
    def do_say(self):
        pass

class Dog(Animal):
    def do_say(self):
        print('Bhow Bhow!!')

class Cat(Animal):
    def do_say(self):
        print('Meow Meow!!')

## forest factory defined
class ForestFactory(object):
    def make_sound(self, object_type): # 根据具体创建的对象，调用其方法
        return eval(object_type)().do_say()
    
## client
if __name__ == '__main__':
    ff = ForestFactory()
    animal = input('which animal should make_sound Dog or Cat?')
    ff.make_sound(animal)