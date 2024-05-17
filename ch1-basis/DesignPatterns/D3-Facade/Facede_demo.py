"""
门面设计模式

结构型设计模式：

适配器模式:将一个接口转换程客户希望的另一个接口，它试图根据客户端的需求来匹配不同类的接口
桥接模式：将对象的接口与实现进行解耦，使得两者可以独立
装饰器模式：允许在运行时以动态的方式为对象添加功能，可以通过接口给对象添加属性

门面设计模式：

为子系统中的一组接口提供了一个统一的接口，并定义了一个高级接口来帮助客户端通过更加简单的方式使用子系统
本质上是对底层子系统进行组合，并实现了对多个客户端的解耦
"""

# 经理（门面）
class EventManager(object):
    def __init__(self) -> None:
        print('Event Manager:: Talk to the Folks')

    def arrange(self):
        self.hotelier = Hotelier()
        self.hotelier.bookHotel()

        self.florist = Florist()
        self.florist.setFlowerRequirement()

        self.caterer = Caterer()
        self.caterer.setCuisine()

        self.musician = Musician()
        self.musician.setMusicType()

# 子系统1：预定酒店
class Hotelier(object):
    def __init__(self) -> None:
        print('Arranging the Hotel for Manager ?')

    def __isAvailable(self):
        print('Is the Hotel free for the event on given day?')
        return True
    
    def bookHotel(self):
        if self.__isAvailable():
            print('Registered the Booking...\n\n')
    

# 子系统2：花卉装饰
class Florist(object):
    def __init__(self) -> None:
        print('Flower Decorations for the Event ?')

    def setFlowerRequirement(self):
        print('Carnations, Roses and Lilies would be used for Decorations\n\n')

# 子系统3：宴席餐饮
class Caterer(object):
    def __init__(self) -> None:
        print('Food Arrangements for the Event')

    def setCuisine(self):
        print('Chinese & Continental Cuisine to be served\n\n')

# 子系统4：音乐安排
class Musician(object):
    def __init__(self) -> None:
        print('Musical Arrangements for the Marriage')

    def setMusicType(self):
        print('Jazz and Classical will be played\n\n')

# 客户端-经理：婚姻筹划
class You(object):
    def __init__(self) -> None:
        print('You::Whoa! Marriage Arrangement...')

    def askEventManager(self):
        print("You::Let's Contact the Event Manager")
        em = EventManager()
        em.arrange()

    def __del__(self):
        print('You:: Thanks to Event Manager, all preparations done! Phew')

you = You()
you.askEventManager()

"""
最少知识原则：
减少对象之间的交互

最少知识原则和迪米特法则是一致的，即松耦合理论
"""