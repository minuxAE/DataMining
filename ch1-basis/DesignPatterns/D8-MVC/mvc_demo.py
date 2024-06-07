"""
MVC模式将应用程序分为3个基本部分：模型、视图、控制器
将信息的处理和信息的呈现分离开来

MVC的工作机制：模型提供数据和业务逻辑（存储和查询信息），视图负责数据的展示，控制其负责协调模型和视图

模型：声明一个存储和操作数据的类
视图：声明一个类构建用户界面和显示数据
控制器：声明一个链接模型和视图的类
客户端：声明一个类，根据某些操作来获得结果
"""

class Model(object):
    services = {
        'email': {'number': 1000, 'price': 2},
        'sms': {'number': 1000, 'price': 10},
        'voice': {'number': 1000, 'price': 15}
    }

class View(object):
    def list_services(self, services):
        for svc in services:
            print(svc, ' ')

    def list_pricing(self, services):
        for svc in services:
            print('For', Model.services[svc]['number'], svc, 'message you pay $', Model.services[svc]['price'])

class Controller(object):
    def __init__(self) -> None:
        self.model = Model()
        self.view = View()

    def get_services(self):
        services = self.model.services.keys()
        return (self.view.list_services(services))
    
    def get_pricing(self):
        services = self.model.services.keys()
        return (self.view.list_pricing(services))
    
class Client(object):
    controller = Controller()
    print('Services Provided: ')
    controller.get_services()
    print('Pricing for Services: ')
    controller.get_pricing()

