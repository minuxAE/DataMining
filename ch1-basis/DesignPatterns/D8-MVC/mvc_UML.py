"""
Model: 定义与客户端某些任务相关的业务逻辑或者操作
View: 定义客户端查看的视图或者展示，模型根据业务逻辑向视图呈现数据
Controller: 视图和模型之间的接口
"""

class Model(object):
    def logic(self):
        data = 'Got it'
        print('Model: Crunching data as per business logic')
        return data
    
class View(object):
    def update(self, data):
        print('View: Updating the view with results: ', data)

class Controller(object):
    def __init__(self) -> None:
        self.model = Model()
        self.view = View()

    def interface(self):
        print('Controller: Relayed the Client asks')
        data = self.model.logic()
        self.view.update(data)

class Client(object):
    print('Client: asks for certain information')
    controller = Controller()
    controller.interface()