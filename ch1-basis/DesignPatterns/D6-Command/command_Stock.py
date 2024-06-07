"""
创建Order接口定义客户端下达的订单
定义ConcreteCommand进行股票交易
执行交易的Receiver类
"""

from abc import ABCMeta, abstractclassmethod

# 抽象类Order和抽象方法execute
class Order(metaclass=ABCMeta):
    @abstractclassmethod
    def execute(self):
        pass

# 具体类：BuyStockOrder和SellStockOrder
class BuyStockOrder(Order):
    def __init__(self, stock) -> None:
        self.stock = stock

    def execute(self):
        self.stock.buy()

class SellStockOrder(Order):
    def __init__(self, stock) -> None:
        self.stock = stock

    def execute(self):
        self.stock.sell()

# StockTrade类表示例子中的Receiver对象
class StockTrade:
    def buy(self):
        print('You will buy stocks')

    def sell(self):
        print('You will sell stocks')

# Agent类表示调用者
# 代理是客户端和StockExchange之间的中介，并执行客户下达的订单
class Agent:
    def __init__(self) -> None:
        self.__orderQueue = []

    def placeOrder(self, order):
        self.__orderQueue.append(order)
        order.execute()

if __name__ == '__main__':
    # Client
    stock = StockTrade()
    buyStock = BuyStockOrder(stock)
    sellStock = SellStockOrder(stock)

    # Invoker
    agent = Agent()
    agent.placeOrder(buyStock)
    agent.placeOrder(sellStock)
