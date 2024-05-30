"""
代理模式的优点：
1. 代理可以通过缓存笨重的对象或频繁访问的对象来提高应用程序的性能
2. 代理提供了对真实主题的访问授权
3. 远程代理还可以用于网络链接和数据库链接的远程服务器进行交互，并可以用于系统交互
"""


from abc import ABCMeta, abstractclassmethod

class Payment(metaclass=ABCMeta):
    @abstractclassmethod
    def do_pay(self):
        pass # 该方法需要使用代理实现

class Bank(Payment):
    def __init__(self) -> None:
        self.card = None
        self.account = None

    def __getAccount(self):
        self.account = self.card
        return self.account
    
    def __hasFunds(self):
        print('Bank:: Checking if Account', self.__getAccount(), 'has enough funds')
        return True
    
    def setCard(self, card):
        self.card = card

    def do_pay(self):
        if self.__hasFunds():
            print('Bank:: Paying the merchant')
            return True
        else:
            print('Bank:: Sorry, not enough funds!')
            return False

class DebitCard(Payment):
    def __init__(self):
        self.bank = Bank()

    def do_pay(self):
        card = input('Proxy:: Punch in Card Number:')
        self.bank.setCard(card)
        return self.bank.do_pay()



class You:
    def __init__(self) -> None:
        print('You:: Lets buy the Denim shirt')
        self.debitCard = DebitCard()

    def make_payment(self):
        self.isPurchased = self.debitCard.do_pay()

    def __del__(self):
        if self.isPurchased:
            print('You:: Wow! Denim shirt is Mine!')
        else:
            print('You:: I should earn more!')

you = You()
you.make_payment()

    
