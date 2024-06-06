"""
命令设计模式

如下术语：Command, Receiver, Invoker, Client

Command 对象了解Receiver对象的情况，并能调用Receiver对象的方法
调用者方法的参数值存储在Command对象中
调用者知道如何执行命令
客户端用来闯进啊Command对象并设置其为接收者

主要流程：
将请求封装为对象
可用不同的请求对客户进行参数化
允许将请求保存在队列中
提供面向对象的回调
"""

