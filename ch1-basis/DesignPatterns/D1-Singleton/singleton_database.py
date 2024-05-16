import sqlite3
from typing import Any

class MetaSingleton(type):
    _instances = {}
    
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwds)
        return cls._instances[cls]
    
class DataBase(metaclass = MetaSingleton):
    connection = None
    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect('D1-Singleton/db.sqlite3')
            self.cursor_obj = self.connection.cursor()
        return self.cursor_obj
    
db1 = DataBase().connect()
db2 = DataBase().connect()

# 实现了单例模式，db1和db2指向了相同的对象
print('Database Object DB1', db1)
print('Database Object DB2', db2)