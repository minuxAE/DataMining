"""
使用Tornado开发应用程序，管理用户的各种任务，还具备添加任务、更新任务、删除任务的权限
"""
import tornado
import tornado.web
import tornado.ioloop
import tornado.httpserver
import sqlite3

"""
数据库操作由如下4个应用程序完成：
IndexHandler: 返回存储在数据库中的所有任务
NewHandler: 添加新任务
UpdateHandler: 标记任务书为完成或者重新打开
DeleteHandler: 从数据库中删除任务
"""

def _execute(query:str):
    

class IndexerHandler(tornado.web.RequestHandler):
    def get(self):
        query = "select * from task"
        todos = _execute(query)
        self.render('index.html', todos=todos)
        