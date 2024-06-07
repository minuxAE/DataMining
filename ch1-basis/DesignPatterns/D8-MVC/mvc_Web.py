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
    db = 'D8-MVC/my.db'
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()

class IndexerHandler(tornado.web.RequestHandler):
    def get(self):
        query = "select * from task"
        todos = _execute(query)
        self.render('index.html', todos=todos)

class NewHandler(tornado.web.RequestHandler):
    def post(self):
        name = self.get_argument('name', None)
        query = 'create table if not exists task (id INTEGER PRIMARY KEY, name TEXT, status NUMERIC)'
        _execute(query)

        query = "insert into task (name, status) values ('%s', %d) " %(name, 1)
        _execute(query)
        self.redirect('/')

    def get(self):
        self.render('new.html')

class UpdateHandler(tornado.web.RequestHandler):
    def get(self, id, status):
        query = "update task set status=%d where id=%s"%(int(status), id)
        _execute(query)
        self.redirect('/')

class DeleteHandler(tornado.web.RequestHandler):
    def get(self, id):
        query = "delete from task where id=%s" % id
        _execute(query)
        self.redirect('/')
        

class RunApp(tornado.web.Application):
    def __init__(self):
        Handlers = [
            (r'/', IndexerHandler),
            (r'/todo/new', NewHandler),
            (r'/todo/update/(\d+)/status/(\d+)', UpdateHandler),
            (r'/todo/delete/(\d+)', DeleteHandler)
        ]
        settings = dict(
            debug=True,
            template_path = 'templates',
            static_path='static'
        )

        tornado.web.Application.__init__(self, Handlers, **settings)

if __name__ == '__main__':
    http_server = tornado.httpserver.HTTPServer(RunApp())
    http_server.listen(5050)
    tornado.ioloop.IOLoop.instance().start()


"""
With MVC, developers can split the software application into three major parts:
model, view and controller. This helps in achieving easy maintenance, enforcing loose coupling, and decreasing complexity.
"""