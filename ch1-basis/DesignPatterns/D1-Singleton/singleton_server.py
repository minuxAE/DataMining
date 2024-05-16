"""
单例实现基础设施监控服务
"""

class ServerCheck:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not ServerCheck._instance:
            ServerCheck._instance = super(ServerCheck, cls).__new__(cls, *args, **kwargs)
        return ServerCheck._instance
    
    def __init__(self) -> None:
        self._servers = []

    def addServer(self):
        self._servers.append("Server 1")
        self._servers.append("Server 2")
        self._servers.append("Server 3")
        self._servers.append("Server 4")

    def changeServer(self):
        self._servers.pop()
        self._servers.append("Server 5")

sc1 = ServerCheck()
sc2 = ServerCheck()

sc1.addServer()

print('Check For Servers (1)...')

for i in range(4):
    print('Checking ', sc1._servers[i])

sc2.changeServer()

print('Check For Servers (2)...')

for i in range(4):
    print('Checking ', sc2._servers[i])





