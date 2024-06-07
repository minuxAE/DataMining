class ComputerState(object):
    name = 'state'
    allowed = []

    def switch(self, state):
        if state.name in self.allowed:
            print('Current:', self, '=> switched to new state', state.name)
        else:
            print('Current:', self, '=> switching to', state.name, 'not possible.')

    def __str__(self):
        return self.name
    
class Off(ComputerState):
    name = 'off'
    allowed = ['on']

class On(ComputerState):
    name = 'on'
    allowed = ['off', 'suspend', 'hibernate']

class Suspend(ComputerState):
    name = 'suspend'
    allowed = ['on']

class Hibernate(ComputerState):
    name = 'hibernate'
    allowed = ['on']

class Computer(object):
    def __init__(self, model='HP') -> None:
        self.model = model
        self.state = Off()

    def change(self, state):
        self.state.switch(state)

if __name__ == '__main__':
    comp = Computer()
    comp.change(On)
    comp.change(Off)

    comp.change(On)
    comp.change(Suspend)
    comp.change(Hibernate)
    comp.change(On)
    comp.change(Off)


