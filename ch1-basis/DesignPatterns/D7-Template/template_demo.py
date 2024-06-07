"""
模板方法使用场景：
1. 多个算法或者类实现相似逻辑
2. 子类中实现算法有助于减少重复代码
3. 可以让子类利用覆盖实现行为来定义多个算法的时候

交叉编译

使用基本操作定义算法框架
重新定义子类的操作
实现代码重用并避免重复工作
利用通用接口或者实现
"""

from abc import ABCMeta, abstractclassmethod

class Compiler(metaclass=ABCMeta):
    @abstractclassmethod
    def collectSource(self):
        pass

    @abstractclassmethod
    def compileToObject(self):
        pass

    @abstractclassmethod
    def run(self):
        pass

    def compileAndRun(self):
        self.collectSource()
        self.compileToObject()
        self.run()

class iOSCompiler(Compiler):
    def collectSource(self):
        print('Collecting Swift Source Code!')

    def compileToObject(self):
        print('Compiling Swift Code to LLVM bitcode')

    def run(self):
        print('Program running on runtime environment')

iOS = iOSCompiler()
iOS.compileAndRun()


