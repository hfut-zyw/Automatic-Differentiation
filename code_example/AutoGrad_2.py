import abc
import numpy as np


class Node(object):
    """
    计算图的父类，其子类包括：
    Varible:数据向量，参数向量
    Operator:各类算子
    Layer：各类神经元
    等等......
    """

    def __init__(self, *parents):
        # 算子Operator自动调用来创建节点，并进行双向连接
        # 初始节点由Varible创建，并作为节点传入算子的父亲参数中
        # 表达式写完后，图自动创建完毕。尾节点children为空列表，初始变量节点parents为空列表
        self.parents = list(parents)
        self.children = []
        self.value = None
        self.grad = None
        for parent in self.parents:
            parent.children.append(self)

    # 获取基本属性的方法
    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def shape(self):
        return self.value.shape

    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]

    # 计算图构建完毕后，forward遍历所有节点，并通过compute方法计算出节点的value值，compute方法由各种各样多态的算子完成
    # 由于compute还没定义这里就用到了，所以后面需要使用抽象方法声明一下
    def forward(self):
        for parent in self.parents:
            if parent.value is None:
                parent.forward()     # 完成父节点的计算
        self.compute()               # 完成自己的计算

    # 在每个节点的value值被计算出后，backward遍历所有节点，算出result对每个节点的梯度，存在grad值当中
    def backward(self, result):
        if self is result:
            self.grad = np.mat(np.eye(self.dimension()))
            return self.grad
        else:
            self.grad = np.mat(np.zeros((result.dimension(), self.dimension())))
            for child in self.children:                                          # 计算从本节点出发的所有路线的梯度
                self.grad += child.backward(result) * child.get_jacobi(self)     # 前面节点的累积梯度*前节点对本节点的雅可比
            return self.grad

    @abc.abstractmethod
    def compute(self):
        """抽象方法，后面重写"""

    @abc.abstractmethod
    def get_jacobi(self):
        """抽象方法，后面重写"""


class Variable(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        self.value = np.mat(value)


class Operator(Node):
    """就是为了方便把Node的子类分类"""
    pass


class Add(Operator):
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, other):
        return np.mat(np.eye(self.dimension()))


class Sub(Operator):
    def compute(self):
        self.value = self.parents[0].value - self.parents[1].value

    def get_jacobi(self, other):
        if other is self.parents[0]:
            return np.mat(np.eye(self.dimension()))
        return -np.mat(np.eye(self.dimension()))


x = Variable(np.array([[2]]))
y = Variable([[4]])
z = Add(x, y)

print(z.value)
z.forward()
print(z.value)
x.backward(z)
print(z.grad, x.grad, x.parents)
