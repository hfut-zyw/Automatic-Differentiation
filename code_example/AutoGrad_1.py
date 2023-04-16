class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        self.father = None
        self.value = None
        self.grad = 0
        self.diff_left = None
        self.diff_right = None
        self.requries_grad = None

    def __add__(self, other):
        temp = Node(self, other)
        self.father = temp
        other.father = temp
        temp.value = self.value + other.value
        temp.diff_left = 1
        temp.diff_right = 1
        return temp

    def __sub__(self, other):
        temp = Node(self, other)  # 创建新节点
        self.father = temp  # 建立连接，注意，这里的节点全都是单进单出，父亲孩子只有一个，如果同一个节点使用两次，就会fail
        other.father = temp
        temp.value = self.value - other.value
        temp.diff_left = 1
        temp.diff_right = -1
        return temp

    def __mul__(self, other):
        temp = Node(self, other)
        self.father = temp
        other.father = temp
        temp.value = self.value * other.value
        temp.diff_left = other.value
        temp.diff_right = self.value
        return temp

    def backward(self):
        if self.isroot():
            self.grad = 1
        if self.isleaf():
            return
        self.left.grad += self.grad * self.diff_left
        self.left.backward()
        self.right.grad += self.grad * self.diff_right
        self.right.backward()

    def isroot(self):
        if self.father is None:
            return True
        return False

    def isleaf(self):
        if self.left is None and self.right is None:
            return True
        return False


class mytensor(Node):
    def __init__(self, value, requries_grad=False):
        Node.__init__(self)
        self.value = value


x = mytensor(2, requries_grad=True)
y = mytensor(4, requries_grad=True)
five = mytensor(5)

z = x + y
h = y + five
f = z * h

print(f.value)
f.backward()
print(f.grad, x.grad, y.grad)
