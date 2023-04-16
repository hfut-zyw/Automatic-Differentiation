class Tensor:
    def __init__(self, data=None, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad

        self.grad = None
        self.grad_fn = None
        self.is_leaf = False

        if self.requires_grad and self.is_leaf:
            self.grad_fn = AccumulateGrad(self)  # 反向传播的终点节点

    def backward(self):
        self.grad_fn(grads=1)


class to_tensor(Tensor):
    def __init__(self, data=None, requires_grad=False):
        Tensor.__init__(self, data, requires_grad)
        self.is_leaf = True
        if self.requires_grad:
            self.grad_fn = AccumulateGrad(self)


class Mul:
    """
    1.产生前向计算的节点
    2.计算前向计算的结果
    3.记录反向传播的节点以及连接，即构建反向传播图

    """

    def __init__(self):
        pass

    def __call__(self, a, b):
        c = Tensor(data=a.data + b.data)
        if a.requires_grad or b.requires_grad:
            c.requires_grad = True
        if c.requires_grad:
            next_fn=(a.grad_fn, b.grad_fn)
            ctx = (a.data, b.data)
            c.grad_fn = MulBackward(next_fn, ctx=ctx)
        return c


class MulBackward:
    def __init__(self, next_fn, ctx=None):  # *grad_fn是用来建立连接的，ctx是用来求梯度的
        self.next_functions = list(next_fn)
        self.x, self.y = ctx
        self.dx = self.y
        self.dy = self.x

    def __call__(self, grads=1):
        if self.next_functions[0] is not None:
            self.next_functions[0](grads * self.dx)
        if self.next_functions[1] is not None:
            self.next_functions[1](grads * self.dy)
        return


class AccumulateGrad:
    def __init__(self, node):
        self.node = node  # 记录它对应的节点，以便把求出来的梯度传进去
        self.acc_grads = None

    def __call__(self, grads):
        if self.acc_grads is None:
            self.acc_grads = grads
        else:
            self.acc_grads = self.acc_grads + grads
        self.node.grad = self.acc_grads
        return

mul=Mul()
a = to_tensor(data=2)
b = to_tensor(data=8, requires_grad=True)
c = mul(a, b)

d = to_tensor(data=3)
e = mul(c, d)
e.backward()
print(b.grad)