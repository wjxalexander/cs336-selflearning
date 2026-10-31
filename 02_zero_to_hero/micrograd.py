import matplotlib.pyplot as plt
import math
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        # self.grad = 最终输出 L 对这个节点的导数，∂L/∂self。
        self.grad = 0.0  # initially zero because: 0 means no effect
        self._backward = lambda: None

    def __repr__(self):
        return f"Value=(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        # Backpropagation Chain rule https://en.wikipedia.org/wiki/Chain_rule
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data

        out._backward = _backward
        return out

    # Hyperbolic tangent: https://en.wikipedia.org/wiki/Hyperbolic_functions
    def tanh(self):
        raw = self.data
        t_val = (math.exp(2 * raw) - 1) / (math.exp(2 * raw) + 1)
        out = Value(t_val, (self,), "tanh")

        # 1 - o**2
        def _backward():
            self.grad = (1 - (t_val**2)) * out.grad

        out._backward = _backward
        return out


print(Value(2.0))
a = Value(3.0, label="a")
b = Value(3.0, label="b")
c = a + b
c.label = "c"
print(c)
e = Value(1.5, label="e")
d = c * e
d.label = "d"
print("d", d)
# print(d._prev, d._op, c._op, c._prev)

from graphviz import Digraph


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


f = Value(-2.0, label="F")
l = d * f
l.label = "l"

l.grad = 1.0

# dl/dd = f
# dl/df = d
f.grad = d.data
d.grad = f.data

# dl/dc = dl/dd * dd/dc
# dl/de = dl/dd * dd/de
c.grad = e.data * d.grad
e.grad = c.data * d.grad

# dl/da = dl/dc * dc/da
# dl/db = dl/dc * dc/db
a.grad = c.grad * 1
b.grad = c.grad * 1
dot = draw_dot(l)
# dot.render("micrograd_graph", format="png", cleanup=True)  # 生成 micrograd_graph.png

# 沿梯度方向微调叶子节点，让 l 变大
print("before", l.data)
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
e.data += 0.01 * e.grad
f.data += 0.01 * f.grad

c = a + b
d = c * e
l = d * f

print("after", l.data)

plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
plt.grid()
# plt.show()

# inputs x1,x2
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights w1,w2
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias of the neuron
b = Value(6.8813735870195432, label="b")
# x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"

o = n.tanh()
o.label = "o"

# o.grad = 1

# # o = tanh(n)
# # do/dn =  1 - o**2
# n.grad = (1 - o.data**2) * o.grad
# x1w1x2w2.grad = 1 * n.grad
# b.grad = 1 * n.grad
# x1w1.grad = 1 * x1w1x2w2.grad
# x2w2.grad = 1 * x1w1x2w2.grad

# x2.grad = w2.data * x2w2.grad
# w2.grad = x2.data * x2w2.grad

# x1.grad = w1.data * x1w1.grad
# w1.grad = x1.data * x1w1.grad

o.grad = 1.0 # important
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
x2._backward()
w2._backward()
x1._backward()
w1._backward()

dot = draw_dot(o)
dot.render("micrograd_graph_n", format="png", cleanup=True)  # 生成 micrograd_graph.png
