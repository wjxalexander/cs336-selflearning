import random
import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        # * self.grad = ∂L/∂self: final output L's derivative wrt this node
        # start 0: no effect yet, base for += in _backward
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value=(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # Backpropagation Chain rule https://en.wikipedia.org/wiki/Chain_rule
        # ! += accumulate, not = overwrite: node may be reused (a + a,
        # ! or a feeds many nodes); multivariable chain rule sums all paths
        # ! b = a + a -> db/da = 2 not 1; = gives wrong grad silently, no error
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        # ! += accumulate, same reason as __add__
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self 2 * val(a)
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data  # local derivative * out.grad: chain rule

        out._backward = _backward
        return out

    # Hyperbolic tangent: https://en.wikipedia.org/wiki/Hyperbolic_functions
    def tanh(self):
        raw = self.data
        exp_val = math.exp(2 * raw)
        t_val = (exp_val - 1) / (exp_val + 1)
        out = Value(t_val, (self,), "tanh")

        # ! local derivative 1 - o**2; += accumulate
        def _backward():
            self.grad += (1 - (t_val**2)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __radd__(self, other):  # other + self
        return self + other

    def backward(self):
        # * toposort: node enters topo only after all its inputs
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # ! important: the result grad is 1
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, no_of_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(no_of_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # * forward pass, one neuron: x in, out out. n(x) like PyTorch model(x)
        # 1. linear: w1*x1 + w2*x2 + ... + b
        # 2. non-linear: tanh squash to (-1, 1)
        # each *, +, tanh make new Value, record _prev & _backward: graph built here
        # backward not here: out.backward() walks graph in reverse, fills w, b grads
        # w = [0.5, -0.3, 0.8] x = [2.0,  3.0, -1.0]
        # list(zip(w, x)) [(0.5, 2.0), (-0.3, 3.0), (0.8, -1.0)]
        # activation = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        out = activation.tanh()
        return out

    def parameters(self):
        # * trainable Values: [w1, ..., wn, b]. Neuron(3) -> 4 params
        # same objects, not copies: p.data += ... updates the network
        # p.data = current value. p.grad = dloss/dp, set by loss.backward()
        # not params: input x (data), intermediate results (rebuilt each forward pass)
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin, nout):
        # * nin: inputs per neuron, sets its number of w. nout: neurons in this layer
        # every neuron sees full input, so each needs nin weights
        # each neuron gives 1 output, so nout outputs need nout neurons
        # Layer(2, 3) -> [Neuron(2), Neuron(2), Neuron(2)], params = 3 * (2 + 1) = 9
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # ! flat list of Values: extend, not append. append nests lists, p.data fails
        # Layer(3, 4) -> 4 * (3 + 1) = 16 params
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, nin, nouts):
        # * nin: input count of whole net. nouts: output count per layer, a list
        # prepend nin: each layer's nin = previous layer's nout
        # MLP(3, [4, 4, 1]) -> Layer(3, 4), Layer(4, 4), Layer(4, 1)
        nouts_with_nin = [nin, *nouts]
        # self.layers = []
        # for i, nout in enumerate(nouts_with_nin[1:], start=1):
        #     self.layers.append(Layer(nouts_with_nin[i - 1], nout))
        self.layers = [
            Layer(nouts_with_nin[i], nouts_with_nin[i + 1])
            for i in range(len(nouts_with_nin) - 1)
        ]

    def __call__(self, x):
        # * forward pass: each layer's output becomes next layer's input
        # ! Layer returns list, even when nout = 1. list has no backward()
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # all layers' params, flat. MLP(3, [4, 4, 1]) -> 16 + 20 + 5 = 41 params
        return [p for layer in self.layers for p in layer.parameters()]


# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4,4,1])
# print(n(x))
n = MLP(3, [4, 4, 1])
print(n.parameters())
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

# ypred = [n(x) for x in xs]
# print(ypred)

# # the loss
# loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

# # print(loss)

# loss.backward()

# # print(n.layers[0].neurons[0].w[0].grad)

# for p in n.parameters():
#     # ! minus: grad points to loss increase, step opposite to reduce loss
#     # p.data = 1.0, p.grad = 2.0: p up raises loss. 1.0 + -0.01 * 2.0 = 0.98, p down
#     # p.data = 1.0, p.grad = -2.0: p up lowers loss. 1.0 + -0.01 * -2.0 = 1.02, p up
#     p.data += -0.01 * p.grad

for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # ! backward on loss, not ypred: goal is lower loss, so each param needs dloss/dp
    # loss holds prediction and target: w, b, x -> ypred -> (ypred - ygt)**2 -> loss
    # ypred.backward() gives only dypred/dp: how p moves prediction, not if it is right
    # chain rule: dloss/dp = dloss/dypred * dypred/dp
    #   dloss/dypred = 2 * (ypred - ygt): which way prediction should move
    #   dypred/dp: how p affects prediction
    # ygt = 1.0, ypred = 0.5: dloss/dypred = -1.0, prediction too low, raise it
    # ygt = 1.0, ypred = 1.5: dloss/dypred = 1.0, prediction too high, lower it
    # ygt = 1.0, ypred = 1.0: dloss/dypred = 0.0, prediction right, no change
    # loss sums all samples, so p.grad sums each sample's contribution

    # ! zero grads before backward: p.grad must reflect only this round
    # _backward uses +=, so without zeroing every earlier round's grad adds in
    # no zero -> round 3: p.grad = g1 + g2 + g3, g1 and g2 come from old params
    # why only this round: a grad is the slope at the params it was computed on
    # params update each round, so an old grad describes a position p already left
    # loss = p**2, lr = 0.1: p = 3.0, grad 6.0 -> p = 2.4, true grad 4.8, no zero uses 10.8
    # at minimum p = 0: true grad 0, p should stop. Stale grad keeps moving it past minimum
    # zero clears earlier rounds only: paths and samples in one round still add with +=
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
