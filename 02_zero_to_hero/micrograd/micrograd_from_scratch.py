import math
import random


def val_guard(val):
    return val if isinstance(val, Value) else (Value(val))


class Value:
    def __init__(self, data, _prev=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._backward = lambda: None

    def _val_guard(self, val):
        return val if isinstance(val, Value) else (Value(val))

    def __add__(self, other):
        other = val_guard(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * self.grad

        # 犯错：这里backward 赋值给了self
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = val_guard(other)
        out = Value(self.data * other.data, (self, other))

        # 犯错：这里忘记累加
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self))

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Value((math.exp(2 * x) - 1) / (math.exp(2 * x) + 1), (self,))

        def _backard():
            self.grad += 1 - out.data**2

        out._backward = _backard
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
    def backward(self):
        topo = []
        visited = set()
        # dfs
        def _helper(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    _helper(prev)
                topo.append(v)
        _helper(self)
        # 犯错:这里没写
        self.grad = 1.0

        for item in reversed(topo):
            item._backward()


class Neuron:
    def __init__(self, number_of_inputs):
        self.w = [Value(random.uniform(-1,1)) for _ in range(number_of_inputs)] # 构建矩阵
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        out = 0.0
        for i,v in enumerate(x):
            out += self.w[i] * v
        out += self.b
        return out
    def parameters(self):
        return [*self.w, self.b]

class Layer:
    def __init__(self, number_of_inputs, number_of_outputs):
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_outputs)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]

    def paramters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        full_arr = [nin, *nouts]
        self.layers = [Layer(full_arr[v], full_arr[v+1]) for v in range(len(nouts))]
    def __call__(self, x):
        assert len(self.layers) < 1, ("layers cannot be zero")
        out = self.layers[0](x)
        for i in range(len(self.layers) - 1):
            out = self.layers[i + 1](out)
        return out
    def parameters(self):
        params = []
        for n in self.layers:
            params.extend(n.parameters())
        return params