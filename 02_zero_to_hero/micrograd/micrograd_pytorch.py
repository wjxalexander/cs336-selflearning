import torch

# # inputs x1,x2
# x1 = Value(2.0, label="x1")
# x2 = Value(0.0, label="x2")
# # weights w1,w2
# w1 = Value(-3.0, label="w1")
# w2 = Value(1.0, label="w2")
# # bias of the neuron
# b = Value(6.8813735870195432, label="b")

x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

bias = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)

n = x1 * w1 + x2 * w2 + bias
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("---")
print("x1", x1.data.item(), x1.grad.item())
print("w1", w1.data.item(), w1.grad.item())
print("x2", x2.data.item(), x2.grad.item())
print("w2", w2.data.item(), w2.grad.item())
print("bias", bias.data.item(), bias.grad.item())
