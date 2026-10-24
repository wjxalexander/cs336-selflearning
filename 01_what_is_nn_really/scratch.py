import torch

from data_set import load_mnist
import math

from IPython.core.debugger import set_trace

(raw_x_train, raw_y_train), (rawx_valid, raw_y_valid) = load_mnist()


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (raw_x_train, raw_y_train, rawx_valid, raw_y_valid)
)

n, c = x_train.shape
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# We are initializing the weights here with Xavier initialisation
#  (by multiplying with 1/sqrt(n)).
# 这三行之后，整个模型就是 xb @ weights + bias：一个线性层
# 7850 个可训练的数（784×10 + 10）。
# 训练循环做的就是反复调整这 7850 个数，让 10 列"模板"越来越像对应的数字。

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)


bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

# Let’s implement negative log-likelihood to use as the loss function (again, we can just use standard Python):


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


print(accuracy(preds, yb))

# We can now run a training loop. For each iteration, we will:

# select a mini-batch of data (of size bs)

# use the model to make predictions

# calculate the loss

# loss.backward() updates the gradients of the model, in this case, weights and bias.

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
