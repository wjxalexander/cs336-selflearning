# =============================================================================
# Legend: the PyTorch building blocks used in this file. Search for the tag.
#
# [nn.Module]    A callable that acts like a function but also holds state
#                (layer weights). It knows its Parameters, so it can zero their
#                grads and loop over them for updates.
# [nn.Parameter] A tensor wrapper that tells a Module "this weight must be
#                updated during backprop". Only tensors with requires_grad=True
#                get updated.
# [F]            torch.nn.functional: stateless activations, loss functions,
#                and non-stateful versions of layers (conv, linear).
# [optim]        torch.optim: optimizers such as SGD that update Parameters
#                after the backward step.
# [Dataset]      An abstract interface with __len__ and __getitem__,
#                e.g. TensorDataset.
# [DataLoader]   Wraps any Dataset and returns an iterator over mini-batches.
# =============================================================================

import torch
from torch import optim  # [optim]

from data_set import load_mnist
import math
import torch.nn.functional as F  # [F]
from torch import nn  # [nn.Module] [nn.Parameter]
from torch.utils.data import TensorDataset  # [Dataset]
from torch.utils.data import DataLoader  # [DataLoader]
import numpy as np

from IPython.core.debugger import set_trace

(raw_x_train, raw_y_train), (rawx_valid, raw_y_valid) = load_mnist()


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (raw_x_train, raw_y_train, rawx_valid, raw_y_valid)
)



n, c = x_train.shape
bs = 64  # batch size
# [Dataset] TensorDataset pairs x and y, so train_ds[i] returns (x_i, y_i).
train_ds = TensorDataset(x_train, y_train)
# [DataLoader] Yields shuffled (xb, yb) mini-batches of size bs from the Dataset.
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)


# [Dataset] + [DataLoader] Validation needs no backprop, so it uses a 2x batch.
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# We are initializing the weights here with Xavier initialisation
#  (by multiplying with 1/sqrt(n)).
# 这三行之后，整个模型就是 xb @ weights + bias：一个线性层
# 7850 个可训练的数（784×10 + 10）。
# 训练循环做的就是反复调整这 7850 个数，让 10 列"模板"越来越像对应的数字。

# Before refactor: raw tensors with requires_grad. No Module tracks them yet,
# so we must update and zero them by hand. Later [nn.Parameter] replaces this.
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# def log_softmax(x):
#     return x - x.exp().sum(-1).log().unsqueeze(-1)

# def model(xb):
#     return log_softmax(xb @ weights + bias)


xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]

# Let’s implement negative log-likelihood to use as the loss function (again, we can just use standard Python):

# def nll(input, target):
#     return -input[range(target.shape[0]), target].mean()

# loss_func = nll
# F.cross_entropy(logits, target)  ==  nll(log_softmax(logits), target)
# [F] A stateless loss function: no weights, just math on its inputs.
loss_func = F.cross_entropy

# def model(xb):
#     return xb @ weights + bias
# preds = model(xb)  # predictions

# print(loss_func(preds, yb))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):
#         #         set_trace()
#         start_i = i * bs
#         end_i = start_i + bs
#         xb = x_train[start_i:end_i]
#         yb = y_train[start_i:end_i]
#         pred = model(xb)
#         loss = loss_func(pred, yb)

#         loss.backward()
#         with torch.no_grad():
#             weights -= weights.grad * lr
#             bias -= bias.grad * lr
#             weights.grad.zero_()
#             bias.grad.zero_()

# print(loss_func(model(xb), yb), accuracy(model(xb), yb))


# [nn.Module] Subclassing nn.Module makes the model callable: model(xb) runs
# forward(xb). It also holds the weights as state.
class Mnist_Logistic(nn.Module):
    # [nn.Parameter] Version 1: wrapping tensors in nn.Parameter registers them
    # with the Module, so model.parameters() and model.zero_grad() find them.
    # def __init__(self):
    #     super().__init__()
    #     self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
    #     self.bias = nn.Parameter(torch.zeros(10))
    def __init__(self):
        super().__init__()
        # [nn.Module] Version 2: nn.Linear is itself a Module that creates its
        # own weight and bias Parameters. Assigning it as an attribute nests
        # it, so its Parameters belong to this model too.
        self.lin = nn.Linear(784, 10)

    # def forward(self, xb):
    #     return xb @ self.weights + self.bias
    def forward(self, xb):
        return self.lin(xb)


def get_model():
    model = Mnist_Logistic()
    # [optim] SGD receives every Parameter via model.parameters().
    # opt.step() then replaces the manual "p -= p.grad * lr" loop.
    return model, optim.SGD(model.parameters(), lr=lr)


model, opt = get_model()

print("before: ", loss_func(model(xb), yb))


def fit():
    # for epoch in range(epochs):
    #     for i in range((n - 1) // bs + 1):
    #         start_i = i * bs
    #         end_i = start_i + bs
    #         xb = x_train[start_i:end_i]
    #         yb = y_train[start_i:end_i]
    #         pred = model(xb)
    #         loss = loss_func(pred, yb)

    #         loss.backward()
    #         with torch.no_grad():
    #             # [nn.Module] Looping over model.parameters() and calling
    #             # model.zero_grad() only works because the Module tracks them.
    #             # for p in model.parameters():
    #             #     p -= p.grad * lr
    #             # model.zero_grad()
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            """step 1"""
            # start_i = i * bs
            # end_i = start_i + bs
            # xb = x_train[start_i:end_i]
            # yb = y_train[start_i:end_i]
            """step 2"""
            # [Dataset] __getitem__ with a slice returns x and y together.
            xb, yb = train_ds[i * bs : i * bs + bs]

            pred = model(xb)  # [nn.Module] calling the model runs forward()
            loss = loss_func(pred, yb)  # [F] cross_entropy

            loss.backward()
            opt.step()  # [optim] update every Parameter
            opt.zero_grad()  # [optim] reset grads for the next batch


# fit()

def fit2():
    # [DataLoader] No more index math: iterating yields (xb, yb) batches.
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
# fit2()

def fit3():
    for epoch in range(epochs):
        # [nn.Module] train()/eval() switch mode for layers like Dropout and
        # BatchNorm. Linear ignores it, but it is a good habit.
        model.train()
        for xb, yb in train_dl:  # [DataLoader] shuffled training batches
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            # [DataLoader] valid_dl is not shuffled, order does not matter here.
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
        print(epoch, valid_loss / len(valid_dl))

fit3()
print("after: ", loss_func(model(xb), yb))

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)  # [nn.Module] forward + [F] loss

    # [optim] Passing an optimizer means "train on this batch".
    # Passing None means "only measure the loss" (used for validation).
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit4(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    # [Dataset] in, [DataLoader] out.
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)