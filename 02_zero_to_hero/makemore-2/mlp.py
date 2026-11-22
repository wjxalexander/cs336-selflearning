import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures
from pathlib import Path

words = (
    (Path(__file__).resolve().parent.parent / "makemore" / "names.txt")
    .read_text()
    .splitlines()
)

chars = sorted(list(set("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}

stoi["."] = 0

itos = {i: s for s, i in stoi.items()}

block_size = 3
num = 5
# reminder: the x, y store the indexs of s not s itself
X = []
Y = []
for word in words[:num]:
    context = [0] * block_size  # 0 means separator index
    for s in word + ".":
        ix = stoi[s]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '---->', s )
        context = context[1:] + [ix]  # ... -> ..e -> .em ...

X = torch.tensor(X)
Y = torch.tensor(Y)
print(f"X shape is {X.shape}")
C = torch.randn(
    (27, 2), requires_grad=True
)  # embeding aks lookup table, 2每个字母用几个数来表示，这是自己定的。 叫做 embedding 的维度。

"""
F.one_hot(torch.tensor(5), 27).float() @ C 
取出 C 中的第五个 ==> C[5]
"""

# print(C[torch.tensor([5,6,7])], C[X])
"""
对，就是 lookup。C[X] 没有任何计算，只是按编号查表。
"""

emb = C[X]
print(emb.shape)  # torch.Size([n, 3, 2])
print(emb[3, 1])  # 第 3 个样本的第 1 个字母
print(C[X[3, 1]])  # 同一个字母，直接查表，应该一样

"""
所以3 是上下文 长度 2 是每个字母用多少个数字来代替？
对，完全正确。`emb` 的三个维度分别是：

```
emb.shape = (n, 3, 2)
             ↑  ↑  ↑
             │  │  └─ 每个字母用 2 个数表示（embedding 维度）
             │  └──── 上下文有 3 个字母（视频里叫 block_size）
             └─────── 有 n 个样本
```

所以 6 = 3 × 2 的含义是：**一个样本的全部输入，一共 6 个数。**

这两个数都是你自己选的超参数，改了以后 `W1` 的第一个数要跟着变：

| 上下文长度 | embedding 维度 | 每个样本的输入 | `W1` 的形状 |
|---|---|---|---|
| 3 | 2 | 6 | (6, 100) |
| 3 | 10 | 30 | (30, 200)，视频后面改成这个 |
| 8 | 10 | 80 | (80, …) |

对比一下计数法就能看出好处：上下文从 3 个字母加到 8 个，
这里的输入只是从 6 个数变成 16 个数，线性增长；而计数表要从 27⁴ 变成 27⁹ 个格子，指数增长。这
就是 embedding 加隐藏层能处理长上下文的原因。
"""
""" Firset layer """
n_in = emb.shape[1] * emb.shape[2]  # 3 * 2 = 6
n_of_first_layer_nuerons = 100
W1 = torch.randn(n_in, n_of_first_layer_nuerons, requires_grad=True)
B1 = torch.randn(100, requires_grad=True)

"""
所有样本里，第 0 个位置上那个字母的 embedding
emb[  :  ,  0  ,  :  ]
      ↑     ↑     ↑
      │     │     └─ 2 个数全要
      │     └─────── 只要第 0 个字母（上下文里最左边那个）
      └───────────── 32 个样本全要
"""
# print(torch.cat([emb[:,0,:], emb[:, 1 ,: ], emb[:,2,:]], dim=1)) # 返回3个 独立的tensor
# https://blog.ezyang.com/2019/05/pytorch-internals/
# a = torch.cat(torch.unbind(emb, 1), dim=1)
# b = emb.view(32, 6)
# print(a.shape, torch.equal(a, b))     # torch.Size([32, 6]) True

h = emb.view(-1, 6) @ W1 + B1
h_tanh = torch.tanh(h)
print(h_tanh.shape)
""" second layer """
w2 = torch.randn(n_of_first_layer_nuerons, 27, requires_grad=True)  # (input, output)
b2 = torch.randn(27, requires_grad=True)

logits = h_tanh @ w2 + b2
print(logits.shape)
"""
soft max
counts = logits.exp()                              # softmax 的前半
prob = counts / counts.sum(1, keepdim=True)        # softmax 的后半
loss = -prob[torch.arange(n), Y].log().mean()      # 取正确答案那格，负对数，平均
"""
loss = F.cross_entropy(logits, Y)
print(f"loss is: {loss.item()}")


# backward
paramters = [C, W1, B1, w2, b2]  # requires_grad is set where each one is created

for p in paramters:
    p.grad = None
loss.backward()
for p in paramters:
    p.data += -(0.1 * p.grad)


ix = torch.randint(0, X.shape[0], (32,))
em = C[X[ix]]
# ....
loss = F.cross_entropy(logits, Y[ix])

learning_rate = torch.linspace(-3, 0, 1000)
lrs = 10**learning_rate

for i in range(100):
    # ....
    lr = lrs[i]
    for p in paramters:
        p.data += -(lr * p.grad)


# chars = sorted(set(list(''.join(words))))
# stoi = { s: i + 1 for i, s in enumerate(chars)}
# stoi['.'] = 0
# itos = {i: s for s, i in stoi.items()}
def build_dataset(words):
    block_size = 3
    x = []
    y = []
    for word in words:
        context = [0] * block_size
        seq = word + "."
        for ch in seq:
            x.append(context)
            y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
