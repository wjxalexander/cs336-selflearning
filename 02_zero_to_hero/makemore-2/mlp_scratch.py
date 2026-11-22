import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures
from pathlib import Path

words = (
    (Path(__file__).resolve().parent.parent / "makemore" / "names.txt")
    .read_text()
    .splitlines()
)

chars = sorted(set(list("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

num_of_chars = 3


def convert_to_train_set(words):
    """
    emma -> emma.
    ... -> e
    ..e -> m
    .em -> m
    emm -> a
    mma -> .
    """
    train_set = []
    target_set = []
    for word in words:
        seq = word + "."
        context = [0] * num_of_chars
        for ch in seq:
            train_set.append(context)
            target_set.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(train_set), torch.tensor(target_set)


x, y = convert_to_train_set(words[:2])
print(x.shape)


def mlp(words):
    train_set, target_set = convert_to_train_set(words)
    # 每个char 用二维embbeding 表示
    embeding_dimension = 2
    C = torch.randn(27, embeding_dimension, requires_grad=True)
    num_of_input = num_of_chars * embeding_dimension
    num_of_nerons = 100
    # layer one
    W1 = torch.randn((num_of_input, num_of_nerons), requires_grad=True)
    B1 = torch.randn(num_of_nerons, requires_grad=True)
    # layer two: 输出27的字符
    W2 = torch.randn((num_of_nerons, 27), requires_grad=True)
    B2 = torch.randn(27, requires_grad=True)
    learning_rate = 0.1
    parameters = [C, W1, B1, W2, B2]
    final_loss = 0.0
    for i in range(30000):
        """mini batch: 随机取出几个"""
        idx = torch.randint(0, train_set.shape[0], (64,))
        train_mini_batch = train_set[idx]
        target_mini_batch = target_set[idx]
        """forward: """
        """first layer: tanh"""
        # 取出所有的train set 的对应的未被优化的值
        embeding = C[train_mini_batch]
        """
        W1.shape -> [6, 100] 
        embeding.shape -> [n, 3, 2]
        目标 embedding @ W1 + B1 目前不兼容怎么办
        embeding.view(-1, 6) -> [n, 6]
        输入的维度要符合 neuron layer 的定义
        """
        h = embeding.view(-1, num_of_input)
        layer_1 = torch.tanh(h @ W1 + B1)
        """sencond layer:"""
        logits = layer_1 @ W2 + B2

        # back progation 准备工作
        for param in parameters:
            param.grad = None
        loss = F.cross_entropy(logits, target_mini_batch)
        final_loss = loss.item()
        loss.backward()
        for param in parameters:
            param.data += -(learning_rate * param.grad)
    print(f"final loss is: {final_loss}")
    return [C, W1, B1, W2, B2]

def word_generator(words):
    C, W1, B1, W2, B2 = mlp(words)
    names = []

    for _ in range(10):
        context = [0] * num_of_chars
        out = []
        while True:
            embeding = C[torch.tensor(context)]
            h = torch.tanh(embeding.view(1, -1) @ W1 + B1)
            logits = h @ W2 + B2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1,).item()
            out.append(ix)
            context = context[1:] + [ix]
            if(ix == 0):
                break
        names.append(''.join(itos[i] for i in out))
    return names
"""
报错了：
因为 embeding = C[train_set]（第 51 行）和 h = embeding.view(...)（第 71 行）写在了循环外面，第二次 backward() 时要重走一段已经被释放的计算图。

PyTorch 每做一次运算就在计算图里记一个节点。C → embeding → h 这一段只在循环外建了一次。
循环里每轮都新建 h → layer_1 → logits → loss 这一段，但它们都接在同一个旧的 h 上。
第 1 轮 loss.backward() 一路回传到 C，然后 PyTorch 为了省内存，把走过的图的中间值全部释放，包括 C → embeding → h 那段。
第 2 轮 loss.backward() 又要经过 h → embeding → C，而这段已经被释放，于是报错。
修法：把 embedding 查表和 view 挪进循环，让每轮前向都从 C 重新建完整的图：

"""
for name in word_generator(words):
    print(name)