from pathlib import Path
import torch
import torch.nn.functional as F

words = (Path(__file__).parent / "names.txt").read_text().splitlines()

SEP = "."
chars = sorted(set(list("".join(words))))

# 1st go stoi and itos
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi[SEP] = 0
print(stoi.items())
itos = {i: s for s, i in stoi.items()}
# 2nd get xs and ys xs: train set ys: correct set
xs = []
ys = []
for word in words:
    seq = [SEP] + list(word) + [SEP]  # construct sequence
    for x, y in zip(seq, seq[1:]):
        # x next is y
        idx1 = stoi[x]
        idx2 = stoi[y]
        xs.append(idx1)
        ys.append(idx2)

# note: 这里忘记
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator()

# 构建W
W = torch.randn((27, 27), generator=g, requires_grad=True)
for i in range(100):
    x_encode = F.one_hot(xs, 27).float()
    # FORWARD: 得到 "计数"
    count = (x_encode @ W).exp()

    # soft sum
    p = count / count.sum(dim=1, keepdim=True)

    # loss function: 对数相加等于 p1 * p2 * p3
    # 但是 自己还是不太明白这里为什么能做为 lossfunction
    loss = -p[torch.arange(num), ys].log().mean() + (0.01 * (W**2)).mean()
    print(f"loss value is: {loss.item()}")
    # reset W grad
    W.grad = None
    loss.backward()

    # print(W.grad, W.data) 这里 一开始用的+= LOSS 越来越大
    W.data += -(10 * W.grad)

g2 = torch.Generator()

for i in range(5):
    out = []
    ix = 0
    while True:
        # 只要一个字母
        x_encode = F.one_hot(torch.tensor([ix]), 27).float()
        count = (x_encode @ W).exp()
        p = count / count.sum(dim=1, keepdim=True)
        # print(p)
        next_ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g2).item()
        out.append(itos[next_ix])
        ix = next_ix
        if(ix == 0):
            break
    print("".join(out))
