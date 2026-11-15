import torch
import torch.nn.functional as F
from pathlib import Path

words = (Path(__file__).parent / "names.txt").read_text().splitlines()

SEP = "."
chars = sorted(set(list("".join(words))))
xs = []
ys = []

"""
..emma
 emma.
"""

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi[SEP] = 0
# print(stoi.items())
itos = {i: s for s, i in stoi.items()}

for word in words:
    seq = [SEP] + [SEP] + list(word) + [SEP]
    for ch1, ch2, ch3 in zip(seq, seq[1:], seq[2:]):
        xs.append([stoi[ch1], stoi[ch2]])
        ys.append(stoi[ch3])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"xs shape is: {xs.shape}, ys shape is: {ys.shape}")

w = torch.randn((27,27,27), requires_grad=True)

for _i in range(100):
    # for id1, id2 in xs:
    #     print(w[id1, id2])
    #     log_count.append(w[id1, id2])
    #forward
    logits = w[xs[:,0], xs[:,1]]
    
    #loss
    # num = xs.shape[0]
    # count_exp = logits.exp()
    # p = count_exp / count_exp.sum(1, keepdim=True)
    # fine = (0.01 * (w ** 2)).mean()
    # loss = -p[torch.arange(num), ys].log().mean() + (0.01 * (w**2)).mean()
    loss =F.cross_entropy(logits, ys)
    w.grad = None
    loss.backward()
    print(f"loss is {loss}")
    w.data += -(20 * w.grad)


for i in range(5):
    out = []
    idx = [0, 0]
    while True:
        logits = w[idx[0], idx[1]]
        # cout_exp = logits.exp()
        # p = cout_exp / cout_exp.sum()
        p = F.softmax(logits,dim=0)
        next_idx = torch.multinomial(p, num_samples=1, replacement=True).item()
        if next_idx == 0:
            break
        out.append(itos[next_idx])
        idx=[idx[1], next_idx]
    print("".join(out))