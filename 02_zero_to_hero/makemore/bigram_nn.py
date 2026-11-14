import torch

from pathlib import Path
import torch.nn.functional as F

words = (Path(__file__).parent / "names.txt").read_text().splitlines()

## create the training set of bigrams (x,y)
xs, ys = [], []
chars = sorted(list(set("".join(words))))
stoi = { c: i + 1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {s: i for i, s in stoi.items()}

for word in words[:1]:
    seq = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(seq, seq[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# print(xs, ys)
"""
'.'  编号 0  →  [1, 0, 0, 0, ..., 0]    长度 27
'a'  编号 1  →  [0, 1, 0, 0, ..., 0]
'e'  编号 5  →  [0, 0, 0, 0, 0, 1, ..., 0]

为什么不直接把编号 5 喂给网络

如果直接输入数字 5，网络会把它当成一个大小：5 比 1 大，"e" 好像是 “a” 的 5 倍，"e" 和 “f” 比 “e” 和 “z” 更接近。这些关系毫无意义，字母只是类别，没有大小和远近。

One-hot 让每个字母占一个独立的位置，彼此之间没有大小关系，网络就不会学到这种虚假的规律。
"""

# 这里仅仅是为了和karpathy 视频中大家看到的结果一致 实际上是不需要的
# g = torch.Generator().manual_seed(2147483647 + 1)
# W = torch.randn((27, 27), generator=g, requires_grad=True)
# firing rate of 13th neron looking at the 3rd input 
# # print((x_encoded @ W)[3, 13])
# x_encoded = F.one_hot(xs, num_classes=27 ).float()
# log_counts = x_encoded @ W # 这里只有一层神经网络

# # 这两行实际上就是softmax:
# counts = log_counts.exp() # equivalent to N in counting module 
# probs = counts / counts.sum(dim=1, keepdim= True)

# print(probs, probs[0].sum(), probs.shape)

# nlls = torch.zeros(5)
# for i in range(5):
#   # i-th bigram:
#   x = xs[i].item() # input character index
#   y = ys[i].item() # label character index
#   print('--------')
#   print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
#   print('input to the neural net:', x)
#   print('output probabilities from the neural net:', probs[i])
#   print('label (actual next character):', y)
#   p = probs[i, y]
#   print('probability assigned by the net to the the correct character:', p.item())
#   logp = torch.log(p)
#   print('log likelihood:', logp.item())
#   nll = -logp
#   print('negative log likelihood:', nll.item())
#   nlls[i] = nll

# print('=========')
# print('average negative log likelihood, i.e. loss =', nlls.mean().item())

# ys 是目标 所以我们感兴趣的是 probs[0, ys[0]] probs[1, ys[1]] probs[2, ys[2]] probs[3, ys[3]] ... 
# 0,1,2,3,4.... => torch.arange(5)
# forward
g = torch.Generator().manual_seed(2147483647 + 1)
w = torch.randn(27,27, generator=g, requires_grad=True)
x_encoded = F.one_hot(xs, num_classes=27).float()

#soft max
counts = (x_encoded @ w).exp()
p = counts / counts.sum(dim=1, keepdim=True)

loss = -p[torch.arange(5), ys].log().mean()

print(f"loss: {loss}")
# back pass
# 梯度为 1 的只有最终的 loss（起点），不是 Wx+b 这些中间结果：
# - loss.backward() 先把 loss 自己的梯度设为 1（∂loss/∂loss = 1），
#   和 micrograd backward() 里的 self.grad = 1.0 是同一回事
# - 然后用链式法则往回传：probs、counts、x@W 的梯度都是一路乘出来的，一般不是 1，
#   最后得到我们要的 ∂loss/∂W
#
# 为什么只有 W 要重置：看谁每轮新建、谁一直存在
# - x@W、counts、probs、loss：每次前向传播都重新算出新张量，本来就没有旧梯度，不用重置
# - W：同一个张量从头用到尾，需要重置
# - backward() 对梯度是累加（+=）不是覆盖，不清空的话第二轮会加在第一轮上，方向就错了
#   （和 micrograd 里"为什么只用重置 layer 的参数"是同一个道理）
#
# 小补充：
# - W.grad = None 和设成 0 效果一样，但更省事，下次累加时 PyTorch 把 None 当 0
# - PyTorch 默认不保存中间张量的 .grad（访问是 None），只有 requires_grad=True 的
#   叶子张量（如 W）才保存；想看中间结果的梯度要先调用 .retain_grad()
w.grad = None # reset to zero
loss.backward()
print(w.grad)