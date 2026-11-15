import torch

from pathlib import Path
import torch.nn.functional as F

words = (Path(__file__).parent / "names.txt").read_text().splitlines()

## create the training set of bigrams (x,y)
xs, ys = [], []
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {s: i for i, s in stoi.items()}

for word in words:
    seq = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(seq, seq[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)
print(xs, ys)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
# nelement()：张量里元素的总个数，这里就是 bigram 训练样本的数量（全量数据约 22.8 万）
# - 用它代替写死的 5，这样 words[:1] 换成全部 words 后代码不用改
# - 和 xs.numel() 完全等价；一维张量时也等于 len(xs)，
#   多维时不同：3x4 的张量 nelement() 是 12，len() 是 3（只看第一维）
num = xs.nelement()
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

# 下面是完整的训练循环：forward -> loss -> backward -> update，重复 100 轮

# 固定随机种子只是为了结果可复现，实际上不需要
g = torch.Generator().manual_seed(2147483647 + 1)
# requires_grad=True：告诉 PyTorch 要对 w 求梯度，否则 loss.backward() 会报错
w = torch.randn(27, 27, generator=g, requires_grad=True)
for k in range(100):
    # forward
    x_encoded = F.one_hot(xs, num_classes=27).float()

    # softmax：把任意实数（logits）变成概率分布
    # - x_encoded @ w 是 logits（log counts），可正可负
    # - exp() 之后全为正数，相当于计数模型里的 N
    # - 每行除以行和，得到每行加起来等于 1 的概率
    counts = (x_encoded @ w).exp()
    p = counts / counts.sum(dim=1, keepdim=True)

    # loss = 负对数似然（NLL）+ 正则项
    #
    # 第一项 -p[torch.arange(num), ys].log().mean()：负对数似然
    # - p[torch.arange(num), ys]：整数数组索引（fancy indexing），两个索引一一配对，
    #   取出 p[0, ys[0]]、p[1, ys[1]]、...，即模型给每个正确字符的概率
    # - 为什么先 log 再求平均，而不是直接把概率相加：
    #   要最大化的是似然 = 所有概率的乘积，log 把连乘变成连加
    #   （和 bigram_counting.py 里的 log_likelihood 一样），求和就藏在 mean() 里。
    #   直接相加概率的话，一个猜得很离谱的样本（p 接近 0）几乎不受惩罚；
    #   取 log 后 p -> 0 时 log(p) -> -inf，会被重罚
    # - 负号：log(p) <= 0，越大越好；加负号变成越小越好的 loss
    # - softmax + NLL 合起来就是交叉熵（cross-entropy），这一项等价于
    #   F.cross_entropy(x_encoded @ w, ys)
    #   注意传入的是 softmax 之前的 logits；它数值上更稳定，实际训练都用它
    #
    # 第二项 0.01 * (w ** 2).mean()：L2 正则化（weight decay）
    # - 正则化 = 在 loss 里加一项惩罚，不让模型为了拟合训练数据而走极端（过拟合）
    # - 这里惩罚的是权重的大小，把 w 往 0 拉；
    #   w 全为 0 时 exp(0) = 1，每行都是均匀分布 1/27
    # - 作用等价于 bigram_counting.py 里从 1 开始计数的平滑：
    #   防止模型把没见过的 bigram（如 "jq"）的概率压到接近 0
    # - 为什么能防止：要让某个概率接近 0，它的 logit 必须比同一行其他的低很多，
    #   也就是 w 里要有绝对值很大的数，而正则项惩罚的正是这一点。
    #   把概率压到 0 这件事，数据项上的收益极小，正则项上的代价很大，不划算。
    #
    #   用缩小版的问题看（只有 a、o、q 三个字符，"j" 后面出现过 9 次 a、1 次 o、0 次 q，
    #   w 里 "j" 那一行就是 3 个数，下面的数字是实际跑出来的）：
    #
    #   例子 1：比较两组权重的"总账"（总 loss = 数据项 + 正则项）
    #     w (a, o, q)        q 的概率        数据项    正则项    总 loss
    #     A: 2.2, 0, -20     0.0000000002    0.3251    1.3495    1.6745
    #     B: 2.2, 0, -3      0.005           0.3300    0.0461    0.3762
    #   从 B 到 A，数据项只好了 0.005（数据里本来就没有 q，只是多腾出一点点概率给 a、o），
    #   正则项却从 0.05 涨到 1.35（-20 的平方是 400）。
    #   不加正则只看数据项，模型选 A；加了正则看总 loss，模型选 B
    #
    #   例子 2：从 w = [0, 0, 0] 开始真的训练，看 q 的权重怎么走
    #     步数        不加正则: w_q    p(q)         加正则: w_q    p(q)
    #     100         -2.54            0.0065       -2.00          0.0148
    #     1,000       -4.07            0.00067      -2.05          0.0137
    #     10,000      -5.61            0.000067     -2.05          0.0137
    #     100,000     -7.14            0.0000067    -2.05          0.0137
    #   不加正则：w_q 一直往下掉没有尽头，步数每多 10 倍 p(q) 就再小 10 倍。
    #   加正则：1,000 步后停在 -2.05 不动，p(q) 稳定在约 1.4%。
    #   停下来是因为两股力抵消了：数据项把 w_q 往下推，推力正比于 p，p 越小推得越轻；
    #   正则项把 w_q 往 0 拉，拉力正比于 w，绝对值越大拉得越重。
    #   同时模型没有变笨：加正则后 a 的权重 2.12、o 是 -0.06，
    #   "a 比 o 常见得多"这个规律照样学到了
    # - 0.01 是强度：越大分布越平（越接近乱猜），越小越贴近训练数据
    # - 不要写成不带系数的 (w ** 2).sum()：729 个元素加起来远大于数据项，
    #   w 会被压到 0，模型什么都学不到
    loss = -p[torch.arange(num), ys].log().mean() + 0.01 * (w**2).mean()

    print(f"loss: {loss.item()}")
    # back pass
    # 梯度为 1 的只有最终的 loss（起点），不是 wx+b 这些中间结果：
    # - loss.backward() 先把 loss 自己的梯度设为 1（∂loss/∂loss = 1），
    #   和 micrograd backward() 里的 self.grad = 1.0 是同一回事
    # - 然后用链式法则往回传：p、counts、x@w 的梯度都是一路乘出来的，一般不是 1，
    #   最后得到我们要的 ∂loss/∂w
    #
    # 为什么只有 w 要重置：看谁每轮新建、谁一直存在
    # - x@w、counts、p、loss：每次前向传播都重新算出新张量，本来就没有旧梯度，不用重置
    # - w：同一个张量从头用到尾，需要重置
    # - backward() 对梯度是累加（+=）不是覆盖，不清空的话第二轮会加在第一轮上，方向就错了
    #   （和 micrograd 里"为什么只用重置 layer 的参数"是同一个道理）
    #
    # 小补充：
    # - w.grad = None 和设成 0 效果一样，但更省事，下次累加时 PyTorch 把 None 当 0
    # - PyTorch 默认不保存中间张量的 .grad（访问是 None），只有 requires_grad=True 的
    #   叶子张量（如 w）才保存；想看中间结果的梯度要先调用 .retain_grad()
    w.grad = None  # reset to zero
    loss.backward()
    # print(w.grad)

    # update：沿梯度的反方向走一步，0.1 是学习率
    # - 梯度指向 loss 上升最快的方向，所以要减去它
    # - 改 w.data 而不是 w：这一步只是更新数值，不应该被 autograd 记录
    w.data += -50 * w.grad

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        # ----------
        # BEFORE:
        # p = P[ix]
        # ----------
        # NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ w  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
        # ----------

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
