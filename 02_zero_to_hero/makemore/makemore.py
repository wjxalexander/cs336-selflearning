from pathlib import Path
import torch
import matplotlib.pyplot as plt

words = (Path(__file__).parent / "names.txt").read_text().splitlines()


SEP = "."
# why 27 : a-z + SEP tag
SIZE = 27

N = torch.zeros((SIZE, SIZE), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}

stoi[SEP] = 0

for word in words:
    chars = [SEP] + list(word) + [SEP]
    for ch1, ch2 in zip(chars, chars[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        N[idx1][idx2] += 1

# plt.imshow(N)
# plt.show()

itos = {s: i for i, s in stoi.items()}
print("ittt", itos[0])
# print(itos)
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(SIZE):
#     for j in range(SIZE):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         # N is torch tensor use item()
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off');
# plt.show()

# # probability
# p = N[0].float()
# p = p / p.sum()

# print(p, p.sum())
# 随机数生成器：固定种子，让每次运行抽到的随机数都一样（结果可复现）
# - torch.Generator()：新建一个独立的生成器，有自己的状态，不影响全局随机状态
# - .manual_seed(...)：设定种子，并返回生成器本身，所以能链式写成一行
# - 2147483647 = 2^31 - 1（int32 最大值），没有特殊含义，只是 Karpathy 选的数；
#   换成别的数也行，只是抽出来的名字会和视频里不一样
# 用法：传给随机函数，例如
#   torch.rand(3, generator=g)
#   torch.multinomial(p, num_samples=1, replacement=True, generator=g)
# 和 torch.manual_seed(...) 的区别：后者设置全局种子，影响所有没传 generator 的随机操作；
#   用自己的 g 只控制传了 generator=g 的地方
# 注意：每用一次 g，内部状态就往前走一步。可复现的前提是从设种子开始，
#   调用顺序和次数完全一致；想从头再来就重新 manual_seed 一次
# g = torch.Generator().manual_seed(2147483647)
# torch.rand：生成 [0, 1) 均匀分布的随机数（含 0 不含 1），这里造一个假的概率分布来演示抽样
# 相关函数：torch.randn 是标准正态分布（可为负，后面初始化权重会用）；
# #   torch.randint(low, high, size) 是 [low, high) 的随机整数
# p = torch.rand(3, generator=g)
# p = p / p.sum()  # 归一化 -> [0.6064, 0.3033, 0.0903]

# torch.multinomial：把 p 当成一个"有偏的骰子"扔 num_samples 次，返回每次抽中的下标
# - p：每个下标的权重（不要求和为 1，内部会自动归一化）
# - num_samples=20：抽 20 次
# - replacement=True：有放回，同一个下标可以重复抽到；
#   False 则每个下标最多抽一次，num_samples 不能超过 len(p)
# - generator=g：用自己的生成器，结果可复现
# 输出：tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
#   0/1/2 分别出现 9/9/2 次，不是正好 60%/30%/10%：20 次样本太少，
#   随机波动大；num_samples 改成 10000 就会很接近
# 生成名字时会写成：
#   ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#   p 换成 N 某一行归一化后的结果（当前字母后面跟各字母的概率），
#   .item() 把单元素张量变成 int，再用 itos[ix] 变回字母，循环到抽中 '.' 为止
# ix = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
# print(ix)


# q = N[0].float()
# q = q / q.sum()
# h = torch.Generator().manual_seed(2147483647)
# id = torch.multinomial(q, num_samples=1, replacement=True, generator=h)

# print(itos[id.item()])

P = N.float()
sum_by_row = P.sum(dim=1, keepdim=True)
print(sum_by_row.shape)

P = P/sum_by_row
# must be 1 normalized 不是1 就错啦

print(P[1].sum())
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    idx = 0
    out = []
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break

    # print("".join(out))


log_likelihood = 0.0
n = 0
for word in words[:3]:
    str = [SEP] + list(word) + [SEP]
    for ch1, ch2 in zip(str, str[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        p = P[idx1][idx2]
        log_prob = torch.log(p)

        # 为什么是加不是乘：log(a*b) = log(a) + log(b)
        # - 似然 = 所有 bigram 概率的连乘：P(e|.) * P(m|e) * P(m|m) * ...
        # - 取对数后连乘变成求和，所以对 log_prob 用 +=（初始 0.0）；
        #   如果累加原始概率 p，就要用 *=（初始 1.0），两者数学上等价
        # 为什么要取对数：
        # 1. 防下溢：概率都在 0~1 之间，几十万个连乘会小到浮点数表示不了，变成 0.0；
        #    取对数后只是一堆负数相加，数值稳定
        # 2. 好求导：后面 loss = 负对数似然的平均值（-log_likelihood / n），求和比连乘好求导
        log_likelihood += log_prob
        n+=1
        print(f"{ch1}{ch2}", p, log_prob)

# nll（negative log likelihood，负对数似然）= -log_likelihood
# - log_likelihood 是一堆 log(p) 相加，p 在 0~1 之间，所以它总是 <= 0，越接近 0 越好
# - 取负号变成 >= 0，变成"越小越好"，符合 loss 的习惯（优化 = 让 loss 变小）
# - 最好是 0（每个 bigram 都被预测为概率 1）；概率越小，-log(p) 越大
#
# nll / n：平均负对数似然，就是我们要最小化的 loss
# - 不除以 n 的话，nll 会随样本数增长（数据越多 nll 越大），没法在不同数据量之间比较
# - 除以 n 后表示"平均每个 bigram 的 loss"，与数据量无关
# - 参考：均匀乱猜时每个字母概率 1/27，平均 loss = -log(1/27) ≈ 3.2958；
#   低于它说明模型学到了东西
# 当前输出（前 3 个名字 emma/olivia/ava，共 n=16 个 bigram）：
#   tensor(38.7856) tensor(2.4241)
nll = -log_likelihood
print(nll, nll / n)

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)