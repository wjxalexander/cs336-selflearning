from pathlib import Path
import torch
import matplotlib.pyplot as plt

words = (Path(__file__).parent / 'names.txt').read_text().splitlines()


SEP = "."
# why 27 : a-z + SEP tag
SIZE = 27

N  = torch.zeros((SIZE, SIZE), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = { c:i + 1 for i,c in enumerate(chars) }

stoi[SEP] = 0

for word in words:
    chars = [SEP] + list(word) + [SEP]
    for ch1, ch2 in zip(chars, chars[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        N[idx1][idx2] += 1

# plt.imshow(N)
# plt.show()

itos = {s:i for i,s in stoi.items()}

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

# probability 
p = N[0].float()
p = p / p.sum()

print(p, p.sum())
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
g = torch.Generator().manual_seed(2147483647)
# torch.rand：生成 [0, 1) 均匀分布的随机数（含 0 不含 1），这里造一个假的概率分布来演示抽样
# 相关函数：torch.randn 是标准正态分布（可为负，后面初始化权重会用）；
#   torch.randint(low, high, size) 是 [low, high) 的随机整数
p = torch.rand(3, generator=g)
p = p / p.sum()  # 归一化 -> [0.6064, 0.3033, 0.0903]

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
ix = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
print(ix)
