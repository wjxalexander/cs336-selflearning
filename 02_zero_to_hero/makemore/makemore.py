from pathlib import Path
import torch
import matplotlib.pyplot as plt

words = (Path(__file__).parent / 'names.txt').read_text().splitlines()


START_TAG = "<S>"
END_TAG = "<E>"
# why 28 : a-z + start tag + end tag
N  = torch.zeros((28, 28), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = { c:i for i,c in enumerate(chars) }

stoi[START_TAG] = 26
stoi[END_TAG] = 27 

for word in words:
    chars = [START_TAG] + list(word) + [END_TAG]
    for ch1, ch2 in zip(chars, chars[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        N[idx1][idx2] += 1

# plt.imshow(N)
# plt.show()

itos = {s:i for i,s in stoi.items()}

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
plt.show()