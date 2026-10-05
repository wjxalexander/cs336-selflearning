import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
# img, label = training_data[0]
# print(img.shape, img.dtype, img.min(), img.max(), label)
# torch.Size([1, 28, 28]) torch.float32 tensor(0.) tensor(1.) 9

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

"""
简单说X 是题目，y 是答案。
X 里装的是 64 张图片的像素，每张图是 1×28×28 个 0–1 之间的小数。这是模型的输入，也就是模型要"看"的东西。

y 里装的是这 64 张图各自的正确类别，每个只是一个 0–9 的整数，比如 9 代表短靴，1 代表裤子。这是模型要"猜对"的东西。

两者按位置一一对应：X[0] 这张图的答案就是 y[0]，X[1] 的答案是 y[1]，依此类推。
"""
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

"""Creating Models"""
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
"""
pred = model(X)            # 前向：猜
loss = loss_fn(pred, y)    # 算错了多少
loss.backward()            # 反向传播：算出每个参数的梯度
optimizer.step()           # 按梯度更新参数
optimizer.zero_grad()      # 清空梯度，准备下一批
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
         # [64, 1, 28, 28]  →  Flatten  →  [64, 784]
         # 只把每张图自己的 1×28×28 压成 784。
         # https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#nn-flatten
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #之所以要拉平，是因为后面的 nn.Linear(28*28, 512) 
            # 只接受一维向量作为每个样本的输入，要求每个样本正好是 784 个数。
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork().to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

"""train 的 loss 和 test 的 loss 有什么区别

train 的 loss：在模型正在学习的数据上算的，反映"学得怎么样"。
test 的 loss：在模型从没见过的数据上算的，反映"学到的东西能不能用到新数据上"。

两个对比着看很有用。如果 train loss 一直降，test loss 却开始上升，说明模型在死记训练数据，而不是学到了规律，这就是过拟合。
"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')