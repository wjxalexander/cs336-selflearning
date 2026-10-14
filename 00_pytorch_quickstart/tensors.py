import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

x = torch.rand(5, 3)
print(x)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

"""
tensor.T 是转置，行列互换，4×4 转置后还是 4×4
@ 是矩阵乘法运算符：第 i 行和第 j 列逐元素相乘再求和
结果 y1 是 4×4，每个元素都是 1·1 + 0·0 + 1·1 + 1·1 = 3
@ 本质上就是调用 matmul，只是写法不同
"""
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print("y2", y2)
# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

"""
实际什么时候用

最常见的场景是训练时记录 loss：

python
loss = loss_fn(pred, y)    # loss 是张量，在 GPU 上，还连着计算图
print(loss.item())         # 取出数值打印
losses.append(loss.item()) # 存进列表画曲线

如果直接 losses.append(loss) 存张量，会把整个计算图也留在内存里，训练越久内存占用越大，这是新手常见的坑。

注意
只能用于一个元素的张量，tensor.item() 对 4×4 的张量会直接报错
想把多个元素的张量变成 Python 数据，用 .tolist()：tensor.tolist() 得到嵌套列表
在 GPU 上，.item() 会让 CPU 等 GPU 算完再取值，循环里每一步都调用会稍微拖慢速度，所以一般每隔若干步打印一次就行
"""
print("z2", z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)