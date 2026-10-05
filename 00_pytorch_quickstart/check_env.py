import platform

import torch

print(f"python : {platform.python_version()}")
print(f"torch  : {torch.__version__}")
print(f"cuda   : {torch.cuda.is_available()}")
print(f"mps    : {torch.backends.mps.is_available()}")

x = torch.rand(5, 3)
print(x)
