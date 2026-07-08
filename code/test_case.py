import torch

a = torch.tensor(10)
print(a,a.shape,a.view(1,-1))
print(a.view(1,1,-1))