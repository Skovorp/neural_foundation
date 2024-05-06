import torch

device = torch.device('cuda')
t = torch.rand(10000, device=device)
print(t.mean())