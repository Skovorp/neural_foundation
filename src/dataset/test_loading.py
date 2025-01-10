from safetensors import safe_open
from safetensors.torch import save_file
import torch

def save_tsr(x, pth):
    save_file(x, pth)

def load_tsr(pth):
    tensors = {}
    with safe_open(pth, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

save_tsr({'a': torch.zeros(10), 'b': torch.zeros(10)}, 'hui.pt')
print(load_tsr('hui.pt'))