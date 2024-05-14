import torch
from torch import nn
import numpy as np
from utils import make_pretrain_mask


class Encoder(nn.Module):
    def __init__(self, inp_size, emb_dim, **kwargs):
        super().__init__()
        self.proj = nn.Linear(inp_size, emb_dim)

    def forward(self, x):
        # x -- (batch, chunks, channels, time)
        batch_size, num_chunks, channels, time = x.shape
        x = x.view(batch_size, num_chunks, -1)
        x = self.proj(x)
        return x
    
    
class ContextNetwork(nn.Module):
    def __init__(self, emb_dim, ffn_dim, nhead, transformer_num_layers, mask_prob, mask_length, **kwargs):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.randn(emb_dim))
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            dim_feedforward=ffn_dim, 
            nhead=nhead, 
            norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_num_layers, enable_nested_tensor=False)

    def forward(self, x):
        batch_size, num_chunks, emb_dim = x.shape
        mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length)
        x[mask] = self.mask_emb
        assert ((x[0, :, 0].detach().cpu() == self.mask_emb[0].detach().cpu()) == mask[0, :].detach().cpu()).all(), "masking failed :("
        x = self.transformer_encoder(x)
        return x
    
    def avg_part_masked(self, batch):
        batch_size, num_chunks, _, _ = batch.shape
        res = 0
        for _ in range(100):
            mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length)
            res += (mask * 1.0).mean().item() # true is mask
        return res / 100