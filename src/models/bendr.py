import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.training_utils import make_pretrain_mask
from models.neural_gpt import BaseModel
import math
from utils.training_utils import emb_std
# from torchtune.modules import RotaryPositionalEmbeddings

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len, **kwargs):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(self, x):
        inds = torch.arange(x.size(1), device=x.device, dtype=torch.long)
        x = x + self.embeddings(inds).unsqueeze(0)
        x = self.norm(x)
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len, **kwargs):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = nn.Parameter(pe, requires_grad=False)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(self, x):
        # x (batch_size, seq_len, emb_dim)
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        x = self.norm(x)
        return x
    

class ConvPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, kernel_size, groups, **kwargs):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        
    def forward(self, x):
        # x is (batch, seq_len, emb_dim)
        x = x.permute(0, 2, 1) # (batch, channels, time)
        x = x + self.conv(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, emb_dim)
        x = self.norm(x)
        return x


class EncoderLinear(BaseModel):
    def __init__(self, inp_size, emb_dim, **kwargs):
        super().__init__()
        self.proj = nn.Linear(inp_size, emb_dim)

    def forward(self, batch):
        data = batch['data']
        # x -- (batch, chunks, channels, time)
        batch_size, num_chunks, channels, time = data.shape
        x = data.view(batch_size, num_chunks, -1)
        x = self.proj(x)
        batch['encoder_features'] = x
        return batch
    
class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-2, -1)
    
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L844
class EncoderConv(BaseModel):
    def __init__(self, emb_dim, norm, **kwargs):
        super().__init__()
        assert norm in ('group', 'layer'), 'norm should be group or layer'
        self.norm = norm
        self.stack = nn.Sequential(
            self.make_block(in_channels=4,       out_channels=emb_dim, kernel_size=3, stride=3),
            self.make_block(in_channels=emb_dim, out_channels=emb_dim, kernel_size=2, stride=2),
            self.make_block(in_channels=emb_dim, out_channels=emb_dim, kernel_size=2, stride=2),
            self.make_block(in_channels=emb_dim, out_channels=emb_dim, kernel_size=2, stride=2),
            self.make_block(in_channels=emb_dim, out_channels=emb_dim, kernel_size=2, stride=2),
            self.make_block(in_channels=emb_dim, out_channels=emb_dim, kernel_size=2, stride=2),
        )
    
    def make_block(self, in_channels, out_channels, kernel_size, stride):
        if self.norm == "group":
            return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.GELU(),
        )  
        elif self.norm == "layer":
            # Somehow this doesnt work at all. I checked wav2vec implementation 10 times, it's the same
            # In paper they say layernorm, but "default" mode in repo uses groupnorm and only for the first layer
            # I cant figure it out, lets just use groupnorm
            return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.Sequential(
                TransposeLast(),
                nn.LayerNorm(out_channels), # normalises each embedding individually
                TransposeLast(), 
            ),
            nn.GELU(),
        )
        assert False
         
        

    def forward(self, batch):
        x = batch['data'] # (batch, channels, time)
        x = self.stack(x) # (batch, more_channels, less_time)
        x = x.permute(0, 2, 1) # (batch, num_chunks, emb_dim)
        batch['encoder_features'] = x
        return batch
    
    
class MyIdentity(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return x.clone()
    
class ContextNetwork(BaseModel):
    def __init__(self, emb_dim, ffn_dim, nhead, transformer_num_layers, mask_prob, mask_length, min_masked, pe, **kwargs):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.randn(emb_dim))
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masked = min_masked
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            dim_feedforward=ffn_dim, 
            nhead=nhead, 
            norm_first=True,
            batch_first=True,
            activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_num_layers, enable_nested_tensor=False)
        self.target_proj = nn.Linear(emb_dim, emb_dim) 
        
        if pe['type'] == "conv":
            self.positional_emb = ConvPositionalEncoding(emb_dim, **pe)
        elif pe['type'] == "learned":
            self.positional_emb = LearnedPositionalEncoding(emb_dim, **pe)
        elif pe['type'] == 'sin':
            self.positional_emb = SinusoidalPositionalEncoding(emb_dim, **pe)
        else:
            assert False, "bad pe['type']"

    def forward(self, batch, run_full=False):
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # with pass:
        batch_size, num_chunks, emb_dim = batch['encoder_features'].shape
        if run_full:
            x = self.positional_emb(batch['encoder_features'])
            assert not (batch['encoder_features'] == x).all(), "Inplace operations on encoder_features are HARAM!" # THIS IS COMPUTE INTENSIVE!
            x = self.transformer_encoder(x)
            batch['full_context_vectors'] = x
            return batch
        x = batch['encoder_features'].clone()
        mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length, self.min_masked, x.device)
        x[mask] = self.mask_emb
        batch['mask'] = mask
        # assert ((x[0, :, 0].detach().cpu() == self.mask_emb[0].detach().cpu()) == mask[0, :].detach().cpu()).all(), "masking failed :("  # THIS IS COMPUTE INTENSIVE!
        # assert not (batch['encoder_features'] == x).all(), "Inplace operations on encoder_features are HARAM!" # THIS IS COMPUTE INTENSIVE!
        x = self.positional_emb(x)
        x = self.transformer_encoder(x)
        batch['context_vectors'] = x
        batch['targets'] = self.target_proj(batch['encoder_features'])
        return batch
    
    def avg_part_masked(self, batch):
        batch_size, num_chunks, _, _ = batch.shape
        res = 0
        for _ in range(100):
            mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length, self.min_masked)
            res += (mask * 1.0).mean().item() # true is mask
        return res / 100
