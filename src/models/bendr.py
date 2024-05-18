import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.training_utils import make_pretrain_mask
from models.neural_gpt import BaseModel
import math
# from torchtune.modules import RotaryPositionalEmbeddings

class LearnedPositionalEncoding(nn.Module):

    def __init__(self, emb_dim, max_len = 500):
        super().__init__()
        self.embeddings = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        inds = torch.arange(x.size(1), device=x.device, dtype=torch.long)
        x = x + self.embeddings(inds)
        return x


class Encoder(BaseModel):
    def __init__(self, inp_size, emb_dim, **kwargs):
        super().__init__()
        self.proj = nn.Linear(inp_size, emb_dim)

    def forward(self, batch):
        x = batch['data'].clone()
        # x -- (batch, chunks, channels, time)
        batch_size, num_chunks, channels, time = x.shape
        x = x.view(batch_size, num_chunks, -1)
        x = self.proj(x)
        batch['encoder_features'] = x
        return batch
    
    
class ContextNetwork(BaseModel):
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
        self.target_proj = nn.Identity() # nn.Linear(emb_dim, emb_dim)
        self.positional_emb = LearnedPositionalEncoding(emb_dim)

    def forward(self, batch):
        x = batch['encoder_features'].clone()
        batch_size, num_chunks, emb_dim = x.shape
        mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length)
        x[mask] = self.mask_emb
        assert ((x[0, :, 0].detach().cpu() == self.mask_emb[0].detach().cpu()) == mask[0, :].detach().cpu()).all(), "masking failed :(" 
        assert not (batch['encoder_features'] == x).all(), "Inplace operations on encoder_features are HARAM!"
        x = self.positional_emb(x)
        x = self.transformer_encoder(x)
        batch['mask'] = mask
        batch['context_vectors'] = x
        batch['targets'] = self.target_proj(batch['encoder_features'])
        return batch
    
    def avg_part_masked(self, batch):
        batch_size, num_chunks, _, _ = batch.shape
        res = 0
        for _ in range(100):
            mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length)
            res += (mask * 1.0).mean().item() # true is mask
        return res / 100
    
def calc_loss(batch, log_temp):
    batch_size, num_tokens, emb_size = batch['targets'].shape
    targets, preds = batch['targets'].clone(), batch['context_vectors'].clone()
    norm_targets = torch.norm(targets, 2, dim=2, keepdim=True) # batch_size, num_tokens
    norm_preds = torch.norm(preds, 2, dim=2, keepdim=True) # batch_size, num_tokens
    
    targets = targets / norm_targets
    preds = preds / norm_preds
    
    # targets = torch.cat([targets, 100 * torch.ones(batch_size, 5, emb_size, device=batch['targets'].device)], dim=1)
    
    sim = preds @ targets.permute(0, 2, 1) # batch_size, num_tokens, num_tokens
    sim = sim[batch['mask']] # num_masked, num_tokens -- for every masked prediction, logits  per all seq
    labels = torch.tile(torch.arange(num_tokens), (batch_size, 1))
    labels = labels[batch['mask']].to(batch['targets'].device)
    sim = sim * math.exp(log_temp)
    unreduced_loss = F.cross_entropy(sim, labels, reduction='none')
    
    # set batch=1, num_chunks=8 to see how things works
    # print(sim[:batch['mask'][0].sum()])
    # print(batch['mask'][0])
    # print(labels[:batch['mask'][0].sum()])
    # print(F.cross_entropy(sim, labels, reduction='none'))
    # example of good vals
    # tensor([[ 2.7511, -0.5308, -0.3625, -0.2960, -0.2756, -0.3741, -0.3812, -0.5015],
    #     [-0.6111,  2.7569, -0.2216, -0.6170, -0.3746, -0.5203, -0.1013, -0.3689],
    #     [-0.1247, -0.1079, -0.2098, -0.2186,  2.7137, -0.1079, -0.1882, -0.3002],
    #     [-0.0668, -0.6090, -0.4044, -0.5796, -0.3214,  2.5963, -0.1835, -0.6229],
    #     [-0.3907, -0.5906, -0.5988, -0.4737, -0.7476, -0.6327,  2.7852, -0.6409],
    #     [-0.3811, -0.4140, -0.5004, -0.3402, -0.3983, -0.5376, -0.3075,  2.6888]],
    #    device='cuda:0', grad_fn=<MulBackward0>)
    # tensor([[ True,  True, False, False,  True,  True,  True,  True]])
    # tensor([0, 1, 4, 5, 6, 7], device='cuda:0')
    # loss: 0.27600 -- close to orthogonal optimum
    
    batch['per_masktoken_loss'] = unreduced_loss
    batch['loss'] = unreduced_loss.mean()
    
    return batch
