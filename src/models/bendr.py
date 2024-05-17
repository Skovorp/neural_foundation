import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.training_utils import make_pretrain_mask
from models.neural_gpt import BaseModel


class Encoder(BaseModel):
    def __init__(self, inp_size, emb_dim, **kwargs):
        super().__init__()
        self.proj = nn.Linear(inp_size, emb_dim)

    def forward(self, batch):
        x = batch['data']
        # x -- (batch, chunks, channels, time)
        batch_size, num_chunks, channels, time = x.shape
        x = x.view(batch_size, num_chunks, -1)
        x = self.proj(x)
        batch['encoder_features'] = x
        return batch
    
    
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
        self.target_proj = nn.Identity() # nn.Linear(emb_dim, emb_dim)

    def forward(self, batch):
        x = batch['encoder_features']
        batch_size, num_chunks, emb_dim = x.shape
        mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length)
        x[mask] = self.mask_emb
        assert ((x[0, :, 0].detach().cpu() == self.mask_emb[0].detach().cpu()) == mask[0, :].detach().cpu()).all(), "masking failed :("
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
    
def calc_loss(batch):
    batch_size, num_tokens, emb_size = batch['targets'].shape
    targets, preds = batch['targets'], batch['context_vectors']
    norm_targets = torch.norm(targets, 2, dim=2, keepdim=True) # batch_size, num_tokens
    norm_preds = torch.norm(preds, 2, dim=2, keepdim=True) # batch_size, num_tokens
    
    targets = targets / norm_targets
    preds = preds / norm_preds
    
    # targets = torch.cat([targets, 100 * torch.ones(batch_size, 5, emb_size, device=batch['targets'].device)], dim=1) 
    
    sim = preds @ targets.permute(0, 2, 1) # batch_size, num_tokens, num_tokens
    sim = sim[batch['mask']] # num_masked, num_tokens -- for every masked prediction, logits  per all seq
    labels = torch.tile(torch.arange(num_tokens), (batch_size, 1))
    labels = labels[batch['mask']].to(batch['targets'].device)
    loss = F.cross_entropy(sim, labels)
    
    return loss
