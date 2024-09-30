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

    def forward(self, x):
        inds = torch.arange(x.size(1), device=x.device, dtype=torch.long)
        x = x + self.embeddings(inds)
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
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_num_layers, enable_nested_tensor=False)
        self.target_proj = nn.Linear(emb_dim, emb_dim) 
        
        if pe['type'] == "conv":
            self.positional_emb = ConvPositionalEncoding(emb_dim, **pe)
        elif pe['type'] == "learned":
            self.positional_emb = LearnedPositionalEncoding(emb_dim, **pe)
        else:
            assert False, "bad pe['type']"

    def forward(self, batch):
        x = batch['encoder_features'].clone()
        batch_size, num_chunks, emb_dim = x.shape
        mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length, self.min_masked)
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
            mask = make_pretrain_mask(batch_size, num_chunks, self.mask_prob, self.mask_length, self.min_masked)
            res += (mask * 1.0).mean().item() # true is mask
        return res / 100
    
# Loss code is hard. Bendr probably has a few errors. Sources are here 
# BENDR: https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py#L271
# Wav2vec2: https://github.com/facebookresearch/fairseq/blob/920a548ca770fb1a951f7f4289b4d3a0c1bc226f/fairseq/models/wav2vec/wav2vec2.py#L499


def pick_negatives(x, num_negatives):
    # for each el in seq_len get num_negatives other els from the same batch
    batch_size, seq_len, emb_size = x.shape 
    
    self_indexes = torch.arange(seq_len, device=x.device).unsqueeze(-1).expand(-1, num_negatives) # (seq_len, num_negatives)

    neg_idxs = torch.randint(low=0, high=seq_len - 1, size=(batch_size, seq_len, num_negatives), device=x.device)
                             
    neg_idxs[neg_idxs >= self_indexes.unsqueeze(0)] += 1
    neg_idxs = neg_idxs.view(batch_size, seq_len * num_negatives, 1).expand(batch_size, seq_len * num_negatives, emb_size)

    res = torch.gather(x, 1, neg_idxs).view(batch_size, seq_len, num_negatives, emb_size)
    return res

def calc_loss_proper(batch, temp, num_negatives):
    features, context, mask = batch['targets'], batch['context_vectors'], batch['mask']
    batch_size, _, emb_size = features.shape
    
    masked_context = context[mask].view(batch_size, -1, emb_size)
    masked_features = features[mask].view(batch_size, -1, emb_size)
    
    num_masked = masked_features.size(1)
    
    negs = pick_negatives(masked_features, num_negatives) # (batch_size, num_masked, num_negatives, emb_size)
    sim_targets = torch.cat([masked_features.unsqueeze(2), negs], dim=2) # (batch_size, num_masked, 1 + num_negatives, emb_size)
    
    sims_raw = F.cosine_similarity(masked_context.unsqueeze(2), sim_targets, dim=3) # (batch_size, num_masked, 1 + num_negatives)

    # дальше тут надо -inf втыкать какогото хуя если негатив совпадет с таргетом идеально, ебал рот эту хуйню пока
    sims = sims_raw / temp
    sims = sims.permute(0, 2, 1) # (batch_size, 1 + num_negatives, num_masked)
    
    ce_target = torch.zeros(batch_size, num_masked, dtype=torch.long, device=sims.device) 
    unreduced_loss = F.cross_entropy(sims, ce_target, reduction='none')
    
    batch['per_masktoken_loss'] = unreduced_loss
    
    # avg = batch['targets'].mean((0, 1), keepdim=True)
    # avg = batch['targets'].mean(1, keepdim=True)
    # penalty = ((batch['targets'] - avg) ** 2).mean()
    # penalty = (batch['targets'] ** 2).mean()
    
    batch['loss'] = unreduced_loss.mean() # + calc_self_entropy(batch['targets'].clone())
    
    
    with torch.no_grad():
        assert sims.argmax(1).shape == (batch_size, num_masked)
        batch['acc_feature_choice'] = ((sims.argmax(1) == 0) * 1.0).mean()
        batch['mean_correct_sim'] = sims_raw[:, :, 0].mean()
        batch['mean_destractor_sim'] = sims_raw[:, :, 1:].mean()
    
    return batch

# WHY switching to identity worsened results?
# smaller variance? targets are too close to eachother? its not like sound?


def calc_self_entropy(x):
    x = x / torch.norm(x, dim=1, keepdim=True)
    bs, n, _ = x.shape
    sim = x @ x.transpose(1, 2)
    diag_inf = torch.diagflat(torch.tile(torch.tensor(torch.inf, device=x.device), [n])).unsqueeze(0)
    q = torch.nn.functional.softmax(sim - diag_inf, dim=1) 
    return (torch.log(q + 1e-5) * q).mean()