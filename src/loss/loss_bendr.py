import torch
import torch.nn.functional as F
import numpy as np

    
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


def calc_loss_effective(batch, temp):
    features, context, mask = batch['targets'], batch['context_vectors'], batch['mask']
    batch_size, _, emb_size = features.shape
    
    masked_context = context[mask].view(batch_size, -1, emb_size).to(torch.float32)
    masked_features = features[mask].view(batch_size, -1, emb_size).to(torch.float32)
    
    num_masked = masked_features.size(1)
    
    sims = masked_context @ masked_features.transpose(-1, -2)
    sims = sims / torch.norm(masked_context, dim=2).unsqueeze(2)
    sims = sims / torch.norm(masked_features, dim=2).unsqueeze(1) # batch_size, num_masked, num_masked
    # TODO: getting sim inf. look above how to fix it
    assert sims.min() > -1 and sims.max() < 1, f"sims.min(): {sims.min()} | sims.max(): {sims.max()}"
    sims = sims / temp
    sims = sims.permute(0, 2, 1) 
    
    ce_target = torch.arange(num_masked, device=sims.device).unsqueeze(0).expand(batch_size, num_masked)
    unreduced_loss = F.cross_entropy(sims, ce_target, reduction='none')
    
    batch['per_masktoken_loss'] = unreduced_loss
    batch['loss'] = unreduced_loss.mean()
    
    with torch.no_grad():
        sum_diag = torch.diagonal(sims, dim1=1, dim2=2).sum()
        batch['mean_correct_sim'] = temp * sum_diag / (batch_size * num_masked)
        batch['mean_destractor_sim'] = temp * (sims.sum() - sum_diag) / (batch_size * num_masked * (num_masked - 1))
        assert sims.argmax(1).shape == (batch_size, num_masked)
        corrects = torch.arange(num_masked, device=sims.device).unsqueeze(0).expand(batch_size, num_masked)
        batch['acc_feature_choice'] = ((sims.argmax(1) == corrects) * 1.0).mean()
    
    return batch
    
    
    
    
    