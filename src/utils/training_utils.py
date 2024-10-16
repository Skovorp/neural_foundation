import numpy as np
from torch import nn
import torch
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from utils.data_utils import fig2img
import time
import pandas as pd

def benchmark_previous(encoder_res):
    """Loss is calculated as MSE(encoder_res[:, 1:, :], smt). 
    encoder_res shape is (batch_size, num_tokens, emb_dim)
    lets predict current embedding for next embedding"""
    with torch.no_grad():
        return ((encoder_res[:, 1:, :] - encoder_res[:, :-1, :]) ** 2).mean().item()


def benchmark_best_constant(encoder_res):
    """Loss is calculated as MSE(encoder_res[:, 1:, :], smt). 
    encoder_res shape is (batch_size, num_tokens, emb_dim)
    lets predict mean of embedding for each embedding"""
    with torch.no_grad():
        target = encoder_res[:, 1:, :]
        pred = target.mean(1, keepdims=True)
        return ((target - pred) ** 2).mean().item()

def benchmark_cumsum(encoder_res):
    """Loss is calculated as MSE(encoder_res[:, 1:, :], smt). 
    encoder_res shape is (batch_size, num_tokens, emb_dim)
    lets predict mean of all previous embeddings for each embedding"""
    with torch.no_grad():
        target = encoder_res[:, 1:, :]
        pred = encoder_res.cumsum(1)
        pred = pred / (torch.arange(pred.shape[1], device=encoder_res.device) + 1).reshape(1, -1, 1)
        pred = pred[:, :-1, :]
        return ((target - pred) ** 2).mean().item()


def make_pretrain_mask_legacy(batch_size, num_chunks, mask_prob, mask_length, min_masked):
    if min_masked * 2 != num_chunks:
        print(f"WARNING!!! masking not 50%. min_masked: {min_masked}, num_chunks: {num_chunks}")
    assert mask_length > 0
    
    mask = 1 * (torch.rand(batch_size, num_chunks) < mask_prob)
        
    mask[:, mask_length:] = mask[:, mask_length:] - mask[:, : -mask_length]
    mask = mask.cumsum(1) > 0
    
    # we need to have the same amount of masked tokens in each sequence to calculate loss easier
    non_zero = torch.count_nonzero(mask, dim=1)
    common_num_masked = min_masked # max(non_zero.max(), min_masked)
    # print(f"Masked tokens: {max_num_masked} / {num_chunks}")
    # if common_num_masked > min_masked:
    #     print(f"Overshoot num masked: {common_num_masked}")
    diff_num_mask = common_num_masked - non_zero # positive -> add more masks (True)
    iteration_starts = torch.randint(low=0, high=num_chunks, size=(batch_size, ))
    
    for i in range(batch_size):
        j = iteration_starts[i] # otherwise all beginning chunks are always masked
        while diff_num_mask[i] != 0:
            if diff_num_mask[i] > 0 and not mask[i, j]:
                mask[i, j] = True
                diff_num_mask[i] -= 1
            elif diff_num_mask[i] < 0 and mask[i, j]:
                mask[i, j] = False
                diff_num_mask[i] += 1
            j = (j + 1) % num_chunks
    assert (diff_num_mask == 0).all(), "Mask failed, different count of masked tokens in sequences"
    assert (mask.sum(1) == min_masked).all()
    return mask

def suggest_mask(batch_size, num_chunks, mask_prob, mask_length, device):
    mask = torch.where(torch.rand(batch_size, num_chunks, device=device) < mask_prob, 1.0, 0.0) 
    mask[:, mask_length:] = mask[:, mask_length:] - mask[:, : -mask_length]
    mask = mask.cumsum(1) > 0
    return mask


@torch.no_grad()
def make_pretrain_mask(batch_size, num_chunks, mask_prob, mask_length, min_masked, device, **kwargs):
    while True:
        masks = suggest_mask(batch_size * 256, num_chunks, mask_prob, mask_length, device) # works fast for bs=32, num_chunks=768, makes infrequent retries. on collab works faster than x256
        ok_masks = masks.int().sum(axis=1) == min_masked
        res = masks[ok_masks, :][:batch_size, :]
        if res.shape[0] == batch_size:
            return res


def emb_std(embs):
    """Calculate average embedding across all batch elements and seq length. Then calculate std around it"""
    with torch.no_grad():
        avg = embs.mean((0, 1), keepdim=True)
        assert avg.shape == (1, 1, embs.size(2))
        var = ((embs - avg) ** 2).mean()
        return (var ** 0.5).item()
        
def emb_mean(embs):
    """Just average of everything. Is it 0??"""
    with torch.no_grad():
        return embs.mean().item()
    
def best_ce_loss(num_chunks, temp):
    """When we calculate CE loss over cos sim scores, loss can be pretty big even for pretty good predictions
    This function calculates best ce loss in case:
    1) All distractors get orthogonal embeddings. This is pretty good
    2) All distractors git inverse embedding. This is theoretically optimal
    Cos sim scores are divided by temp. Should be 1 for vanilla score"""
    max_val = math.exp(1 / temp)
    ort_val = 1 # e ^ 0
    inv_val = math.exp(-1 / temp)
    
    return {
        'ort_best_loss': -1 * math.log(max_val / (max_val + (num_chunks - 1) * ort_val)),
        'inverse_best_loss': -1 * math.log(max_val/ (max_val + (num_chunks - 1) * inv_val)),
    }
    
def warn_one_batch(is_one_batch):
    for i in range(10):
        if is_one_batch:
            print("💀💀💀💀💀💀💀💀💀💀💀💀💀💀 THIS IS ONE BATCH TEST 💀💀💀💀💀💀💀💀💀💀💀💀💀💀")
        else:
            print("🚨🚨🚨 REAL RUN 🚨🚨🚨")
            

def info_about_training(dataset, loader, encoder, context_network, cfg, device):
    sample_batch = next(iter(loader))
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0]['data'].shape}")   
    print("Shape of sample batch:", sample_batch['data'].shape)
    print(f"Total length of segment: {sample_batch['data'].shape[2] / 250:.2f}s") 
    
    print(f"Encoder:\n{encoder}")
    print(f"ContextNetwork:\n{context_network}")
    # print(f"% tokens masked every batch: {100 * context_network.avg_part_masked(sample_batch['data']):.2f}%")
    loss_benches = best_ce_loss(cfg['context_network']['num_negatives'], cfg['context_network']['temp'])
    print(f"Optimal loss for orthogonal distractors: {loss_benches['ort_best_loss']:.3f}")
    print(f"Optimal loss for inverse distractors:    {loss_benches['inverse_best_loss']:.3f}")
    
    sample_batch['data'] = sample_batch['data'].to(device=device, dtype=torch.float32)
    sample_batch = encoder(sample_batch)
    
    print("Encoder output shape:", sample_batch['encoder_features'].shape)
    time_masking(cfg, device)
    
    
def plot_sim_image(targets, contexts, masks):
    assert targets.shape == contexts.shape and len(masks.shape) == 1 and len(targets.shape) == 2

    with torch.no_grad(), torch.cuda.amp.autocast():
        targets = targets / torch.norm(targets, dim=1, keepdim=True)
        contexts = contexts / torch.norm(contexts, dim=1, keepdim=True)
        sims = (contexts @ targets.T).detach().cpu()
        masks = masks.detach().cpu()    
    
    # weird GPT plotting code, very brittle! dont touch
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]}, figsize=(8.2, 6), sharey=True)

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("left", size="5%", pad=0.7)
    cax.set_xlabel("sims")

    im = ax[0].imshow(sims)
    fig.colorbar(im, cax=cax)  # Place the colorbar on the left
    cax.yaxis.set_ticks_position('left')  # Ensure the ticks are on the left
    ax[0].set_xlabel("Targets")
    ax[0].set_ylabel("Contexts")
    mask_colors = np.where(masks, 1, 0)  # True becomes 1 (blue), False becomes 0 (white)
    ax[1].imshow(mask_colors[:, None], cmap='Blues', aspect='auto')  # Single column mask
    ax[1].set_xticks([])  
    ax[1].set_yticks([])  
    ax[1].set_xlabel("Blue=mask")
    plt.tight_layout()
    
    fig.canvas.draw()
    res = fig2img(fig)
    plt.close()
    return res


def plot_pca(targets, contexts, masks):
    assert targets.shape == contexts.shape and len(masks.shape) == 1 and len(targets.shape) == 2
    
    with torch.no_grad():
        cat_vecs = torch.cat([contexts.detach().cpu(), targets.detach().cpu()], 0)
        cat_vecs = cat_vecs / torch.norm(cat_vecs, dim=1, keepdim=True)
        u, _, _ = torch.pca_lowrank(cat_vecs, q=2)
        msk = masks.cpu().detach()

    cont_2d = u[:msk.shape[0], :]
    tgt_2d = u[msk.shape[0]:, :]
    plt.scatter(cont_2d[msk, 0], cont_2d[msk, 1], c='r', alpha=0.1, label='Masked Context')
    plt.scatter(cont_2d[~msk, 0], cont_2d[~msk, 1], c='b', alpha=0.1, label='Visible Context')
    plt.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c='g', alpha=0.1, label='Targets')
    plt.legend()
    plt.title('PCA of context + targets')
    plt.gcf().canvas.draw()
    res = fig2img(plt.gcf())
    plt.close()
    return res


@torch.no_grad()
def distance_edge(bool_tensor):
    out1 = torch.zeros_like(bool_tensor, dtype=torch.int32)
    out2 = torch.zeros_like(bool_tensor, dtype=torch.int32)
    
    count = 0
    for i in range(len(bool_tensor)):
        if bool_tensor[i]:  
            count += 1
            out1[i] += count  
        else:
            count = 0  
    
    count = 0
    for i in range(len(bool_tensor) - 1, -1, -1):
        if bool_tensor[i]: 
            count += 1
            out2[i] += count 
        else:
            count = 0 
    
    return torch.minimum(out1, out2)

@torch.no_grad()
def loss_edge_dist_distribution(mask, per_masktoken_loss):
    dist = torch.stack([distance_edge(mask[i]) for i in range(mask.shape[0])], 0)
    dist = dist[mask]
    loss = per_masktoken_loss.flatten()
    df = pd.DataFrame({'dist': dist.detach().cpu(), 'loss': loss.detach().cpu()})
    df = df.groupby('dist', as_index=False).agg(
        mean_loss=('loss', 'mean'),  # Calculate the mean of 'loss'
        count=('loss', 'size')       # Count the number of occurrences
    )
    df = df.sort_values(by='dist')  # Sort the result by 'dist'
    print(df)
    return df
    

def mn(x):
    return sum(x) / len(x)


def time_masking(cfg, device):
    t = []
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    for _ in range(1000):
        s = time.time()
        make_pretrain_mask(
            batch_size=cfg['data_train']['batch_size'], 
            num_chunks=768,
            mask_prob=cfg['context_network']['mask_prob'],
            mask_length=cfg['context_network']['mask_length'],
            min_masked=cfg['context_network']['min_masked'],
            device=device)
        torch.cuda.synchronize(device)
        t.append(time.time() - s)
    final_time = (sum(t) / len(t)) * 1000
    assert final_time < 3, f"Masking works for longer than 3ms: {final_time:.3f}ms"
    print(f"Masking works for {final_time:.3f}ms (assuming 768 chunks)")