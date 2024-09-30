import numpy as np
from torch import nn
import torch
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from utils.data_utils import fig2img


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


def make_pretrain_mask(batch_size, num_chunks, mask_prob, mask_length, min_masked):
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
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0]['processed_data'].shape}")   
    print("Shape of sample batch:", sample_batch['data'].shape)
    print(f"Total length of segment: {sample_batch['sample_processed'].shape[0] / 250:.2f}s") 
    
    print(f"Encoder:\n{encoder}")
    print(f"ContextNetwork:\n{context_network}")
    # print(f"% tokens masked every batch: {100 * context_network.avg_part_masked(sample_batch['data']):.2f}%")
    loss_benches = best_ce_loss(cfg['context_network']['num_negatives'], cfg['context_network']['temp'])
    print(f"Optimal loss for orthogonal distractors: {loss_benches['ort_best_loss']:.3f}")
    print(f"Optimal loss for inverse distractors:    {loss_benches['inverse_best_loss']:.3f}")
    
    sample_batch['data'] = sample_batch['data'].to(device=device, dtype=torch.float32)
    sample_batch = encoder(sample_batch)
    
    print("Encoder output shape:", sample_batch['encoder_features'].shape)
    
    
def plot_sim_image(targets, contexts, masks):
    assert targets.shape == contexts.shape and len(masks.shape) == 1 and len(targets.shape) == 2

    with torch.no_grad():
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