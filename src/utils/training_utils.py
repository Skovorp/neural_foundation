import numpy as np
from torch import nn
import torch
import math


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
    # print(mask.sum(1))
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