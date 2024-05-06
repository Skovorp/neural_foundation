import json
import h5py
import numpy as np
from torch import nn
import torch

def load_recording(path_to_h5):
    """
    Returns: 
        eeg data: np.array of type np.float32 with shape (4, X)"
        metadata: dict
    """
    with h5py.File(path_to_h5, "r") as f:
        assert list(f.keys()) == ['eeg4_datasetName'], f"expected structure to be ['eeg4_datasetName'], got {f.keys()}"
        data = f['eeg4_datasetName']
        data_dtype = np.dtype([('timestamp_name', '<u4'), ('eeg1_name', '<f4'), ('eeg2_name', '<f4'), ('eeg3_name', '<f4'), ('eeg4_name', '<f4')])
        assert data.dtype == data_dtype, f"expected dtype to be {data_dtype}, got {data.dtype}"
        # assert (data['timestamp_name'] == np.arange(data.shape[0]) * 4).all(), f"expected timestamp_name to be 0, 4, 8 ... (len(data) - 1) * 4, but got {data['timestamp_name'][:5]}... failed in {(1 * ~(data['timestamp_name'] == np.arange(data.shape[0]) * 4)).sum()} / {data.shape[0]} positions"
        timestamps = data['timestamp_name'] 
        final_data = np.vstack([data['eeg1_name'], data['eeg2_name'], data['eeg3_name'], data['eeg4_name']])
        
        meta = dict(f['eeg4_datasetName'].attrs.items())
        assert list(meta.keys()) == ['eeg4_datasetAttribute', 'eeg4_datasetAttributeStartTime'], f"expected metadata to have keys ['eeg4_datasetAttribute', 'eeg4_datasetAttributeStartTime'], got {meta.keys()}"
        meta['eeg4_datasetAttribute'] = json.loads(meta['eeg4_datasetAttribute'])
        assert meta['eeg4_datasetAttribute']['channelCount'] == 4, f"expected data to have 4 channels. got {meta['eeg4_datasetAttribute']['channelCount']}"
        
        return final_data, timestamps, meta
    
    
def turn_into_patches(data, chunk_length, chunk_stride):
    """Cuts tensor into chunks along time dimention. (4, long) -> (~long / chunk_stride, 4, chunk_length)"""
    # maybe change to torch.tensor_split, torch.chunk
    assert len(data.shape) == 2 and data.shape[0] == 4, f'expected 2d tensor of shape (4, *), got {data.shape}'
    data_prep = data.unsqueeze(0).unsqueeze(3) # (1, 4, long, 1)
    chunked = nn.functional.unfold(data_prep, kernel_size=(chunk_length, 1), stride=(chunk_stride, 1)) # (1, 4 * chunk_length, ~long // chunk_stride)
    assert len(chunked.shape) == 3 and chunked.shape[0] == 1 and chunked.shape[1] == 4 * chunk_length, f'expected (1, 4 * chunk_length, ~long // chunk_stride) got {chunked.shape}'
    chunked = chunked.reshape(4, chunk_length, -1)
    chunked = chunked.permute(2, 0, 1)
    assert (chunked[0, :, :] == data[:, :chunk_length]).all(), "first chunk didn't match the beginning of data"
    return chunked


def plot_spec(data):
    return None

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
