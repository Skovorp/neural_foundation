import json
import h5py
import numpy as np
from torch import nn
import torch
from torch import stft
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import PIL



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


def plot_spec(data, n_fft=250, hop_length=125):
    """Input: 1d torch tensor. Assumes 250hz sampling frequency. Returns PIL.Image"""
    assert len(data.shape) == 1, f"expected 1d input, got shape: {data.shape}"
    spec = stft(data, n_fft=n_fft, window=torch.hann_window(n_fft), hop_length=hop_length, center=False, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt((spec * spec).sum(2))
    
    freqs = torch.linspace(0, data.shape[0]/250, spec.shape[1])
    times = torch.linspace(0, 125, spec.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolor(freqs, times, spec, shading='auto',
               norm=LogNorm(vmin=max(spec.min().item(), 1e-9), vmax=max(spec.max().item(), 1e-9)))
    fig.colorbar(c, ax=ax)
    
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.canvas.draw()
    res = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return res


def plot_first_n(data, n=1000):
    """Input: 1d torch tensor. Assumes 250hz sampling frequency. Returns PIL.Image"""
    assert len(data.shape) == 1, f"expected 1d input, got shape: {data.shape}"
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if n is None:
        n = data.shape[0]
    n = min(n, data.shape[0])
    data = data[:n]
    times = torch.arange(n) / 250
    ax.plot(times, data)
    
    ax.set_ylabel('Data')
    ax.set_xlabel('Time [sec]')
    fig.canvas.draw()
    res = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return res

def calc_part_clipped(data, clip_val):
    part_clipped = (torch.abs(data) > (clip_val - 0.01)).sum().item()
    part_clipped = part_clipped / torch.numel(data)
    return part_clipped