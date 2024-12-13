import json
import h5py
import numpy as np
from torch import nn
import torch
from torch import stft
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import PIL
import traceback



def load_recording(path_to_h5):
    """
    Returns: 
        eeg data: np.array of type np.float32 with shape (4, X)"
        metadata: dict
    """
    try:
        path_to_h5 = str(path_to_h5)
        path_to_h5 = path_to_h5 if path_to_h5[-3:] == ".h5" else path_to_h5 + '.h5'
        with h5py.File(path_to_h5, "r") as f:
            assert 'eeg4_datasetName' in list(f.keys()), f"No eeg4_datasetName in h5 keys: {f.keys()}"
            # ppg1_datasetName is heartrate data that's recorded every 10ms and not 4ms like eeg -- i just ignore it
            # also can have gyroscope3_datasetName, accelerometer3_datasetName
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
    except Exception:
        traceback.print_exc()
        print(f"\n!!! Cant open file, skipping: {path_to_h5}")
        return None, None, None
    
    
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


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def plot_spec(data, n_fft=250, hop_length=125):
    """Input: 1d torch tensor or np.ndarray. Assumes 250hz sampling frequency. Returns PIL.Image"""
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
        
    
    assert len(data.shape) == 1, f"expected 1d input, got shape: {data.shape}"
    spec = stft(data, n_fft=n_fft, window=torch.hann_window(n_fft).to(data.device), hop_length=hop_length, center=False, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt((spec * spec).sum(2))
    
    freqs = torch.linspace(0, data.shape[0]/250, spec.shape[1])
    times = torch.linspace(0, 125, spec.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolor(freqs.cpu().numpy(), times, spec.cpu().numpy(), shading='auto',
               norm=LogNorm(vmin=max(spec.min().item(), 1e-9), vmax=max(spec.max().item(), 1e-9)))
    fig.colorbar(c, ax=ax)
    
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.canvas.draw()
    # res = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    res = fig2img(fig)
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
    ax.plot(times, data.cpu())
    
    ax.set_ylabel('Data')
    ax.set_xlabel('Time [sec]')
    fig.canvas.draw()
    # res = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    res = fig2img(fig)
    plt.close()
    return res

@torch.no_grad()
def calc_percent_clipped(data, clip_val):
    abs_data = data.abs()
    part_clipped = (abs_data == clip_val).sum().item() * 100.0
    part_clipped = part_clipped / torch.numel(data)
    return part_clipped


def check_std_channels(data, std_min, std_max):
    # data is (4, datapoints)
    stds = data.std(axis=1)
    if stds.isnan().any() or stds.isinf().any():
        return False
    if (stds < std_min).any() or (stds > std_max).any():
        return False
    return True

def band_pass_brickwall(data, min_freq, max_freq):
    # zero all freqs less then min_freq and more then max_freq, decline from 1 to 0 gradually across "side" samples
    # default slope is 0.5hz, slope's middle is centered at cutoff freq. Slope is done by convolution
    # does fft, mask, ifft
    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    assert min_freq < max_freq, 'you are dumb'
    side = int(data.shape[-1] / 500) 
    bin_freqs = torch.abs(torch.fft.fftfreq(data.shape[-1], d=1 / 250))
    mask = torch.ones_like(bin_freqs, dtype=torch.float32, device=data.device)
    mask_for_zero = (bin_freqs >= max_freq) | (bin_freqs <= min_freq)
    mask[mask_for_zero] = 0
    smooth_kernel = (torch.ones(side) / side).reshape(1, 1, side).to(mask.device)
    mask = torch.nn.functional.conv1d(mask.unsqueeze(0), smooth_kernel, bias=None, stride=1, padding='same')
    
    
    freqs = torch.fft.fft(data, axis=-1)
    freqs = freqs * mask
    
    filtered = torch.fft.ifft(freqs, axis=-1)
    filtered = filtered.real
    discard_samples = 250 * 20 * 20 # experiment shows that ripples from step between stard and end subside after 20 sec, lets add 10x safety
    filtered = filtered[:, discard_samples:-discard_samples]
    
    return filtered[0]