import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from utils.data_utils import load_recording, turn_into_patches, plot_spec, band_pass_brickwall

from torchaudio.functional import highpass_biquad, bandreject_biquad
torch.multiprocessing.set_start_method('spawn')


class EEGDataset(Dataset):
    def __init__(self, data_path, limit, chunk_length, chunk_stride, num_chunks, pin_window,  is_processed, buffer_length=0, clip_val=None, last_is_serial=None, **kwargs):
        super().__init__()
        self.data_dir = Path(data_path)
        self.is_processed = is_processed
        if not is_processed:
            self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet')
            self.needed_filenames = self.metadata.sort_values(by='duration', ascending=False)['filename_h5'].to_list()
        else:
            self.needed_filenames = list(self.data_dir.glob('*'))
        
        if limit is not None:
            self.needed_filenames = self.needed_filenames[:limit]
            
        
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.num_chunks = num_chunks
        self.pin_window = pin_window
        self.buffer_length = buffer_length
        self.clip_val = clip_val
        self.last_is_serial = last_is_serial
    
    def __len__(self,):
        return len(self.needed_filenames)
    
    def process_data(self, data):
        if self.is_processed:
            return data
        data = torch.stack([
            band_pass_brickwall(data[0], 1, 40),
            band_pass_brickwall(data[1], 1, 40),
            band_pass_brickwall(data[2], 1, 40),
            band_pass_brickwall(data[3], 1, 40),
        ])
        data = torch.clip(data, -1.95e-4, 1.95e-4)
        data = data[:, self.buffer_length:]
        # high pass should remove DC offset, but lets substract mean anyways
        data = data - data.mean(1, keepdim=True)
        # lets normalize variance across all channels, to preserve info between channels
        data = data / ((data ** 2).mean() ** 0.5) # make std 1
        # data = torch.clamp(data, min=-self.clip_val, max=self.clip_val)
        return data
    
    
    def chunked_length(self):
        return (self.num_chunks - 1) * self.chunk_stride + self.chunk_length + self.buffer_length
    
    
    def pick_chunked(self, data):
        chunked_length = self.chunked_length()
        if self.pin_window:
            start = 0
        else:
            assert data.shape[1] >= chunked_length, f"Not enough points in sample. Need {chunked_length}, have only {data.shape[1]}"
            start = torch.randint(low=0, high=data.shape[1] - chunked_length + 1, size=(1,))[0]
        return data[:, start : start + chunked_length]


    def __getitem__(self, index):
        if not self.is_processed:
            data, timestamps, meta = load_recording(self.data_dir / self.needed_filenames[index])
            data = torch.tensor(data).cuda()
        else:
            # print(self.needed_filenames[index]) 
            data = torch.load(self.data_dir / self.needed_filenames[index])
        serial = self.needed_filenames[index].stem.split('_')[-1] if self.last_is_serial else None
        data = data
        data = self.process_data(data)
        data = self.pick_chunked(data)
        
        chunked_data = turn_into_patches(data, self.chunk_length, self.chunk_stride)
        assert chunked_data.shape[0] == self.num_chunks, f"sample should have {self.num_chunks}, got {chunked_data.shape[0]}"
        return {
            'raw_data': data,
            'processed_data': data,
            'chunked_data': chunked_data,
            'serial': serial
        }
        
    
def collate_fn(elements):
    res = {'data': []}
    
    for el in elements:
        res['data'].append(el['chunked_data'])
    res['data'] = torch.stack(res['data'])
    
    res['sample_raw'] = elements[0]['raw_data'][0]
    res['sample_processed'] = elements[0]['processed_data'][0]
    res['serial'] = [el['serial'] for el in elements]
    return res
