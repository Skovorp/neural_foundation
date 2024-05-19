import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from utils.data_utils import load_recording, turn_into_patches, plot_spec

from torchaudio.functional import highpass_biquad, bandreject_biquad


class EEGDataset(Dataset):
    def __init__(self, data_path, limit, chunk_length, chunk_stride, num_chunks, pin_window, buffer_length, clip_val, **kwargs):
        super().__init__()
        self.data_dir = Path(data_path)
        self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet')
        
        if limit is None:
            limit = len(self.metadata)
        self.needed_filenames = self.metadata.sort_values(by='duration')['filename_h5'][:limit].to_list()
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.num_chunks = num_chunks
        self.pin_window = pin_window
        self.buffer_length = buffer_length
        self.clip_val = clip_val
    
    def __len__(self,):
        return len(self.needed_filenames)
    
    def process_data(self, data):
        data = highpass_biquad(data, 250, 1) 
        data = bandreject_biquad(data, 250, 50)
        data = bandreject_biquad(data, 250, 100)
        data = data[:, self.buffer_length:]
        # high pass should remove DC offset, but lets substract mean anyways
        data = data - data.mean(1, keepdim=True)
        # lets normalize variance across all channels, to preserve info between channels
        data = data / ((data ** 2).mean() ** 0.5) # make std 1
        data = torch.clamp(data, min=-self.clip_val, max=self.clip_val)
        return data
    
    # def pick_tokens(self, chunked):
    #     if not self.pin_window:
    #         assert chunked.shape[0] >= self.num_chunks, f"Not enough tokens in sample. Need {self.num_chunks}, have only {chunked.shape[0]}"
    #         start = torch.randint(low=0, high=chunked.shape[0] - self.num_chunks, size=(1,))[0]
    #         return chunked[start : start + self.num_chunks, :, :]
    #     else:
    #         return chunked[100 : 100 + self.num_chunks, :, :]
    
    def chunked_length(self):
        return (self.num_chunks - 1) * self.chunk_stride + self.chunk_length + self.buffer_length
    
    def pick_chunked(self, data):
        chunked_length = self.chunked_length()
        if self.pin_window:
            start = data.shape[1] // 2
        else:
            assert data.shape[1] >= chunked_length, f"Not enough points in sample. Need {chunked_length}, have only {data.shape[1]}"
            start = torch.randint(low=0, high=data.shape[1] - chunked_length + 1, size=(1,))[0]
        return data[:, start : start + chunked_length]


    def __getitem__(self, index):
        raw_data, timestamps, meta = load_recording(self.data_dir / self.needed_filenames[index])
        raw_data = torch.tensor(raw_data)
        raw_data = self.pick_chunked(raw_data)
        
        proc_data = self.process_data(raw_data)
        
        chunked_data = turn_into_patches(proc_data, self.chunk_length, self.chunk_stride)
        assert chunked_data.shape[0] == self.num_chunks, f"sample should have {self.num_chunks}, got {chunked_data.shape[0]}"
        return {
            'raw_data': raw_data,
            'processed_data': proc_data,
            'chunked_data': chunked_data
        }
        
    
    
def collate_fn(elements):
    res = {'data': []}
    
    for el in elements:
        res['data'].append(el['chunked_data'])
    res['data'] = torch.stack(res['data'])
    
    res['sample_raw'] = elements[0]['raw_data'][0]
    res['sample_processed'] = elements[0]['processed_data'][0]
    return res
