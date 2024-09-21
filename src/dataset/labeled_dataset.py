import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from scipy.signal import butter, sosfilt

from utils.data_utils import load_recording, turn_into_patches


class EEGLabeledDataset(Dataset):
    def __init__(self, data_path, limit, chunk_length, chunk_stride, num_chunks, pin_window, cache_processed_path, **kwargs):
        super().__init__()
        self.data_dir = Path(data_path)
        self.cache_processed_path = Path(cache_processed_path)
        
        # data is not processed, aka stored as raw eeg .h5 files
        self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet')
        self.needed_filenames = self.metadata.sort_values(by='filename_h5')['filename_h5'].to_list()
        
        if limit is not None:
            self.needed_filenames = self.needed_filenames[:limit]
            
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.num_chunks = num_chunks
        self.pin_window = pin_window
        
        self.cached_ids = set([int(x.name[:-3]) for x in self.cache_processed_path.glob('*.pt')]) # assume cached names are 1.pt 2.pt etc.
        
        

    def process_data(self, data):
        """Converts numpy array of eeg data (shape (4, time_stamps)) to ready to train normalised torch tensor"""
        sos = butter(4, Wn=[4, 40], btype='bandpass', analog=False, output='sos', fs=250)
        data = sosfilt(sos, data)
        
        sos = butter(4, Wn=[45, 55], btype='bandstop', analog=False, output='sos', fs=250)
        data = sosfilt(sos, data)
        
        sos = butter(3, Wn=[62.5 - 5, 62.5 + 5], btype='bandstop', analog=False, output='sos', fs=250)
        data = sosfilt(sos, data)
        
        sos = butter(3, Wn=[83.3 - 5, 83.3 + 5], btype='bandstop', analog=False, output='sos', fs=250)
        data = sosfilt(sos, data)
        
        data = torch.tensor(data)
        data = data[:, 500:-500]
        
        data = torch.clip(data, -1e-4, 1e-4)

        data = data - data.mean(1, keepdim=True)
        data = data / ((data ** 2).mean() ** 0.5)
        return data
    
    def __len__(self,):
        return len(self.needed_filenames)
    
    def chunked_length(self):
        return (self.num_chunks - 1) * self.chunk_stride + self.chunk_length
    
    def pick_chunked(self, data):
        chunked_length = self.chunked_length()
        if self.pin_window:
            start = 0
        else:
            assert data.shape[1] >= chunked_length, f"Not enough points in sample. Need {chunked_length}, have only {data.shape[1]}"
            start = torch.randint(low=0, high=data.shape[1] - chunked_length + 1, size=(1,))[0]
        return data[:, start : start + chunked_length], start
        
    def __getitem__(self, index):
        if index in self.cached_ids:
            proc_data = torch.load(self.cache_processed_path / f"{index}.pt")
            #  = torch.rand(4, 5)
        else:
            raw_data, _, meta = load_recording(self.data_dir / self.needed_filenames[index]) # data is raw eeg numpy array 
            proc_data = self.process_data(raw_data)
            assert self.pin_window
            
            
            torch.save(proc_data, self.cache_processed_path / f"{index}.pt")
            self.cached_ids.add(index)
            
        data, start_ind = self.pick_chunked(proc_data) 
        chunked_data = turn_into_patches(data, self.chunk_length, self.chunk_stride)
        assert chunked_data.shape[0] == self.num_chunks, f"sample should have {self.num_chunks}, got {chunked_data.shape[0]}"
        return {
            # 'raw_data': raw_data,
            'processed_data': proc_data,
            'chunked_data': chunked_data,
            'serial': index
        }
        
        
def collate_fn(elements):
    res = {'data': []}
    
    for el in elements:
        res['data'].append(el['chunked_data'])
    res['data'] = torch.stack(res['data'])
    
    # res['sample_raw'] = elements[0]['raw_data'][0]
    res['sample_processed'] = elements[0]['processed_data'][0]
    res['serial'] = [el['serial'] for el in elements]
    return res

