import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from utils import load_recording, turn_into_patches


class EEGDataset(Dataset):
    def __init__(self, data_path, limit, chunk_length, chunk_stride, num_chunks, **kwargs):
        super().__init__()
        self.data_dir = Path(data_path)
        self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet')
        
        self.needed_filenames = self.metadata['filename_h5'][:limit].to_list()
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.num_chunks = num_chunks
    
    def __len__(self,):
        return len(self.needed_filenames)
    
    def pick_tokens(self, chunked):
        assert chunked.shape[0] >= self.num_chunks, f"Not enough tokens in sample. Need {self.num_chunks}, have only {chunked.shape[0]}"
        start = torch.randint(low=0, high=chunked.shape[0] - self.num_chunks, size=(1,))[0]
        return chunked[start : start + self.num_chunks, :, :]
    
    def __getitem__(self, index):
        data, timestamps, meta = load_recording(self.data_dir / self.needed_filenames[index])
        data = torch.tensor(data)
        data = turn_into_patches(data, self.chunk_length, self.chunk_stride)
        data = self.pick_tokens(data)
        return data
        
    
    