import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from utils.data_utils import load_recording, turn_into_patches


class EEGLabeledDataset(Dataset):
    def __init__(self, data_path, cache_processed_path, train_length, dataset_mode, limit=None, rebuild_cache=True, **kwargs):
        super().__init__()
        self.data_dir = Path(data_path)
        self.cache_processed_path = Path(cache_processed_path)
        self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet')
        self.available_filenames = self.metadata.sort_values(by='filename_h5')['filename_h5'].to_list()
        self.train_length = train_length
        
        assert dataset_mode in ('beginning_from_each', 'intersecting_from_one', 'full'), 'bad dataset_mode'
        self.dataset_mode = dataset_mode
        
        if rebuild_cache:
            [f.unlink() for f in self.cache_processed_path.glob("*")] 
        
        self.cached_ids = self.get_cached_ids()
        if len(self.cached_ids) == 0:
            print("Building cache...")
            self.prepare_cache()
            self.cached_ids = self.get_cached_ids()
            
        if limit is not None:
            self.cached_ids = set(list(self.cached_ids)[:limit])
         
    def prepare_cache(self, ):
        if self.dataset_mode == "beginning_from_each":
            for i in tqdm(range(len(self.available_filenames)), desc='One chunk from each file'):
                raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[i]) # data is raw eeg numpy array 
                proc_data = self.process_data(raw_data)
                proc_data = proc_data[:, :self.train_length]
                torch.save(proc_data, self.cache_processed_path / f"{i}.pt")
        elif self.dataset_mode == "intersecting_from_one":
            raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[0])
            proc_data = self.process_data(raw_data)
            for i in tqdm(range(40), desc="40 chunks from 100min of first file"):
                start = (self.train_length // 2) * i
                chunk = proc_data[:, start : start + self.train_length]
                torch.save(chunk, self.cache_processed_path / f"{i}.pt")
        elif self.dataset_mode == "full":
            numel = 0
            for i in tqdm(range(len(self.available_filenames)), desc='Chunks from each file without intersection'):
                raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[i]) # data is raw eeg numpy array 
                proc_data = self.process_data(raw_data)
                j = 0
                while True:
                    start = (self.train_length) * j
                    chunk = proc_data[:, start : start + self.train_length]
                    if chunk.shape[1] != self.train_length:
                        break
                    torch.save(chunk.detach().clone(), self.cache_processed_path / f"{numel}.pt")
                    j += 1
                    numel += 1
    
    def get_cached_ids(self, ):
        # assume cached names are 0.pt, 1.pt 2.pt etc.
        return set([int(x.name[:-3]) for x in self.cache_processed_path.glob('*.pt')]) 
        
        
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
        
        data = torch.tensor(data, dtype=torch.float32)
        data = data[:, 500:-500]
        
        data = torch.clip(data, -1e-4, 1e-4)

        return data
    
    
    def __len__(self,):
        return len(self.cached_ids)
    
    def normalize_data(self, data):
        data = data - data.mean(1, keepdim=True)
        data = data / ((data ** 2).mean() ** 0.5)
        return data
    
    def __getitem__(self, index):
        if index in self.cached_ids:
            data = torch.load(self.cache_processed_path / f"{index}.pt")
            data = self.normalize_data(data)
            return {'data': data}
        else:
            raise Exception("bad ind")
        
        
def collate_fn(elements):
    res = {'data': []}
    
    for el in elements:
        res['data'].append(el['data'])
    res['data'] = torch.stack(res['data'])
    return res

