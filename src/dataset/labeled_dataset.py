import torch 
from torch import nn 
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from utils.data_utils import load_recording, calc_percent_clipped, check_std_channels
from safetensors import safe_open
from safetensors.torch import save_file

def save_tsr(x, pth):
    save_file(x, pth)

def load_tsr(pth):
    tensors = {}
    with safe_open(pth, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors
        


class EEGLabeledDataset(Dataset):
    def __init__(self, data_path, cache_processed_path, train_length, dataset_mode, target_config, clipped_threshold, norm_std_range_min=0.01, norm_std_range_max=1.99, limit=None, rebuild_cache=True, **kwargs):
        super().__init__()
        self.none_user_id = -100
        self.num_user_ids = None
        self.data_dir = Path(data_path)
        self.cache_processed_path = Path(cache_processed_path)
        self.metadata = pd.read_parquet(self.data_dir / 'metadata.parquet') if (self.data_dir / 'metadata.parquet').exists() else None
        self.prepare_metadata()
        
        self.available_filenames = self.metadata.sort_values(by='filename_h5')['filename_h5'].to_list() if self.metadata is not None else None
        self.train_length = train_length
        self.target_config = target_config
        self.clipped_threshold = clipped_threshold
        self.norm_std_range_min = norm_std_range_min
        self.norm_std_range_max = norm_std_range_max
        self.clip_val = 1e-4
        
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
            
    def prepare_metadata(self, ):
        if self.metadata is None:
            return

        user_ids = {serial: idx for idx, serial in enumerate(self.metadata['filename_h5'].dropna().unique())}
        self.num_user_ids = len(user_ids)
        self.metadata['user_id'] = self.metadata['filename_h5'].map(user_ids).fillna(self.none_user_id)
        
            
            
    def get_targets(self, fn, start_ind, save_dict):
        assert self.metadata is not None
        row = self.metadata[self.metadata['filename_h5'] == fn].iloc[0]
        if self.target_config['user_id']:
            save_dict['user_id'] = torch.tensor(row['user_id'])
        if self.target_config['activity']:
            save_dict['activity'] = torch.zeros(768) # Placeholder for now, dont forget that process_data cuts some points from ends
        return save_dict
         
    def prepare_cache(self, ):
        assert self.available_filenames is not None, "self.available_filenames is None"
        if self.dataset_mode == "beginning_from_each":
            assert not self.compute_targets
            for i in tqdm(range(len(self.available_filenames)), desc='One chunk from each file'):
                raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[i]) # data is raw eeg numpy array 
                proc_data = self.process_data(raw_data)
                proc_data = proc_data[:, :self.train_length]
                save_tsr(proc_data, self.cache_processed_path / f"{i}.pt")
        elif self.dataset_mode == "intersecting_from_one":
            assert not self.compute_targets
            raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[0])
            proc_data = self.process_data(raw_data)
            for i in tqdm(range(40), desc="40 chunks from 100min of first file"):
                start = (self.train_length // 2) * i
                chunk = proc_data[:, start : start + self.train_length]
                save_tsr(chunk, self.cache_processed_path / f"{i}.pt")
        elif self.dataset_mode == "full":
            numel = 0
            for i in tqdm(range(len(self.available_filenames)), desc='Chunks from each file without intersection'):
                raw_data, _, _ = load_recording(self.data_dir / self.available_filenames[i]) # data is raw eeg numpy array 
                if raw_data is None:
                    continue
                proc_data = self.process_data(raw_data)
                start = 0
                while True:
                    chunk = proc_data[:, start : start + self.train_length]
                    if chunk.shape[1] != self.train_length:
                        break
                    is_ok = calc_percent_clipped(chunk, self.clip_val) < self.clipped_threshold
                    is_ok = is_ok and check_std_channels(self.normalize_data(chunk), self.norm_std_range_min, self.norm_std_range_max) 
                    # this might be MEGA slow, multiprocessing???
                    if not is_ok:
                        start += self.train_length // 10
                        continue
                    to_save = {'data': chunk.detach().clone()}
                    to_save = self.get_targets(self.available_filenames[i], start, to_save)
                    save_tsr(to_save, self.cache_processed_path / f"{numel}.pt")
                    start += self.train_length
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
        
        data = torch.clip(data, -self.clip_val, self.clip_val)

        return data
    
    
    def __len__(self,):
        return len(self.cached_ids)
    
    def normalize_data(self, data):
        data = data - data.mean(1, keepdim=True)
        data = data / ((data ** 2).mean() ** 0.5)
        return data
    
    def __getitem__(self, index):
        if index in self.cached_ids:
            r = load_tsr(self.cache_processed_path / f"{index}.pt")
            r['data']= self.normalize_data(r['data'])
            return r
        else:
            raise Exception("bad ind")
