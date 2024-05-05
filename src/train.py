from dataset import EEGDataset
from model import Encoder, Decoder

from torch.utils.data import DataLoader
import torch
import yaml

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    
    dataset = EEGDataset(**cfg['data'])
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0].shape}")
    
    loader = DataLoader(dataset, cfg['data']['batch_size'], drop_last=True)
    sample_batch = next(iter(loader))
    print("Shape of sample batch:", sample_batch.shape)
    
    
    encoder = Encoder(**cfg['encoder'])
    decoder = Decoder()
    encoder_res = encoder(sample_batch)
    res = decoder(encoder_res)
    loss = ((res[:, :-1, :] - encoder_res[:, 1:, :]) ** 2).mean()
    
    
    
    
    # assert encoder_res.shape[0] == cfg['data']['batch_size'] \
    #     and encoder_res.shape[1] == cfg['data']['num_chunks'] \
    #     and encoder_res.shape[2] == cfg['encoder']['n_filters_time'], \
    #     f'expected encoder output to be (batch_size, num_chunks, n_filters_time, ?), got {encoder_res.shape[2]}'
    
    

    