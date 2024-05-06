from dataset import EEGDataset
from model import Encoder, Decoder

from torch.utils.data import DataLoader
import torch
import yaml
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='Lazy modules.*')
warnings.filterwarnings('ignore', message='Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*') # https://github.com/pytorch/pytorch/issues/121834

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    device = torch.device('cuda')
    
    dataset = EEGDataset(**cfg['data'])
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0].shape}")
    
    loader = DataLoader(dataset, cfg['data']['batch_size'], drop_last=True)
    sample_batch = next(iter(loader))
    print("Shape of sample batch:", sample_batch.shape)
    
    
    encoder = Encoder(**cfg['encoder']).to(device)
    decoder = Decoder().to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        **cfg['optimizer']
    )
    
    losses = []
    for epoch_num in range(1, cfg['training']['num_epochs'] + 1):
        pbar = tqdm(loader)
        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            encoder_res = encoder(batch)
            res = decoder(encoder_res)
            loss = ((encoder_res[:, 1:, :] - res[:, :-1, :]) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # pbar.set_description(f"Loss: {loss.item():.5f}")
            # bench_loss_previous = ((encoder_res[:, 1:, :] - encoder_res[:, :-1, :]) ** 2).mean()
            # bench_loss_average = ((encoder_res[:, 1:, :] - encoder_res[:, :-1, :].cumsum(1) ) ** 2).mean()
            pbar.set_description(f"loss: {loss.item():.5f}")
        print(f"Epoch {epoch_num:>3} average loss {sum(losses) / len(losses):.5f}")
