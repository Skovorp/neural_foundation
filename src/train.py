from dataset import EEGDataset, collate_fn
from model import Encoder, Decoder
from utils import benchmark_previous, benchmark_best_constant, benchmark_cumsum

from torch.utils.data import DataLoader
import torch
import yaml
from tqdm import tqdm
import wandb
import numpy as np

import warnings
warnings.filterwarnings('ignore', message='Lazy modules.*')
warnings.filterwarnings('ignore', message='Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*') # https://github.com/pytorch/pytorch/issues/121834

# fix random seeds for reproducibility
SEED = 456
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    wandb.init(
        project='neural_foundation',
        config=cfg,
        mode='disabled'
    )
    
    device = torch.device('cuda')
    
    dataset = EEGDataset(**cfg['data'])
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0]['chunked_data'].shape}")
    
    loader = DataLoader(dataset, cfg['data']['batch_size'], shuffle=False, drop_last=True, collate_fn=collate_fn)
    sample_batch = next(iter(loader))['data']
    print("Shape of sample batch:", sample_batch.shape)
    
    
    encoder = Encoder(**cfg['encoder']).to(device)
    decoder = Decoder().to(device)
    print(f"Encoder:\n{encoder}")
    print(f"\nDecoder:\n{decoder}")
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        **cfg['optimizer']
    )
    
    for epoch_num in range(1, cfg['training']['num_epochs'] + 1):
        pbar = tqdm(loader)
        losses = []
        for batch in pbar:
            optimizer.zero_grad()
            batch = batch['data'].to(device)
            
            encoder_res = encoder(batch)
            res = decoder(encoder_res)
            loss = ((encoder_res[:, 1:, :] - res[:, :-1, :]) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            bench_previous_loss = benchmark_previous(encoder_res)
            bench_best_constant_loss = benchmark_best_constant(encoder_res)
            bench_cumsum_loss = benchmark_cumsum(encoder_res)      
            
            pbar.set_description(f"loss: {loss.item():.5f} bench loss: {bench_best_constant_loss:.5f}")      
            wandb.log({
                'step_loss': loss.item(),
                'bench_previous_loss': bench_previous_loss,
                'bench_best_constant_loss': bench_best_constant_loss,
                'bench_cumsum_loss': bench_cumsum_loss
            })
            
        print(f"Epoch {epoch_num:>3} average loss {sum(losses) / len(losses):.5f}")
        wandb.log({
            'epoch_loss': sum(losses) / len(losses)
        })

wandb.finish()
