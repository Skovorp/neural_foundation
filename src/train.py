from dataset import EEGDataset, collate_fn
from models.bendr import Encoder, ContextNetwork
from utils import benchmark_previous, benchmark_best_constant, benchmark_cumsum
from utils import plot_spec, plot_first_n

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import torch
import yaml
from tqdm import tqdm
import wandb
import numpy as np

import warnings
warnings.filterwarnings('ignore', message='Lazy modules.*')
warnings.filterwarnings('ignore', message='Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*') # https://github.com/pytorch/pytorch/issues/121834
warnings.filterwarnings('ignore', message='.*The epoch parameter in `scheduler.step().*')

# fix random seeds for reproducibility
SEED = 456
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    with open('configs/config_bendr.yaml', 'r') as file:
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
    sample_batch = next(iter(loader))
    print("Shape of sample batch:", sample_batch['data'].shape)
    print(f"Total length of segment: {sample_batch['sample_raw'].shape[0] / 250:.2f}s")
    
    
    encoder = Encoder(**cfg['encoder']).to(device)
    context_network = ContextNetwork(**cfg['context_network']).to(device)
    print(f"Encoder:\n{encoder}")
    print(f"\ContextNetwork:\n{context_network}")
    print(f"% tokens masked every batch: {100 * context_network.avg_part_masked(sample_batch['data']):.2f}%")
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(context_network.parameters()),
        **cfg['optimizer']
    )
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=cfg['scheduler']['warmup_steps'])
    other_scheduler = LinearLR(optimizer, start_factor=1, end_factor=1e-5, total_iters=cfg['scheduler']['total_steps'] - cfg['scheduler']['warmup_steps'])
    scheduler = SequentialLR(optimizer, [warmup_scheduler, other_scheduler], milestones=[cfg['scheduler']['warmup_steps']])
    
    for epoch_num in range(1, cfg['training']['num_epochs'] + 1):
        pbar = tqdm(loader)
        losses = []
        for batch in pbar:
            optimizer.zero_grad()
            data = batch['data'].to(device)
            
            encoder_res = encoder(data)
            res = context_network(encoder_res)
            loss = ((encoder_res - res) ** 2).mean()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            
            bench_previous_loss = benchmark_previous(encoder_res)
            bench_best_constant_loss = benchmark_best_constant(encoder_res)
            bench_cumsum_loss = benchmark_cumsum(encoder_res)      
            
            pbar.set_description(f"loss: {loss.item():.5f}")   # bench loss: {bench_best_constant_loss:.5f}
            wandb.log({
                'step_loss': loss.item(),
                'bench_previous_loss': bench_previous_loss,
                'bench_best_constant_loss': bench_best_constant_loss,
                'bench_cumsum_loss': bench_cumsum_loss,
                'lr': scheduler.get_last_lr()[0],
                'encoder_mean': encoder_res.mean(),
                'encoder_sq_mean': (encoder_res ** 2).mean(),
                'encoder_hist': wandb.Histogram(encoder_res.detach().cpu().numpy(), num_bins=512)
            })
            # wandb.log({
            #     'sample_raw_spec': wandb.Image(plot_spec(batch['sample_raw'])),
            #     'sample_proc_spec': wandb.Image(plot_spec(batch['sample_processed'])),
            #     'sample_raw_plot': wandb.Image(plot_first_n(batch['sample_raw'])),
            #     'sample_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'])),
            # })
            
        print(f"Epoch {epoch_num:>3} average loss {sum(losses) / len(losses):.5f}")
        wandb.log({
            'epoch_loss': sum(losses) / len(losses)
        }, commit=False)

wandb.finish()
