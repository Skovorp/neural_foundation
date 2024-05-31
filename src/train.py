from dataset import EEGDataset, collate_fn
from models.bendr import Encoder, ContextNetwork, calc_loss
from utils.training_utils import emb_std, emb_mean, best_ce_loss
from utils.data_utils import plot_spec, plot_first_n, calc_part_clipped

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import torch
import yaml
from tqdm import tqdm
import wandb
import numpy as np

import lovely_tensors as lt
lt.monkey_patch()

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
        # mode='disabled'
    )
    
    device = torch.device('cuda')
    
    dataset = EEGDataset(**cfg['data'])
    print(f"Dataset length: {len(dataset)}. Shape of first: {dataset[0]['chunked_data'].shape}")
    
    loader = DataLoader(dataset, cfg['data']['batch_size'], num_workers=cfg['data']['num_workers'], shuffle=False, drop_last=True, collate_fn=collate_fn)
    sample_batch = next(iter(loader))
    print("Shape of sample batch:", sample_batch['data'].shape)
    print(f"Total length of segment: {sample_batch['sample_raw'].shape[0] / 250:.2f}s")
    
    
    encoder = Encoder(**cfg['encoder']).to(device)
    context_network = ContextNetwork(**cfg['context_network']).to(device)
    print(f"Encoder:\n{encoder}")
    print(f"\ContextNetwork:\n{context_network}")
    print(f"% tokens masked every batch: {100 * context_network.avg_part_masked(sample_batch['data']):.2f}%")
    loss_benches = best_ce_loss(cfg['data']['num_chunks'], cfg['context_network']['log_temp'])
    print(f"Optimal loss for orthogonal distractors: {loss_benches['ort_best_loss']:.3f}")
    print(f"Optimal loss for inverse distractors:    {loss_benches['inverse_best_loss']:.3f}")
    
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
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch['data'] = batch['data'].to(device)
            # print(batch['data'])
            
            batch = encoder(batch)
            batch = context_network(batch)
            batch = calc_loss(batch, cfg['context_network']['log_temp'])
            # print(batch)
            
            batch['loss'].backward()
            optimizer.step()
            scheduler.step()
            losses.append(batch['loss'].item())
            
            pbar.set_description(f"loss: {batch['loss'].item():.5f}")
            
            if cfg['training']['heavy_logs_every'] != -1 and ((epoch_num - 1) * len(loader) + batch_idx) % cfg['training']['heavy_logs_every'] == 0:
                wandb.log({
                    'encoder_hist': wandb.Histogram(batch['encoder_features'].detach().cpu().numpy(), num_bins=512),
                    'target_hist': wandb.Histogram(batch['targets'].detach().cpu().numpy(), num_bins=512),
                    'context_hist': wandb.Histogram(batch['context_vectors'].detach().cpu().numpy(), num_bins=512),
                    'loss_hist': wandb.Histogram(batch['per_masktoken_loss'].detach().cpu().numpy(), num_bins=64),
                    
                    # 'sample_raw_spec': wandb.Image(plot_spec(batch['sample_raw'])),
                    'sample_proc_spec': wandb.Image(plot_spec(batch['sample_processed'])),
                    # 'sample_raw_plot': wandb.Image(plot_first_n(batch['sample_raw'])),
                    'sample_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'])),
                    # 'full_raw_plot': wandb.Image(plot_first_n(batch['sample_raw'], n=None)),
                    'full_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'], n=None)),
                }, commit=False)
            
            wandb.log({
                'step_loss': batch['loss'].item(),
                'lr': scheduler.get_last_lr()[0],
                'encoder_std': emb_std(batch['encoder_features']),
                'encoder_mean': emb_mean(batch['encoder_features']),
                'target_std': emb_std(batch['targets']),
                'target_mean': emb_std(batch['targets']),
                'context_std': emb_std(batch['context_vectors']),
                'context_mean': emb_mean(batch['context_vectors']),
                'data_mean': batch['data'].mean().item(), 
                'data_std': batch['data'].std().item(), 
                # 'batch_part_clipped': calc_part_clipped(batch['data'], cfg['data']['clip_val'])
            })
            
        print(f"Epoch {epoch_num:>3} average loss {sum(losses) / len(losses):.5f}")
        wandb.log({
            'epoch_loss': sum(losses) / len(losses)
        }, commit=False)

wandb.finish()
