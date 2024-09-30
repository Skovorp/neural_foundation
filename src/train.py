from dataset.labeled_dataset import EEGLabeledDataset, collate_fn
from models.bendr import EncoderConv, ContextNetwork, calc_loss_proper
from utils.training_utils import emb_std, emb_mean, warn_one_batch, info_about_training, plot_pca, plot_sim_image

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import torch
import yaml
from tqdm import tqdm
import wandb
import numpy as np
from datetime import datetime

# import lovely_tensors as lt
# lt.monkey_patch()

import warnings
warnings.filterwarnings('ignore', message='Lazy modules.*')
warnings.filterwarnings('ignore', message='Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*') # https://github.com/pytorch/pytorch/issues/121834
warnings.filterwarnings('ignore', message='.*The epoch parameter in `scheduler.step().*')
warnings.filterwarnings('ignore', message='.*You are using `torch.load`.*')

# fix random seeds for reproducibility
SEED = 456
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    with open('configs/clean_config_bendr.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    wandb.init(
        project='neural_foundation',
        config=cfg,
        # mode='disabled'
    )
    
    device = torch.device('mps')
    
    dataset = EEGLabeledDataset(**cfg['data'])
    loader = DataLoader(dataset, cfg['data']['batch_size'], num_workers=cfg['data']['num_workers'], 
                        persistent_workers=cfg['data']['persistent_workers'],
                        shuffle=False, drop_last=True, collate_fn=collate_fn)
    encoder = EncoderConv(**cfg['encoder']).to(device)
    context_network = ContextNetwork(**cfg['context_network']).to(device)
    info_about_training(dataset, loader, encoder, context_network, cfg, device)
    warn_one_batch(cfg['data']['pin_window'])
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(context_network.parameters()),
        **cfg['optimizer']
    )
    
    total_steps = len(loader) * cfg['training']['num_epochs']
    warmup_steps = int(cfg['scheduler']['part_warmup_steps'] * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1, total_iters=warmup_steps)
    other_scheduler = LinearLR(optimizer, start_factor=1, end_factor=1e-6, total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, other_scheduler], milestones=[warmup_steps])
    
    for epoch_num in range(1, cfg['training']['num_epochs'] + 1):
        pbar = tqdm(loader)
        losses = []
        accs = []
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch['data'] = batch['data'].to(device, dtype=torch.float32)
            
            batch = encoder(batch)
            batch = context_network(batch)
            batch = calc_loss_proper(batch, cfg['context_network']['temp'],  cfg['context_network']['num_negatives'])
            
            batch['loss'].backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(batch['loss'].item())
            accs.append(batch['acc_feature_choice'].item())
            
            pbar.set_description(f"loss: {batch['loss'].item():.5f} acc: {100 * batch['acc_feature_choice'].item():.2f}%")
            
            if cfg['training']['heavy_logs_every'] != -1 and ((epoch_num - 1) * len(loader) + batch_idx) % cfg['training']['heavy_logs_every'] == 0:
                wandb.log({
                    'encoder_hist': wandb.Histogram(batch['encoder_features'].detach().cpu().numpy(), num_bins=512),
                    'target_hist': wandb.Histogram(batch['targets'].detach().cpu().numpy(), num_bins=512),
                    'context_hist': wandb.Histogram(batch['context_vectors'].detach().cpu().numpy(), num_bins=512),
                    'loss_hist': wandb.Histogram(batch['per_masktoken_loss'].detach().cpu().numpy(), num_bins=64),
                    
                    'sample_sim': wandb.Image(plot_sim_image(batch['targets'][0], batch['context_vectors'][0], batch['mask'][0])),
                    'sample_pca': wandb.Image(plot_pca(batch['targets'][0], batch['context_vectors'][0], batch['mask'][0]))
                    # 'sample_proc_spec': wandb.Image(plot_spec(batch['sample_processed'])),
                    # 'sample_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'])),
                    # 'full_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'], n=None)),
                }, commit=False)
            
            wandb.log({
                'step_loss': batch['loss'].item(),
                'step_acc_feature_choice': batch['acc_feature_choice'].item(),
                'mean_correct_sim': batch['mean_correct_sim'].item(),
                'mean_destractor_sim': batch['mean_destractor_sim'].item(),
                'lr': scheduler.get_last_lr()[0],
                
                'encoder_std': emb_std(batch['encoder_features']),
                'target_std': emb_std(batch['targets']),
                'context_std': emb_std(batch['context_vectors']),
                
                'encoder_mean': emb_mean(batch['encoder_features']),
                'target_mean': emb_mean(batch['targets']),
                'context_mean': emb_mean(batch['context_vectors']),
                # 'batch_partnnnn_clipped': calc_part_clipped(batch['data']) -- how to calc after normalisation????
            })
            
            
        if epoch_num % cfg['save']['every'] == 0:
            torch.save(encoder.state_dict(), f"{cfg['save']['dir']}/encoder_{datetime.now().isoformat()}.pt")
            torch.save(context_network.state_dict(), f"{cfg['save']['dir']}/context_network_{datetime.now().isoformat()}.pt")
            
        print(f"\tEpoch {epoch_num:>3} average loss: {sum(losses) / len(losses):.5f} average acc: {100 * sum(accs) / len(accs):.2f}%")
        wandb.log({
            'epoch_loss': sum(losses) / len(losses),
            'epoch_acc_feature_choice': sum(accs) / len(accs)
        }, commit=False)

wandb.finish()
