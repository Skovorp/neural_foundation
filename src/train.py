from dataset.labeled_dataset import EEGLabeledDataset
from models.bendr import EncoderConv, ContextNetwork
from loss.loss_bendr import calc_loss_proper, calc_loss_effective
from utils.training_utils import emb_std, emb_mean, warn_one_batch, info_about_training, plot_pca, plot_sim_image, mn, loss_edge_dist_distribution
from eval.reg import evaluate_logistic_regression

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR
import torch
import yaml
from tqdm import tqdm
import wandb
import numpy as np
from datetime import datetime
import os

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
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

from dotenv import load_dotenv
load_dotenv()

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(False)
# torch.backends.cuda.enable_flash_sdp(True) # Need H1


if __name__ == "__main__":
    with open('configs/od_config_bendr.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    wandb.init(
        project='neural_foundation',
        config=cfg,
        mode='disabled'
    )
    os.makedirs(cfg['save']['dir'], exist_ok=True)
    
    device = torch.device(cfg['training']['device'])
    if cfg['training']['device']:
        print("Device:", torch.cuda.get_device_name())
    
    run_key = str(datetime.now().isoformat())
    
    num_cpu = 16 # len(os.sched_getaffinity(0))
    train_set = EEGLabeledDataset(**cfg['data_train'])
    train_loader = DataLoader(train_set, cfg['data_train']['batch_size'], num_workers=num_cpu, 
                        persistent_workers=True, pin_memory=True,
                        shuffle=True, drop_last=True)
    val_set = EEGLabeledDataset(**cfg['data_val'])
    val_loader = DataLoader(val_set, cfg['data_val']['batch_size'], num_workers=num_cpu, 
                    persistent_workers=True, pin_memory=True,
                    shuffle=False, drop_last=False)
    
    encoder = EncoderConv(**cfg['encoder']).to(device)
    context_network = ContextNetwork(**cfg['context_network']).to(device)
    info_about_training(train_set, train_loader, encoder, context_network, cfg, device)
    warn_one_batch(cfg['data_val']['dataset_mode'] != "full")
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(context_network.parameters()),
        **cfg['optimizer']
    )
    
    total_steps = len(train_loader) * cfg['training']['num_epochs']
    warmup_steps = int(cfg['scheduler']['part_warmup_steps'] * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1, total_iters=warmup_steps)
    other_scheduler = LinearLR(optimizer, start_factor=1, end_factor=1e-6, total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, other_scheduler], milestones=[warmup_steps])
    
    for epoch_num in range(1, cfg['training']['num_epochs'] + 1):
        pbar = tqdm(train_loader)
        train_losses, train_accs = [], []
        encoder.train()
        context_network.train()
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            batch['data'] = batch['data'].to(device, dtype=torch.float32, non_blocking=True)
            
            batch = encoder(batch)
            batch = context_network(batch)
            batch = calc_loss_effective(batch, cfg['context_network']['temp'])
            # batch = calc_loss_proper(batch, cfg['context_network']['temp'],  cfg['context_network']['num_negatives'])
            
            batch['loss'].backward()
            optimizer.step()
            scheduler.step()
            
            train_losses.append(batch['loss'].item())
            train_accs.append(batch['acc_feature_choice'].item())
            
            # pbar.set_description(f"Train {epoch_num} | loss: {batch['loss'].item():.5f} acc: {100 * batch['acc_feature_choice'].item():.2f}%")
            
            if cfg['training']['heavy_logs_every'] != -1 and ((epoch_num - 1) * len(train_loader) + batch_idx) % cfg['training']['heavy_logs_every'] == 0:
                wandb.log({
                    'encoder_hist': wandb.Histogram(batch['encoder_features'].detach().cpu().to(torch.float32).numpy(), num_bins=512),
                    'target_hist': wandb.Histogram(batch['targets'].detach().cpu().to(torch.float32).numpy(), num_bins=512),
                    'context_hist': wandb.Histogram(batch['context_vectors'].detach().cpu().to(torch.float32).numpy(), num_bins=512),
                    'loss_hist': wandb.Histogram(batch['per_masktoken_loss'].detach().cpu().to(torch.float32).numpy(), num_bins=64),
                    
                    'sample_sim': wandb.Image(plot_sim_image(batch['targets'][0], batch['context_vectors'][0], batch['mask'][0])),
                    'sample_pca': wandb.Image(plot_pca(batch['targets'][0], batch['context_vectors'][0], batch['mask'][0])),
                    'loss_edge_dist_distribution': wandb.Table(dataframe=loss_edge_dist_distribution(batch['mask'], batch['per_masktoken_loss']))
                    # 'sample_proc_spec': wandb.Image(plot_spec(batch['sample_processed'])),
                    # 'sample_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'])),
                    # 'full_proc_plot': wandb.Image(plot_first_n(batch['sample_processed'], n=None)),
                }, commit=False)
            
            wandb.log({
                'step_loss': batch['loss'].item(),
                'step_acc': batch['acc_feature_choice'].item(),
                'mean_correct_sim': batch['mean_correct_sim'].item(),
                'mean_destractor_sim': batch['mean_destractor_sim'].item(),
                'lr': scheduler.get_last_lr()[0],
                
                'encoder_std': emb_std(batch['encoder_features']),
                'target_std': emb_std(batch['targets']),
                'context_std': emb_std(batch['context_vectors']),
                
                'encoder_mean': emb_mean(batch['encoder_features']),
                'target_mean': emb_mean(batch['targets']),
                'context_mean': emb_mean(batch['context_vectors']),
            })
            pbar.set_description(f"Train {epoch_num:>3} loss: {batch['loss'].item():.5f} avg loss: {mn(train_losses):.5f} avg acc: {100 * mn(train_accs):.2f}%")
            
        pbar = tqdm(val_loader)
        val_losses, val_accs, seen_els = [], [], 0.0
        user_ids, seq_vecs = [], []
        encoder.eval()
        context_network.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch['data'] = batch['data'].to(device, dtype=torch.float32)
            
                batch = encoder(batch)
                batch = context_network(batch)
                batch = context_network(batch, run_full=True)
                batch = calc_loss_proper(batch, cfg['context_network']['temp'],  cfg['context_network']['num_negatives'])
                val_losses.append(batch['loss'].item() * batch['data'].size(0))
                val_accs.append(batch['acc_feature_choice'].item() * batch['data'].size(0))
                seen_els += batch['data'].size(0)
                user_ids.append(batch['user_id'])
                seq_vecs.append(batch['full_context_vectors'].mean(1).to(torch.device('cpu'), non_blocking=True))
                pbar.set_description(f"Val   {epoch_num:>3} loss: {batch['loss'].item():.5f} avg loss: {sum(val_losses) / seen_els:.5f} avg acc: {100 * sum(val_accs) / seen_els:.2f}%")
        
        user_id_acc = evaluate_logistic_regression(torch.cat(seq_vecs, 0), torch.cat(user_ids, 0), mode='cv', do_norm=True)
        print(user_id_acc)
        if cfg['save']['every'] != -1 and epoch_num % cfg['save']['every'] == 0:
            torch.save(encoder.state_dict(), f"{cfg['save']['dir']}/encoder_{run_key}.pt")
            torch.save(context_network.state_dict(), f"{cfg['save']['dir']}/context_network_{run_key}.pt")
            
        wandb.log({
            'epoch_loss': mn(train_losses),
            'epoch_acc': mn(train_accs),
            'epoch_loss_val': sum(val_losses) / seen_els,
            'epoch_acc_val': sum(val_accs) / seen_els,
            'user_id_acc': user_id_acc,
        }, commit=False)

torch.save(encoder.state_dict(), f"{cfg['save']['dir']}/encoder_{run_key}.pt")
torch.save(context_network.state_dict(), f"{cfg['save']['dir']}/context_network_{run_key}.pt")       
wandb.finish()
