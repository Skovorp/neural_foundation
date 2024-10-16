from dataset.labeled_dataset import EEGLabeledDataset
from models.bendr import EncoderConv, ContextNetwork, calc_loss_proper
from utils.training_utils import emb_std, emb_mean, warn_one_batch, info_about_training, plot_pca, plot_sim_image, mn

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False) # Need H1
np.random.seed(SEED)

# from torch.profiler import profile, record_function, ProfilerActivity
import time

if __name__ == "__main__":
    with open('configs/prof.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    
    device = torch.device(cfg['training']['device'])
    if cfg['training']['device']:
        print("Device:", torch.cuda.get_device_name())
    
    encoder = EncoderConv(**cfg['encoder']).to(device, non_blocking=True)
    context_network = ContextNetwork(**cfg['context_network']).to(device, non_blocking=True)
    
    num_cpu = 8
    train_set = EEGLabeledDataset(**cfg['data_train'])
    train_loader = DataLoader(train_set, cfg['data_train']['batch_size'], num_workers=num_cpu, 
                        persistent_workers=True, pin_memory=True,
                        shuffle=True, drop_last=True)
    
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(context_network.parameters()),
        **cfg['optimizer']
    )
    

    encoder.train()
    context_network.train()
    
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/prof_run'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for batch_idx, batch in enumerate(train_loader):
            prof.step()
            if batch_idx > ((3 + 5 + 3) * 2):
                break

            optimizer.zero_grad(set_to_none=True)
            batch['data'] = batch['data'].to(device, dtype=torch.float32, non_blocking=True)
            
            batch = encoder(batch)
            batch = context_network(batch)
            batch = calc_loss_proper(batch, cfg['context_network']['temp'],  cfg['context_network']['num_negatives'])
            batch['loss'].backward()
            optimizer.step()
            
            
# pin memory  - D 
# больше батч - D
# non blocking? в to? - D
# синк в маске похуй
# откуда синк в начальном переносе - мб с пином станет лучше
# bfloat полный со скейлингом
# и сделай модель больше
# channel first?
