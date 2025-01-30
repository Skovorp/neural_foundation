from dataset.labeled_dataset import EEGLabeledDataset
from models.bendr import EncoderConv, ContextNetwork
from loss.loss_bendr import calc_loss_effective, calc_loss_proper

from torch.utils.data import DataLoader
import torch
import yaml
from tqdm import tqdm
import numpy as np
import time

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
    
    scaler = torch.amp.GradScaler(enabled=cfg["training"]["scaler"], init_scale=2. ** 18)
    
    encoder.train()
    context_network.train()
    
    iteration_times = []
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_mask_smart_b64_20l_flashfp16_loss_eff'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for batch_idx, batch in enumerate(train_loader):
            with torch.autocast(
                device_type=cfg["training"]['device'], enabled=cfg["training"]["mixed_precision"],
                dtype=eval(cfg["training"]["mp_dtype"])
            ):
                prof.step()
                if batch_idx >= ((3 + 5 + 3) * 1):
                    break
                start_time = time.time()

                optimizer.zero_grad(set_to_none=True)
                batch['data'] = batch['data'].to(device, non_blocking=True)
                
                batch = encoder(batch)
                batch = context_network(batch)
                # batch = calc_loss_proper(batch, cfg['context_network']['temp'],  cfg['context_network']['num_negatives'])
                batch = calc_loss_effective(batch, cfg['context_network']['temp'])
                
            scaler.scale(batch['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration_times.append(time.time() - start_time)
    print(f"Single iteration takes {sum(iteration_times[-3:]) * 1000 / 3:.2f}ms")
    print(f"Maximum GPU memory recorded per step: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
