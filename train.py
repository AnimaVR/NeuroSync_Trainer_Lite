# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import multiprocessing
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler  
from config import training_config as config
from utils.training_utils import count_parameters, train_one_epoch, train_one_epoch_multi_gpu, init_weights
from utils.model_utils import save_final_model, build_model, prepare_training_components
from utils.checkpoint_utils import load_checkpoint, save_checkpoint_and_data
from dataset.dataset import prepare_dataloader, prepare_dataloader_with_split  

def train_model(config, model_0, model_1, model_2, model_3, dataloader, val_dataloader, criterion, optimizer, scheduler, devices, use_multi_gpu=False, start_epoch=0, batch_step=0):
    """
    General-purpose training loop that decides whether to use single-GPU 
    or multi-GPU training based on config['num_gpus'].
    """
    n_epochs = config['n_epochs']
    total_batches = n_epochs * len(dataloader)
    lock = multiprocessing.Lock()

    # Print parameter count for the primary model.
    count_parameters(model_0)

    device0 = devices[0]
    use_amp = config.get('use_amp', True)
    scaler = GradScaler() if use_amp else None

    with tqdm(total=total_batches, desc="Training", dynamic_ncols=True) as pbar:
        for epoch in range(start_epoch, n_epochs):
            if use_multi_gpu:
                # Gather all models and corresponding devices (skip any that are None).
                models = [model_0]
                used_devices = [devices[0]]
                if model_1 is not None:
                    models.append(model_1)
                    used_devices.append(devices[1])
                if model_2 is not None:
                    models.append(model_2)
                    used_devices.append(devices[2])
                if model_3 is not None:
                    models.append(model_3)
                    used_devices.append(devices[3])
                batch_step = train_one_epoch_multi_gpu(epoch, models, dataloader, criterion, optimizer, used_devices,
                                                       clip=2.0, batch_step=batch_step, pbar=pbar, total_epochs=n_epochs,
                                                       use_amp=use_amp, grad_scaler=scaler, val_dataloader=val_dataloader,
                                                       validation_interval=20)
            else:
                # Single-GPU training branch.
                batch_step = train_one_epoch(epoch, model=model_0, dataloader=dataloader, criterion=criterion,
                                             optimizer=optimizer, device=device0, clip=2.0, batch_step=batch_step,
                                             pbar=pbar, total_epochs=n_epochs, use_amp=use_amp, grad_scaler=scaler,
                                             val_dataloader=val_dataloader, validation_interval=20)

            scheduler.step()
            save_checkpoint_and_data(epoch, model_0, optimizer, scheduler, batch_step, config, lock, device0)

    save_final_model(model_0)
    return batch_step


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.cuda.empty_cache()
    
    train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_dataloader_with_split(config, val_split=0.1)
    
    desired_gpus = config.get('num_gpus', 1)
    device_count = torch.cuda.device_count()
    
    use_multi_gpu = config.get('use_multi_gpu', False) and (device_count > 1)
    devices = [torch.device(f'cuda:{i}') for i in range(min(device_count, 4))]
    
    while len(devices) < 4:
        devices.append(None)
    model_0 = build_model(config, devices[0] if devices[0] else torch.device('cpu'))
    model_1 = build_model(config, devices[1]) if (use_multi_gpu and desired_gpus >= 2 and devices[1]) else None
    model_2 = build_model(config, devices[2]) if (use_multi_gpu and desired_gpus >= 3 and devices[2]) else None
    model_3 = build_model(config, devices[3]) if (use_multi_gpu and desired_gpus >= 4 and devices[3]) else None
    
    criterion, optimizer, scheduler = prepare_training_components(config, model_0)
    
    start_epoch, batch_step = 0, 0
    
    if config['mode'] == 'resume' and os.path.exists(config['checkpoint_path']):
        start_epoch, batch_step, model_0, optimizer, scheduler = load_checkpoint(config['checkpoint_path'], model_0, optimizer, scheduler, devices[0])
        if model_1 is not None:
            model_1.load_state_dict(model_0.state_dict())
        if model_2 is not None:
            model_2.load_state_dict(model_0.state_dict())
        if model_3 is not None:
            model_3.load_state_dict(model_0.state_dict())
        optimizer.load_state_dict(optimizer.state_dict())
        scheduler.load_state_dict(scheduler.state_dict())
    else:
        model_0.apply(init_weights)
        if model_1 is not None:
            model_1.load_state_dict(model_0.state_dict())
        if model_2 is not None:
            model_2.load_state_dict(model_0.state_dict())
        if model_3 is not None:
            model_3.load_state_dict(model_0.state_dict())
            
    train_model(config, model_0, model_1, model_2, model_3, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                devices, use_multi_gpu=use_multi_gpu, start_epoch=start_epoch, batch_step=batch_step)
