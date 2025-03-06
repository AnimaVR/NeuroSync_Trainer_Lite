# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import multiprocessing
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler  
from config import training_config as config
from utils.training_utils import train_one_epoch, train_one_epoch_multi_gpu
from utils.model_utils import save_final_model, build_model, prepare_training_components
from utils.checkpoint_utils import load_checkpoint, save_checkpoint_and_data
from dataset.dataset import prepare_dataloader, prepare_dataloader_with_split
from utils.training_helpers import count_parameters, init_weights

def train_model(config, model_0, model_1, model_2, model_3, dataloader, val_dataloader, criterion, optimizer, scheduler, devices, use_multi_gpu=False, start_epoch=0, batch_step=0):
    """General-purpose training loop that decides whether to use single- or multi-GPU training."""
    n_epochs = config['n_epochs']
    total_batches = n_epochs * len(dataloader)
    lock = multiprocessing.Lock()
    count_parameters(model_0)
    device0, use_amp, scaler = devices[0], config.get('use_amp', True), GradScaler() if config.get('use_amp', True) else None

    with tqdm(total=total_batches, desc="Training", dynamic_ncols=True) as pbar:
        for epoch in range(start_epoch, n_epochs):
            if use_multi_gpu:
                # Gather models and corresponding devices that are not None:
                models_list = [m for m in (model_0, model_1, model_2, model_3) if m is not None]
                used_devices = [d for m, d in zip((model_0, model_1, model_2, model_3), devices) if m is not None]
                batch_step = train_one_epoch_multi_gpu(
                    epoch, models_list, dataloader, criterion, optimizer, used_devices, clip=2.0,
                    batch_step=batch_step, pbar=pbar, total_epochs=n_epochs, use_amp=use_amp,
                    grad_scaler=scaler, val_dataloader=val_dataloader, validation_interval=20
                )
            else:
                batch_step = train_one_epoch(
                    epoch, model=model_0, dataloader=dataloader, criterion=criterion, optimizer=optimizer,
                    device=device0, clip=2.0, batch_step=batch_step, pbar=pbar, total_epochs=n_epochs,
                    use_amp=use_amp, grad_scaler=scaler, val_dataloader=val_dataloader, validation_interval=20
                )
            scheduler.step()
            save_checkpoint_and_data(epoch, model_0, optimizer, scheduler, batch_step, config, lock, device0)
    save_final_model(model_0)
    return batch_step


if __name__ == "__main__":
    # If you'd like to manually specify which GPUs are visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    torch.cuda.empty_cache()

    # --- Use the split dataloader ---
    train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_dataloader_with_split(config, val_split=0.01)

    # We read the desired number of GPUs and check how many we *actually* have
    desired_gpus = config.get('num_gpus', 1)
    device_count = torch.cuda.device_count()

    # We also check if multi-gpu is turned on
    use_multi_gpu = config.get('use_multi_gpu', False) and (device_count > 1)

    # Create device list: up to 4 devices max, or fewer if device_count < 4
    devices = []
    for i in range(min(device_count, 4)):
        devices.append(torch.device(f'cuda:{i}'))
    # Pad with None if fewer than 4
    while len(devices) < 4:
        devices.append(None)

    # Build the base model on device0
    model_0 = build_model(config, devices[0] if devices[0] else torch.device('cpu'))

    # Conditionally build the other models if multi-GPU is requested
    model_1 = build_model(config, devices[1]) if (use_multi_gpu and desired_gpus >= 2 and devices[1]) else None
    model_2 = build_model(config, devices[2]) if (use_multi_gpu and desired_gpus >= 3 and devices[2]) else None
    model_3 = build_model(config, devices[3]) if (use_multi_gpu and desired_gpus >= 4 and devices[3]) else None

    # Prepare loss, optimizer, scheduler
    criterion, optimizer, scheduler = prepare_training_components(config, model_0)

    # Check for existing checkpoint (resume mode)
    start_epoch, batch_step = 0, 0
    if config['mode'] == 'resume' and os.path.exists(config['checkpoint_path']):
        start_epoch, batch_step, model_0, optimizer, scheduler = load_checkpoint(
            config['checkpoint_path'], model_0, optimizer, scheduler, devices[0]
        )
        # Sync other models with model_0's weights
        if model_1 is not None:
            model_1.load_state_dict(model_0.state_dict())
        if model_2 is not None:
            model_2.load_state_dict(model_0.state_dict())
        if model_3 is not None:
            model_3.load_state_dict(model_0.state_dict())
        optimizer.load_state_dict(optimizer.state_dict())
        scheduler.load_state_dict(scheduler.state_dict())
    else:
        # Initialize model_0 and sync if multi-GPU
        model_0.apply(init_weights)
        if model_1 is not None:
            model_1.load_state_dict(model_0.state_dict())
        if model_2 is not None:
            model_2.load_state_dict(model_0.state_dict())
        if model_3 is not None:
            model_3.load_state_dict(model_0.state_dict())

    # Run training: pass both train_dataloader and val_dataloader
    train_model(
        config,
        model_0,
        model_1,
        model_2,
        model_3,
        train_dataloader,     # <-- Training DataLoader
        val_dataloader,       # <-- Validation DataLoader
        criterion,
        optimizer,
        scheduler,
        devices=devices,
        use_multi_gpu=use_multi_gpu,
        start_epoch=start_epoch,
        batch_step=batch_step
    )


