# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import torch

from config import training_config as config

from utils.training_utils import train_model

from utils.model_utils import build_model, prepare_training_components

from utils.checkpoint_utils import load_checkpoint
from dataset.dataset import prepare_dataloader_with_split
from utils.training_helpers import init_weights


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.cuda.empty_cache()
    train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_dataloader_with_split(config, val_split=0.01)
    desired_gpus = config.get('num_gpus', 1)
    device_count = torch.cuda.device_count()

    use_multi_gpu = config.get('use_multi_gpu', False) and (device_count > 1)

    devices = []
    for i in range(min(device_count, 4)):
        devices.append(torch.device(f'cuda:{i}'))
    while len(devices) < 4:
        devices.append(None)

    model_0 = build_model(config, devices[0] if devices[0] else torch.device('cpu'))
    model_1 = build_model(config, devices[1]) if (use_multi_gpu and desired_gpus >= 2 and devices[1]) else None
    model_2 = build_model(config, devices[2]) if (use_multi_gpu and desired_gpus >= 3 and devices[2]) else None
    model_3 = build_model(config, devices[3]) if (use_multi_gpu and desired_gpus >= 4 and devices[3]) else None

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
            
    train_model(config, model_0, model_1, model_2, model_3, train_dataloader, val_dataloader, criterion, optimizer, scheduler, devices=devices, use_multi_gpu=use_multi_gpu, start_epoch=start_epoch, batch_step=batch_step)



