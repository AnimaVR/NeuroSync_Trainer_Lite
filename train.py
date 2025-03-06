# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import torch

from config import training_config as config
from dataset.dataset import prepare_dataloader_with_split

from utils.training_helpers import prepare_devices_and_models, load_or_initialize_models
from utils.model_utils import prepare_training_components
from utils.training_utils import train_model

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    torch.cuda.empty_cache()

    train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_dataloader_with_split(config, val_split=0.01)

    devices, use_multi_gpu, models = prepare_devices_and_models(config)

    criterion, optimizer, scheduler = prepare_training_components(config, models[0])
    
    models, optimizer, scheduler, start_epoch, batch_step = load_or_initialize_models(config, models, optimizer, scheduler, devices[0])
    
    # Start training.
    train_model(
        config, models[0], models[1], models[2], models[3],
        train_dataloader, val_dataloader, criterion, optimizer, scheduler,
        devices=devices, use_multi_gpu=use_multi_gpu, start_epoch=start_epoch, batch_step=batch_step
    )



