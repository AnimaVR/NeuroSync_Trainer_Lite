# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# training_utils.py

import torch
import torch.nn as nn
import time
import os
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from utils.training_helpers import (
    _compute_loss_single_gpu,
    _backward_and_step_single_gpu,
    _run_validation_single_gpu,
    _compute_losses_multi_gpu,
    _backward_and_step_multi_gpu,
    _sync_models,
    _run_validation_multi_gpu, save_loss_plot, save_gradient_norm_plot, print_epoch_summary, print_training_progress, calculate_gradient_norm
)

def train_one_epoch(
    epoch,
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    clip,
    batch_step=0,
    pbar=None,
    total_epochs=None,
    use_amp=False,              # Whether to enable mixed precision
    grad_scaler=None,           # torch.cuda.amp.GradScaler object
    val_dataloader=None,        # Validation DataLoader
    validation_interval=20      # Validation step every N training batches
):
    """
    Trains the model for one epoch on a single GPU, with optional mixed precision and
    periodic validation.
    """
    if use_amp and grad_scaler is None:
        raise ValueError("use_amp=True but no GradScaler was provided!")

    model.train()
    epoch_loss = 0
    start_time = time.time()
    total_steps = total_epochs * len(dataloader)
    gradient_norms = []
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    if val_dataloader is not None:
        val_iter = iter(val_dataloader)

    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        current_step = batch_step + (epoch * len(dataloader)) + batch_idx
        loss = _compute_loss_single_gpu(model, src, trg, criterion, current_step, total_steps, use_amp)

        total_norm = _backward_and_step_single_gpu(loss, model, optimizer, clip, use_amp, grad_scaler)

        train_steps.append(batch_step)
        train_losses.append(loss.item())
        print_training_progress(batch_idx, total_norm, loss.item(), batch_step, epoch, total_epochs, len(dataloader), pbar)
        gradient_norms.append(total_norm)
        epoch_loss += loss.item()
        batch_step += 1

        if val_dataloader is not None and (batch_idx % validation_interval == 0):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_batch = next(val_iter)
            val_loss = _run_validation_single_gpu(model, val_batch, device, use_amp, criterion)
            print(f"[Epoch {epoch} - Batch {batch_idx}] Validation Loss: {val_loss.item():.4f}")
            val_steps.append(batch_step)
            val_losses.append(val_loss.item())

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, len(dataloader), end_time - start_time)
    save_loss_plot(epoch, train_steps, train_losses, val_steps, val_losses, save_dir="dataset/validation_plots/loss")
    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")
    
    return batch_step

def train_one_epoch_multi_gpu(
    epoch,
    models,          # list or tuple of models (model[0] is primary)
    dataloader,
    criterion,
    optimizer,
    devices,         # list of torch.device objects corresponding to each model
    clip,
    batch_step=0,
    pbar=None,
    total_epochs=None,
    use_amp=False,
    grad_scaler=None,
    val_dataloader=None,
    validation_interval=20
):
    """
    Trains the supplied models for one epoch on multiple GPUs (up to 4) with mixed precision support.
    """
    n = len(models)
    steps_per_epoch = len(dataloader) // n  
    total_steps = total_epochs * steps_per_epoch
    epoch_loss = 0
    gradient_norms = []
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    data_iter = iter(dataloader)
    if val_dataloader is not None:
        val_iter = iter(val_dataloader)

    start_time = time.time()

    for step_idx in range(steps_per_epoch):
        batches = []
        try:
            for _ in range(n):
                batches.append(next(data_iter))
        except StopIteration:
            print(f"Dropping leftover mini-batches at step {step_idx}.")
            break

        inputs = []
        targets = []
        for i in range(n):
            src, trg = batches[i]
            inputs.append(src.to(devices[i], non_blocking=True))
            targets.append(trg.to(devices[i], non_blocking=True))

        optimizer.zero_grad()
        current_step = batch_step + (epoch * steps_per_epoch) + step_idx
        losses = _compute_losses_multi_gpu(models, inputs, targets, criterion, current_step, total_steps, use_amp)
        pre_clip_norm = _backward_and_step_multi_gpu(losses, models, optimizer, devices, clip, use_amp, grad_scaler)

        print_training_progress(step_idx, pre_clip_norm, sum(l.item() for l in losses)/n,
                                  batch_step, epoch, total_epochs, steps_per_epoch, pbar)
        gradient_norms.append(pre_clip_norm)

        _sync_models(models)

        batch_loss = sum(l.item() for l in losses) / n
        epoch_loss += batch_loss
        train_steps.append(batch_step)
        train_losses.append(batch_loss)
        gradient_norms.append(calculate_gradient_norm(models[0])) 
        batch_step += 1

        if val_dataloader is not None and (step_idx % validation_interval == 0):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_batch = next(val_iter)
            val_loss = _run_validation_multi_gpu(models[0], val_batch, devices[0], use_amp, criterion)
            print(f"[Epoch {epoch} - Step {step_idx}] Validation Loss: {val_loss.item():.4f}")
            val_steps.append(batch_step)
            val_losses.append(val_loss.item())

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, steps_per_epoch, end_time - start_time)
    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")
    save_loss_plot(epoch, train_steps, train_losses, val_steps, val_losses, save_dir="dataset/validation_plots/loss")

    return batch_step




