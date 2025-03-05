# training_utils.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import os
import torch.distributed as dist

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
    Trains the model for one epoch on a single GPU, optionally using mixed precision.
    Additionally, if a validation DataLoader is provided, it performs a validation
    step every 'validation_interval' training batches.
    
    NOTE:
      - For the criterion (and scheduler), current_step is computed as:
            current_step = batch_step + (epoch * len(dataloader)) + batch_idx
        to maintain the expected behavior.
      - For plotting, we record only the global batch_step (which is incremented each batch)
        so that the x-axis does not show double the number of steps.
    """
    if use_amp and grad_scaler is None:
        raise ValueError("use_amp=True but no GradScaler was provided!")

    model.train()
    epoch_loss = 0
    start_time = time.time()

    total_steps = total_epochs * len(dataloader)  # Total training steps (if needed by the criterion)
    gradient_norms = []

    # Lists to track losses for plotting (using the global batch_step, not the doubled current_step)
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    # Initialize validation iterator if provided
    if val_dataloader is not None:
        val_iter = iter(val_dataloader)

    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        # --------------------------#
        #    Mixed Precision      #
        # --------------------------#
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            # Keep the current_step calculation for the criterion:
            current_step = batch_step + (epoch * len(dataloader)) + batch_idx
            loss = criterion(model(src), trg, current_step=current_step, total_steps=total_steps)

        if use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            total_norm = calculate_gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            total_norm = calculate_gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # For plotting, record the global batch_step (which is incremented every batch)
        train_steps.append(batch_step)
        train_losses.append(loss.item())

        print_training_progress(batch_idx, total_norm, loss.item(), batch_step, epoch, total_epochs, len(dataloader), pbar)

        gradient_norms.append(total_norm)
        epoch_loss += loss.item()
        batch_step += 1

        # -------------------------------
        # Validation Step Every 'validation_interval' Batches
        # -------------------------------
        if val_dataloader is not None and (batch_idx % validation_interval == 0):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                # Restart the iterator if exhausted
                val_iter = iter(val_dataloader)
                val_batch = next(val_iter)
            model.eval()  # Switch to evaluation mode
            with torch.no_grad():
                val_src, val_trg = val_batch
                val_src, val_trg = val_src.to(device), val_trg.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    val_output = model(val_src)
                    val_loss = criterion(val_output, val_trg)
            print(f"[Epoch {epoch} - Batch {batch_idx}] Validation Loss: {val_loss.item():.4f}")
            # Record validation loss using the global batch_step value
            val_steps.append(batch_step)
            val_losses.append(val_loss.item())
            model.train()  # Switch back to training mode

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, len(dataloader), end_time - start_time)

    # Plot and save the loss curves.
    # The helper function 'save_loss_plot' will create the directory if needed.
    save_loss_plot(epoch, train_steps, train_losses, val_steps, val_losses, save_dir="dataset/validation_plots/loss")

    # Also save the gradient norm plot (existing functionality)
    save_gradient_norm_plot(
        epoch,
        gradient_norms,
        save_dir="dataset/validation_plots/gradient_norms"
    )

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
    
    For the criterion (and scheduler), current_step is computed as:
         current_step = batch_step + (epoch * steps_per_epoch) + step_idx
    to maintain the expected behavior.
    
    For plotting we record only the global batch_step (which increments once per iteration).
    
    Gradients for each corresponding parameter are moved to devices[0] and averaged into models[0].
    The optimizer is assumed to be tied to models[0], and its parameters are synchronized to the others.
    """
    n = len(models)  # number of models/GPUs used
    steps_per_epoch = len(dataloader) // n  
    total_steps = total_epochs * steps_per_epoch
    epoch_loss = 0
    gradient_norms = []

    # Lists to track losses for plotting (using the global batch_step)
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    data_iter = iter(dataloader)
    if val_dataloader is not None:
        val_iter = iter(val_dataloader)

    # Initialize timer here
    start_time = time.time()

    for step_idx in range(steps_per_epoch):
        # Fetch one batch for each model.
        batches = []
        try:
            for _ in range(n):
                batches.append(next(data_iter))
        except StopIteration:
            print(f"Dropping leftover mini-batches at step {step_idx}.")
            break

        # Move each batch to its corresponding device.
        inputs = []
        targets = []
        for i in range(n):
            src, trg = batches[i]
            inputs.append(src.to(devices[i], non_blocking=True))
            targets.append(trg.to(devices[i], non_blocking=True))

        optimizer.zero_grad()

        # Compute current_step for the criterion.
        current_step = batch_step + (epoch * steps_per_epoch) + step_idx

        losses = []
        if use_amp:
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                for i in range(n):
                    out = models[i](inputs[i])
                    loss_i = criterion(out, targets[i], current_step=current_step, total_steps=total_steps)
                    losses.append(loss_i)
            for loss in losses:
                grad_scaler.scale(loss).backward()
        else:
            for i in range(n):
                out = models[i](inputs[i])
                loss_i = criterion(out, targets[i], current_step=current_step, total_steps=total_steps)
                losses.append(loss_i)
                loss_i.backward()

        # If using AMP, unscale gradients for models[1:] manually.
        if use_amp:
            grad_scaler.unscale_(optimizer)
            scale = grad_scaler.get_scale()
            for i in range(1, n):
                for p in models[i].parameters():
                    if p.grad is not None:
                        p.grad.data = p.grad.data / scale

        # Synchronize all devices.
        for d in devices:
            torch.cuda.synchronize(d)

        # ----- Gradient Averaging Across GPUs -----
        for param_tuple in zip(*[m.parameters() for m in models]):
            if all(p.grad is not None for p in param_tuple):
                # Move every gradient to devices[0]
                grad_list = [p.grad.data.to(devices[0]) for p in param_tuple]
                avg_grad = sum(grad_list) / n
                # Copy the averaged gradient into the corresponding parameter of models[0]
                param_tuple[0].grad.data.copy_(avg_grad.view_as(param_tuple[0]))

        # Compute the pre-clip gradient norm (for printing and logging)
        pre_clip_norm = calculate_gradient_norm(models[0])
        # Use pre_clip_norm in our progress print function (like in the single-GPU version)
        print_training_progress(step_idx, pre_clip_norm, sum(l.item() for l in losses)/n,
                                  batch_step, epoch, total_epochs, steps_per_epoch, pbar)
        gradient_norms.append(pre_clip_norm)

        # Clip gradients on the primary model.
        torch.nn.utils.clip_grad_norm_(models[0].parameters(), clip)

        # Optimizer step and update scaler if using AMP.
        if use_amp:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        # Synchronize updated parameters from models[0] to all other models.
        for m in models[1:]:
            for p0, p_other in zip(models[0].parameters(), m.parameters()):
                p_other.data.copy_(p0.data.to(p_other.device))

        # Zero gradients for models[1:].
        for m in models[1:]:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Compute average loss from all GPUs.
        batch_loss = sum(l.item() for l in losses) / n
        epoch_loss += batch_loss

        # Record training loss using the global batch_step.
        train_steps.append(batch_step)
        train_losses.append(batch_loss)

        gradient_norms.append(calculate_gradient_norm(models[0]))
        batch_step += 1

        # ----- Validation Step -----
        if val_dataloader is not None and (step_idx % validation_interval == 0):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                val_batch = next(val_iter)
            models[0].eval()  # Use the primary model for validation.
            with torch.no_grad():
                val_src, val_trg = val_batch
                val_src, val_trg = val_src.to(devices[0]), val_trg.to(devices[0])
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    val_output = models[0](val_src)
                    val_loss = criterion(val_output, val_trg)
            print(f"[Epoch {epoch} - Step {step_idx}] Validation Loss: {val_loss.item():.4f}")
            val_steps.append(batch_step)
            val_losses.append(val_loss.item())
            models[0].train()

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, steps_per_epoch, end_time - start_time)
    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")
    save_loss_plot(epoch, train_steps, train_losses, val_steps, val_losses, save_dir="dataset/validation_plots/loss")

    return batch_step





def save_loss_plot(epoch, train_steps, train_losses, val_steps, val_losses, save_dir="dataset/validation_plots/loss"):
    """
    Save a plot of the training and validation losses over an epoch.

    :param epoch: The current epoch (zero-indexed).
    :param train_steps: A list of training step indices (e.g., [0, 1, 2, ...]).
    :param train_losses: A list of training loss values recorded at each training step.
    :param val_steps: A list of training step indices at which validation was performed.
    :param val_losses: A list of validation loss values recorded at those steps.
    :param save_dir: Directory where the loss plot will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Training Loss", marker='o', markersize=3)
    plt.plot(val_steps, val_losses, label="Validation Loss", marker='x', markersize=8, linestyle='--')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Values (Epoch {epoch + 1})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"loss_epoch_{epoch + 1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")



def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        print(f"Initializing {m} with normal distribution")
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def count_parameters(model):
    """Count and print the number of parameters in a model."""
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {param_count}")
    return param_count

def calculate_gradient_norm(model):
    """Calculate and return the gradient norm for the model."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def print_training_progress(batch_idx, total_norm, batch_loss, batch_step, epoch, total_epochs, dataloader_len, pbar):
    """Print training progress and update the progress bar."""
    print(f"Batch {batch_idx}, Gradient Norm: {total_norm}")
    if pbar is not None:
        pbar.update(1)
    print(f"Step [{batch_step}/{pbar.total}], Epoch [{epoch + 1}/{total_epochs}], Batch [{batch_idx + 1}/{dataloader_len}], Current Loss: {batch_loss:.4f}")

def print_epoch_summary(epoch, total_epochs, epoch_loss, dataloader_len, epoch_time):
    """Print the summary of the epoch."""
    print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {epoch_loss / dataloader_len:.4f}, Time: {epoch_time:.2f} seconds")

def save_gradient_norm_plot(epoch, gradient_norms, save_dir):
    """Save a plot of gradient norms over the batches in an epoch."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label="Gradient Norm")
    plt.xlabel("Batch Index")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm Fluctuations (Epoch {epoch + 1})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"gradient_norms_epoch_{epoch + 1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Gradient norm plot saved to {plot_path}")

