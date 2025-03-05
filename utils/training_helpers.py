# training_helpers.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def _compute_loss_single_gpu(model, src, trg, criterion, current_step, total_steps, use_amp):
    """
    Computes the loss for a single GPU batch using optional AMP.
    """
    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        output = model(src)
        loss = criterion(output, trg, current_step=current_step, total_steps=total_steps)
    return loss

def _backward_and_step_single_gpu(loss, model, optimizer, clip, use_amp, grad_scaler):
    """
    Backpropagates, clips gradients, and takes an optimizer step.
    Returns the computed gradient norm.
    """
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
    return total_norm

def _run_validation_single_gpu(model, val_batch, device, use_amp, criterion):
    """
    Runs a validation step for a single GPU.
    """
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        val_src, val_trg = val_batch
        val_src, val_trg = val_src.to(device), val_trg.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            val_output = model(val_src)
            val_loss = criterion(val_output, val_trg)
    model.train()  # Switch back to training mode
    return val_loss

# -----------------------------------------------------------------------------
# Helper functions for multi GPU training
# -----------------------------------------------------------------------------

def _compute_losses_multi_gpu(models, inputs, targets, criterion, current_step, total_steps, use_amp):
    """
    Computes losses for each GPU (each model in the list) using optional AMP.
    Returns a list of loss tensors.
    """
    losses = []
    if use_amp:
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            for i, model in enumerate(models):
                output = model(inputs[i])
                loss = criterion(output, targets[i], current_step=current_step, total_steps=total_steps)
                losses.append(loss)
    else:
        for i, model in enumerate(models):
            output = model(inputs[i])
            loss = criterion(output, targets[i], current_step=current_step, total_steps=total_steps)
            losses.append(loss)
    return losses

def _backward_and_step_multi_gpu(losses, models, optimizer, devices, clip, use_amp, grad_scaler):
    """
    Backpropagates on each GPU loss, unscales (if AMP), synchronizes gradients, averages them,
    clips gradients, and steps the optimizer.
    Returns the pre-clip gradient norm of the primary model.
    """
    n = len(models)
    if use_amp:
        for loss in losses:
            grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        # Manually unscale gradients for models[1:]
        scale = grad_scaler.get_scale()
        for i in range(1, n):
            for p in models[i].parameters():
                if p.grad is not None:
                    p.grad.data = p.grad.data / scale
    else:
        for loss in losses:
            loss.backward()

    # Synchronize all devices.
    for d in devices:
        torch.cuda.synchronize(d)

    # Gradient Averaging: Move all gradients to devices[0] and average them.
    for param_tuple in zip(*[m.parameters() for m in models]):
        if all(p.grad is not None for p in param_tuple):
            grad_list = [p.grad.data.to(devices[0]) for p in param_tuple]
            avg_grad = sum(grad_list) / n
            param_tuple[0].grad.data.copy_(avg_grad.view_as(param_tuple[0]))

    pre_clip_norm = calculate_gradient_norm(models[0])  
    torch.nn.utils.clip_grad_norm_(models[0].parameters(), clip)

    if use_amp:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()

    return pre_clip_norm

def _sync_models(models):
    """
    Synchronizes parameters from the primary model (models[0]) to all other models
    and zeros their gradients.
    """
    for m in models[1:]:
        for p0, p_other in zip(models[0].parameters(), m.parameters()):
            p_other.data.copy_(p0.data.to(p_other.device))
    # Zero gradients for models[1:].
    for m in models[1:]:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()

def _run_validation_multi_gpu(model, val_batch, device, use_amp, criterion):
    """
    Runs a validation step using the primary model (for multi GPU training).
    """
    model.eval()  # Use primary model for validation.
    with torch.no_grad():
        val_src, val_trg = val_batch
        val_src, val_trg = val_src.to(device), val_trg.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            val_output = model(val_src)
            val_loss = criterion(val_output, val_trg)
    model.train()
    return val_loss


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
