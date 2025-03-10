# validation.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from utils.audio.extraction.extract_features import extract_audio_features
from utils.audio.processing.audio_processing import process_audio_features
from utils.csv.save_csv import save_generated_data_as_csv
from utils.csv.plot_comparison import plot_comparison
from config import training_config


def save_generated_data(generated_facial_data, output_path):
    
    torch.save(generated_facial_data, output_path)

def generate_and_save_facial_data(epoch, audio_path, model, ground_truth_path, lock, device):
    
    audio_features, _ = extract_audio_features(audio_path)
       
    generated_facial_data = process_audio_features(audio_features, model, device, training_config)
    
    output_csv_path = f"generated_facial_data_epoch_{epoch + 1}.pth"

    with lock:
        csv_process = multiprocessing.Process(target=save_generated_data, args=(generated_facial_data, output_csv_path))
        csv_process.start()
        csv_process.join()

    # output_image_path = f"/home/xianchi/python/neurosync/trainer/dataset/validation_plots/comparison_plot_epoch_{epoch + 1}.jpg"
    
    # with lock:
    #     plot_process = multiprocessing.Process(target=plot_comparison, args=(ground_truth_path, output_csv_path, output_image_path))
    #     plot_process.start()
    #     plot_process.join()
            
    # # Save comparison statistics
    # output_stats_path = f"/home/xianchi/python/neurosync/trainer/dataset/validation_plots/stats/comparison_stats_epoch_{epoch + 1}.txt"
    # save_comparison_stats(output_csv_path, ground_truth_path, output_stats_path)

def save_comparison_stats(generated_data_path, ground_truth_path, output_stats_path):
    """
    Compute and save comparison statistics between generated and ground truth data, aligning lengths to the shortest.
    Save per-dimension statistics with labels.
    """
    # Define the dimension names in order
    dimension_labels = [
        'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft',
        'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight',
        'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward',
        'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker',
        'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
        'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft',
        'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower',
        'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
        'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft',
        'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff',
        'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut',
        'HeadYaw', 'HeadPitch', 'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll',
        'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll'
    ]

    # Load generated and ground truth data
    generated_data = pd.read_csv(generated_data_path)
    ground_truth_data = pd.read_csv(ground_truth_path)

    # Extract the required dimensions from the generated and ground truth data
    generated = generated_data.iloc[:, 2:2 + len(dimension_labels)].values
    ground_truth = ground_truth_data.iloc[:, 2:].values

    # Align the lengths to the shortest sequence
    min_length = min(generated.shape[0], ground_truth.shape[0])
    generated = generated[:min_length]
    ground_truth = ground_truth[:min_length]

    # Initialize a dictionary for per-dimension statistics
    per_dimension_stats = {}

    # Compute overall statistics
    diff = ground_truth - generated
    abs_diff = np.abs(diff)
    percentage_diff = (abs_diff / np.clip(np.abs(ground_truth), a_min=1e-6, a_max=None)) * 100

    overall_stats = {
        'Mean Absolute Error (MAE)': np.nanmean(abs_diff),
        'Mean Absolute Percentage Error (MAPE)': np.nanmean(percentage_diff),
        'Mean Squared Error (MSE)': np.nanmean(diff ** 2),
        'Root Mean Squared Error (RMSE)': np.sqrt(np.nanmean(diff ** 2)),
        'Correlation Coefficient (r)': (
            np.corrcoef(generated.flatten(), ground_truth.flatten())[0, 1]
            if np.nanstd(generated) > 1e-6 and np.nanstd(ground_truth) > 1e-6
            else float('nan')
        ),
    }

    # Compute per-dimension statistics
    for i, label in enumerate(dimension_labels):
        if np.nanstd(ground_truth[:, i]) > 1e-6:
            corr_coef = np.corrcoef(generated[:, i], ground_truth[:, i])[0, 1]
        else:
            corr_coef = float('nan')

        per_dimension_stats[label] = {
            'MAE': np.nanmean(abs_diff[:, i]),
            'MAPE': np.nanmean(percentage_diff[:, i]),
            'MSE': np.nanmean(diff[:, i] ** 2),
            'RMSE': np.sqrt(np.nanmean(diff[:, i] ** 2)),
            'Correlation Coefficient': corr_coef,
        }

    # Save statistics to a text file
    with open(output_stats_path, 'w') as stats_file:
        stats_file.write("Overall Comparison Statistics:\n")
        for stat_name, value in overall_stats.items():
            stats_file.write(f"{stat_name}: {value:.4f}\n")
        stats_file.write("\nPer-Dimension Statistics:\n")
        for label, stats in per_dimension_stats.items():
            stats_file.write(f"{label}:\n")
            for stat_name, value in stats.items():
                stats_file.write(f"  {stat_name}: {value:.4f}\n")

    print(f"Comparison statistics saved to {output_stats_path}")



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
            val_trg = val_trg[:, :, 69 : 211]
            val_loss = criterion(val_output, val_trg)
    model.train()  # Switch back to training mode
    return val_loss
