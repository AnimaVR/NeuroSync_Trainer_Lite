# validation.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import multiprocessing
import numpy as np
import pandas as pd

from utils.audio.extraction.extract_features import extract_audio_features
from utils.audio.processing.audio_processing import process_audio_features
from utils.csv.save_csv import save_generated_data_as_csv
from utils.csv.plot_comparison import plot_comparison
from config import training_config

def generate_and_save_facial_data(epoch, audio_path, model, ground_truth_path, lock, device):
    audio_features, _ = extract_audio_features(audio_path)
    generated_facial_data = process_audio_features(audio_features, model, device, training_config)

    # Define output paths
    base_dir = "dataset/validation_plots"
    stats_dir = os.path.join(base_dir, "stats")

    # Ensure required directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    output_csv_path = os.path.join(base_dir, f"generated_facial_data_epoch_{epoch + 1}.csv")

    with lock:
        csv_process = multiprocessing.Process(target=save_generated_data_as_csv, args=(generated_facial_data, output_csv_path))
        csv_process.start()
        csv_process.join()

    output_image_path = os.path.join(base_dir, f"comparison_plot_epoch_{epoch + 1}.jpg")

    with lock:
        plot_process = multiprocessing.Process(target=plot_comparison, args=(ground_truth_path, output_csv_path, output_image_path))
        plot_process.start()
        plot_process.join()
    
    # Save comparison statistics
    output_stats_path = os.path.join(stats_dir, f"comparison_stats_epoch_{epoch + 1}.txt")
    save_comparison_stats(output_csv_path, ground_truth_path, output_stats_path)


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

    # Compute percentage difference safely
    percentage_diff = np.divide(
        abs_diff, 
        np.abs(ground_truth), 
        out=np.zeros_like(abs_diff),  # Handle division by zero
        where=np.abs(ground_truth) > 1e-6  # Only divide where valid
    ) * 100

    # Ensure no NaNs in the final array
    percentage_diff = np.nan_to_num(percentage_diff, nan=0.0, posinf=0.0, neginf=0.0)


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
        if np.nanstd(ground_truth[:, i]) > 1e-6 and np.nanstd(generated[:, i]) > 1e-6:
            corr_coef = np.corrcoef(generated[:, i], ground_truth[:, i])[0, 1]
        else:
            corr_coef = float('nan')  # Avoid invalid correlation calculations

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
