# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import platform

# # Detect OS
# is_windows = platform.system() == "Windows"

# # Get root directory
# root_dir = os.path.dirname(os.path.abspath(__file__))

# # Set ffmpeg path based on OS
# if is_windows:
#     ffmpeg_path = os.path.join(root_dir, 'utils', 'video', '_ffmpeg', 'bin', 'ffmpeg.exe')
    
#     # Check if ffmpeg.exe exists
#     if not os.path.exists(ffmpeg_path):
#         raise FileNotFoundError(
#             f"FFmpeg not found! Please download FFmpeg for Windows from:\n"
#             f"https://www.ffmpeg.org/download.html\n\n"
#             f"Once downloaded, place 'ffmpeg.exe' in:\n"
#             f"{os.path.dirname(ffmpeg_path)}"
#         )
# else:
#     ffmpeg_path = 'ffmpeg'  # Use system-wide ffmpeg on Linux/macOS


training_config = { 
    'mode': 'scratch',       # Training mode: 'scratch' or 'resume'
    'sr': 16000,             # Sample rate 
    'frame_rate': 30,        # Frame rate for facial data
    'hidden_dim': 1024,      # Hidden dimension for the model ### increases increase GPU memory requirements a lot.
    'n_layers': 8,           # Number of layers in the model
    'num_heads': 16,         # Number of attention heads
    'dropout': 0.3,          # Dropout rate
    'batch_size':  128,      # Batch size ## REDUCE THIS IF < 24GB GPU
    'micro_batch_size': 128, # 64 or 128 is sweet spot # If you increase this you need to reduce the batch size - lower = faster inference at accuracy cost, higher = more accuracy at longer inference.
    'learning_rate': 5e-4,   # Learning rate
    'weight_decay': 1e-5,    # Weight decay for the optimizer
    'n_epochs': 50,          # Number of training epochs
    'output_dim': 142 ,        # Use 61 if training on iPhone data alone. On the model available on huggingface, this is 68 because we add dimensions for emotion. hstack extra data to out for more data out.       
    'delta': 1,              # Delta for Huber loss
    'w1': 1.0,               # Weight for Huber loss
    'w2': 1.0, 
    'w3': 1.0, 
    'use_multi_gpu' : False,   
    'num_gpus' : 1,               
    'warmup_epochs': 0, 
    'input_dim': 256,  
    'frame_size': 128,
    # 'ffmpeg_path': ffmpeg_path,  
    'root_dir': os.path.join("dataset", "data_mh"),     
    'model_path': os.path.join("out", "model.pth"),
    'audio_path': os.path.join("dataset", "test_set", "audio.wav"),
    'ground_truth_path': os.path.join("dataset", "test_set", "testset.csv"),
    'checkpoint_path': os.path.join("out", "checkpoints", "checkpoint.pth"), 
    'use_amp': True,
    'in_memory' : True, # if true, use in memory data storage - requires a lot of system memory if your dataset is large and is no quicker than if using lazy loading - just dont ;)
    'freeze_arkit_generator': True,
    'with_emotions': False,
    'arkit_generator_model_path': os.path.join("out", "model.pth")
}

