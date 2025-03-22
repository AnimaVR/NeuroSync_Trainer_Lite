# NeuroSync Trainer Lite

## Using the NeuroSync Model after you have trained it.
To **generate facial blendshapes from audio** and send them to Unreal Engine, you'll need:
- [**NeuroSync Local API**](https://github.com/AnimaVR/NeuroSync_Local_API) – Handles real-time facial data processing.
- [**NeuroSync Player**](https://github.com/AnimaVR/NeuroSync_Player) – Sends the animation data to Unreal Engine or any **LiveLink-compatible** software.

## Overview

**NeuroSync Trainer Lite** is an Open Source Audio2Face tool for training an **audio-to-face blendshape transformer model**, enabling the generation of facial animation from audio input. This is useful for real-time applications like **virtual avatars, game characters, and animation pipelines**.

## Features
- **Audio-Driven Facial Animation** – Train a model to generate realistic blendshape animations from audio input.
- **Multi-GPU Support** – Train efficiently using up to 4 GPUs.
- **Integration with Unreal Engine** – Send trained animation data to Unreal Engine via **NeuroSync Local API** and **NeuroSync Player**.
- **Optimized for iPhone Face Data** – Easily process facial motion capture data from an iPhone.

## Quick Start
### 1. Install Dependencies
Before training, ensure you have the required dependencies installed, including:
- Python 3.9+
- PyTorch with CUDA support (for GPU acceleration)
- NumPy, Pandas, Librosa, OpenCV, Matplotlib, and other required Python libraries
- FFMPEG for linux should be installed globally, Windows users need to get a compiled ffmpeg.exe and put it inside utils\video\ _ffmpeg\bin to correctly strip the audio from the .mov in the face data folders....

### 2. Collect & Prepare Data
To train the model, you need audio and calibrated facial blendshape data. 

**Ensure you 'calibrate' in the LiveLink app before you record your data. This ensures your resting face is 0 or close to 0 in all dimensions.**

Follow these steps:
1. **Record Face & Audio Data** using an iPhone and LiveLink app utilizing **ARKit Blendshapes** as the type of data collected (NOT Metahuman Animator).
2. **Download & Extract the Data** to your local machine.
3. **Move Data to the Correct Folder:**
   - Place each extracted folder inside `dataset/data/`.

If you want a universal model (any voice) duplicate data with voice to voice using ElevenLabs or similar multiple times for multiple voice types and use that data to train. 

For one actor, at least 30 minutes of data is required. The more data the better! (caveat : if you want a universal model 8 voices at 30 mins each would require 256gb of system memory at the current set batch size as an example).

For better results, record the audio externally and time it with the mov then replace the mov with a wav - it will work better to have cleaner audio than the iPhone provides. More samples seems to work better (hence 88200, you can reduce this to 16000 if you want).

### 3. Train the Model
Once your data is ready, start training by running:
```bash
python train.py
```

## Multi-GPU Training
If you want to train using multiple GPUs, update the **configuration file**:
1. Open `config.py`.
2. Set `use_multi_gpu = True`.
3. Define the number of GPUs:
   ```python
   'use_multi_gpu': True,
   'num_gpus': 4  # Adjust as needed, max 4 GPUs
   ```
4. Start training as usual.

You can easily modify the code to support **more than 4 GPUs**—just ask ChatGPT for assistance!

## License
This software is licensed under a dual-license model:

1️⃣ For individuals and businesses earning under $1M per year:

Licensed under the MIT License
You may use, modify, distribute, and integrate the software for any purpose, including commercial use, free of charge.

2️⃣ For businesses earning $1M or more per year:

- A commercial license is required for continued use.
- Contact us to obtain a commercial license.
- By using this software, you agree to these terms.

📜 For more details, see LICENSE.md or contact us.

&copy; 2025 NeuroSync Trainer Lite

