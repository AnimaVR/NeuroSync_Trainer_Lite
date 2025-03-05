# NeuroSync Trainer Lite

## Using the NeuroSync Model after you have trained it.
To **generate facial blendshapes from audio** and send them to Unreal Engine, you'll need:
- [**NeuroSync Local API**](https://github.com/AnimaVR/NeuroSync_Local_API) – Handles real-time facial data processing.
- [**NeuroSync Player**](https://github.com/AnimaVR/NeuroSync_Player) – Sends the animation data to Unreal Engine or any **LiveLink-compatible** software.

### **04/03/2025 Caching Dataloader for low memory capacity systems**

If you dont have very much system memory, training large datasets was impossible - a cache dataloader sample has been added (commented out on dataset.py) so you can have an dataset as large as your hard drive and it still be as performant as an in memory alternative.

This is WIP and not fully tested yet, please beware!


### **21/02/2025 Scaling UP! | New 228m parameter model + config added**

A milestone has been hit and previous research has got us to a point where scaling the model up is now possible with much faster training and better quality overall.

Going from 4 layers and 4 heads to 8 layers and 16 heads means updating your code and model, please ensure you have the latest versions of the api and player as the new model requires some architectural changes.

Enjoy!

### **19/02/2025 Trainer updates**

- **Trainer**: Use [NeuroSync Trainer Lite](https://github.com/AnimaVR/NeuroSync_Trainer_Lite) for training and fine-tuning.

- **Simplified Loss** Removed second order smoothness loss (left code in if you want to research the differences, mostly it just squeezes the end result resulting in choppy animation without smoothing)
- **Mixed Precision** Less memory usage and faster training
- **Data augmentation** Interpolate a slow set and a fast set of data from your data to help with fine detail reproduction, uses a lot of memory so /care - generally just adding the fast is best as adding slow over saturates the data with slow and noisey data (more work to do here... obv's!)


## **Validation**

Loss validation + plotting added

Refactored and optimised multi GPU processing (yey!)

## **Loss Variations** (utils/model.py)

Have added a few types of loss you can uncomment and use to check what works best for you, the new type that penalises known zero'd dimensions seems to work well, (if you are zero'ing any dimensions). **21.02.2025 update to latest version of model.py for the most reliable loss, others still present for research**

Have a play around, I will get around to some better validation soon. ;) **validation now present**

## Interpolate slower and faster versions of your data automatically in data_processing.py with def collect_features(audio_path, audio_features_csv_path, facial_csv_path, sr, include_fast=True, include_slow=False, blend_boundaries=True, blend_frames=30):

Careful, this increases memory usage on the system, a lot.... but it makes fine detail clearer as speed variance is better realised - turn it off if you have 16gb of system memory, use at least 128gb, 256gb > is recommended for larger datasets.

## Single + multi GPU mixed precison training added for 2x speed improvement.

Still wip but its working - disable it if you have issues.

## Info sheet added (it's a paper, but not a paper - if you know what I mean)

[**Download the info sheet here**](https://drive.google.com/file/d/1U9pvs_FY1L-cnSkWvnkbSPe0VVa8PZ8b/view?usp=drive_link)

I have been asked for more technical information, please see above.

## RoPe + Global/Local positional encoding.

Turns out, RoPe and combining global and local positioning is yielding much better results. 

They are enabled now in the trainer, just update your code. For now, check that these bools are also set to True in the api's model.py too when testing (it will be default soon after the model is updated on huggingface)

## New open source anonymous dataset available
### [**Sample dataset from Huggingface**](https://huggingface.co/datasets/AnimaVR/Neurosync_Audio2Face_Dataset) 

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

