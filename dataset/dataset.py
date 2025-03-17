# dataset.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from dataset.data_processing import load_data, process_folder  


# =============================================================================
# PREPROCESSING FOR LAZY MODE: Save Entire Data to .npy Files & Build an Index
#
# For each folder in root_dir, we load the full audio and facial arrays (via load_data)
# and save them as .npy files in "processed_bin". Then, we compute the number
# of sliding window segments (of micro_batch_size frames) and record an index
# entry for each complete window. (Incomplete windows are dropped.)
# =============================================================================
def preprocess_and_cache_to_bin(config, force_reprocess=False):
    """
    Process each folder’s full data arrays, save them as .npy files, and build a global
    index mapping each micro-batch (of micro_batch_size frames) to its source file and start index.
    
    Parameters:
      config         - dict with keys: 'root_dir', 'sr', 'micro_batch_size', etc.
      force_reprocess- if True, re-save even if .npy files exist.
    
    Returns:
      dataset_index  - a list of tuples (audio_bin_file, facial_bin_file, start_index)
    """
    root_dir = config['root_dir']
    sr = config['sr']
    micro_batch_size = config['micro_batch_size']

    bin_dir = os.path.join(root_dir, "processed_bin")
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    
    processed_folders = set()
    dataset_index = []

    raw_examples = load_data(root_dir, sr, processed_folders)
    
    for (audio_features, facial_data) in raw_examples:
        # Create a unique name for this folder's binary files.
        folder_name = "folder_" + str(len(dataset_index))
        
        audio_bin_file = os.path.join(bin_dir, f"{folder_name}_audio.npy")
        facial_bin_file = os.path.join(bin_dir, f"{folder_name}_facial.npy")

        if force_reprocess or (not os.path.exists(audio_bin_file) or not os.path.exists(facial_bin_file)):
            np.save(audio_bin_file, audio_features)
            np.save(facial_bin_file, facial_data)
            print(f"Saved binary files for folder '{folder_name}'")
        
        T = audio_features.shape[0]
        # Only record windows that completely fit (drop excess frames).
        if T < micro_batch_size:
            continue
        
        num_segments = T - micro_batch_size + 1  # sliding window with stride 1
        for start in range(0, num_segments):
            dataset_index.append((audio_bin_file, facial_bin_file, start))

    index_file = os.path.join(bin_dir, "dataset_index.pkl")
    with open(index_file, "wb") as f:
        pickle.dump(dataset_index, f)
    print(f"Saved global dataset index with {len(dataset_index)} segments to {index_file}")
    
    return dataset_index


# =============================================================================
# DATASET CLASS (IN-MEMORY VERSION)
#
# This version loads all processed examples into memory.
# It uses your original sliding window logic and includes reflection padding.
# =============================================================================
class InMemoryAudioFacialDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config['root_dir']
        self.sr = config['sr']
        self.frame_rate = config['frame_rate']
        self.micro_batch_size = config['micro_batch_size']
        self.examples = []
        self.processed_folders = set()

        raw_examples = load_data(self.root_dir, self.sr, self.processed_folders)
        
        for audio_features, facial_data in raw_examples:
            processed_examples = self.process_example(audio_features, facial_data)
            if processed_examples is not None:
                self.examples.extend(processed_examples)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch

    def process_example(self, audio_features, facial_data):
        num_frames_facial = len(facial_data)
        num_frames_audio = len(audio_features)
        max_frames = max(num_frames_audio, num_frames_facial)

        examples = []
        # Process full windows
        for start in range(0, max_frames - self.micro_batch_size + 1):
            end = start + self.micro_batch_size
            
            audio_segment = np.zeros((self.micro_batch_size, audio_features.shape[1]))
            facial_segment = np.zeros((self.micro_batch_size, facial_data.shape[1]))
            
            audio_segment[:min(self.micro_batch_size, num_frames_audio - start)] = audio_features[start:end]
            facial_segment[:min(self.micro_batch_size, num_frames_facial - start)] = facial_data[start:end]

            examples.append((torch.tensor(audio_segment, dtype=torch.float32),
                             torch.tensor(facial_segment, dtype=torch.float32)))

        # Handle incomplete (reflection) window if needed.
        if max_frames % self.micro_batch_size != 0:
            start = max_frames - self.micro_batch_size
            end = max_frames

            audio_segment = np.zeros((self.micro_batch_size, audio_features.shape[1]))
            facial_segment = np.zeros((self.micro_batch_size, facial_data.shape[1]))
            
            segment_audio = audio_features[start:end]
            segment_facial = facial_data[start:end]

            reflection_audio = np.flip(segment_audio, axis=0)
            reflection_facial = np.flip(segment_facial, axis=0)

            audio_segment[:len(segment_audio)] = segment_audio
            audio_segment[len(segment_audio):] = reflection_audio[:self.micro_batch_size - len(segment_audio)]

            facial_segment[:len(segment_facial)] = segment_facial
            facial_segment[len(segment_facial):] = reflection_facial[:self.micro_batch_size - len(segment_facial)]

            examples.append((torch.tensor(audio_segment, dtype=torch.float32),
                             torch.tensor(facial_segment, dtype=torch.float32)))
        
        return examples


# =============================================================================
# DATASET CLASS (LAZY / DISK-BASED VERSION)
#
# This version loads only the index into memory. At each __getitem__ call,
# it memory-maps the corresponding .npy files and slices out a micro-batch.
# Incomplete windows are dropped (no reflection padding).
# =============================================================================
class LazyAudioFacialDataset(Dataset):
    def __init__(self, config, preload_index=True, force_reprocess=False):
        """
        Parameters:
          config         - dict with keys: 'root_dir', 'sr', 'micro_batch_size', etc.
          preload_index  - if True, load the index from disk if available.
          force_reprocess- if True, ignore existing .npy files/index and reprocess.
        """
        self.config = config
        self.root_dir = config['root_dir']
        self.micro_batch_size = config['micro_batch_size']
        
        self.bin_dir = os.path.join(self.root_dir, "processed_bin")
        self.index_file = os.path.join(self.bin_dir, "dataset_index.pkl")

        if preload_index and os.path.exists(self.index_file) and not force_reprocess:
            with open(self.index_file, "rb") as f:
                self.dataset_index = pickle.load(f)
            print(f"Loaded dataset index with {len(self.dataset_index)} segments from {self.index_file}")
        else:
            print("Preprocessing data to binary files and building dataset index...")
            self.dataset_index = preprocess_and_cache_to_bin(config, force_reprocess=force_reprocess)
    
    def __len__(self):
        return len(self.dataset_index)
    
    def __getitem__(self, idx):
        # Unpack the index tuple.
        audio_file, facial_file, start = self.dataset_index[idx]
        micro_batch_size = self.micro_batch_size

        # Memory-map the full arrays.
        audio_data = np.load(audio_file, mmap_mode='r')
        facial_data = np.load(facial_file, mmap_mode='r')
        # Slice out a full window.
        audio_segment = audio_data[start : start + micro_batch_size]
        facial_segment = facial_data[start : start + micro_batch_size]
        
        return (torch.tensor(audio_segment, dtype=torch.float32),
                torch.tensor(facial_segment, dtype=torch.float32))
    
    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch


# =============================================================================
# HELPER: Get the Dataset Class Based on Configuration
#
# If config['in_memory'] is True, use the in-memory version;
# otherwise, use the lazy (disk-based) version.
# =============================================================================
def get_dataset_class(config):
    if config.get("in_memory", True):
        return InMemoryAudioFacialDataset
    else:
        return LazyAudioFacialDataset


# =============================================================================
# DATALOADER PREPARATION FUNCTIONS
#
# These functions create DataLoaders (with optional train/validation split)
# based on the chosen dataset class.
# =============================================================================
def prepare_dataloader_with_split(config, val_split=0.1):
    DatasetClass = get_dataset_class(config)
    dataset = DatasetClass(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=DatasetClass.collate_fn,
        num_workers=4,         # adjust as needed
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=DatasetClass.collate_fn,
        num_workers=4,
        prefetch_factor=2
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    DatasetClass = get_dataset_class(config)
    dataset = DatasetClass(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=DatasetClass.collate_fn,
        num_workers=4,
        prefetch_factor=2
    )
    return dataset, dataloader




