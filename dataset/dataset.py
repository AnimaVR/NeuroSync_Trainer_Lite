# dataset.py

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

from dataset.data_processing import load_data


def prepare_dataloader_with_split(config, val_split=0.1):
    dataset = AudioFacialDataset(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=AudioFacialDataset.collate_fn)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    dataset = AudioFacialDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=AudioFacialDataset.collate_fn)
    return dataset, dataloader

class AudioFacialDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config['root_dir']
        self.sr = config['sr']
        self.frame_rate = config['frame_rate']
        self.micro_batch_size = config['micro_batch_size']
        self.examples = []
        self.processed_folders = set()

        raw_examples = load_data(self.root_dir, self.sr, self.processed_folders)
        
        self.examples = []
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
        for start in range(0, max_frames - self.micro_batch_size + 1):
            end = start + self.micro_batch_size
            
            audio_segment = np.zeros((self.micro_batch_size, audio_features.shape[1]))
            facial_segment = np.zeros((self.micro_batch_size, facial_data.shape[1]))
            
            audio_segment[:min(self.micro_batch_size, num_frames_audio - start)] = audio_features[start:end]
            facial_segment[:min(self.micro_batch_size, num_frames_facial - start)] = facial_data[start:end]

            examples.append((torch.tensor(audio_segment, dtype=torch.float32), torch.tensor(facial_segment, dtype=torch.float32)))

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

            examples.append((torch.tensor(audio_segment, dtype=torch.float32), torch.tensor(facial_segment, dtype=torch.float32)))
        
        return examples



'''

# If you have a dataset larger than your system memory, you can use the dataloader below to load from disk.

# WIP so please take care.

# dataset.py

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from dataset.data_processing import load_data, process_folder  


# =============================================================================
# PREPROCESSING: Save Entire Data to .npy (Binary) Files & Build an Index
#
# For each folder in root_dir, we use your existing processing (or cached .npz)
# to load the full audio and facial arrays. Then, we store each array as a .npy
# file in a new folder ("processed_bin"). We also compute how many sliding
# window segments of micro_batch_size (e.g. 128 frames) are available and record
# an index entry for each segment.
# =============================================================================



def preprocess_and_cache_to_bin(config, force_reprocess=False):
    """
    Preprocess each folder’s entire data arrays and save them as .npy files.
    Then build a global index mapping each micro-batch (of micro_batch_size frames)
    to its source files and starting index.
    
    If force_reprocess is True, we re-save even if the .npy files already exist.
    """
    root_dir = config['root_dir']
    sr = config['sr']
    micro_batch_size = config['micro_batch_size']
    
    # Directory for binary (npy) files
    bin_dir = os.path.join(root_dir, "processed_bin")
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    
    # Set to keep track of processed folders (this mimics your original code)
    processed_folders = set()
    # Global index: each entry is a tuple:
    # (audio_npy_file, facial_npy_file, start_index, reflection_flag)
    dataset_index = []
    
    # Use your load_data to process (or load cached) full arrays per folder.
    raw_examples = load_data(root_dir, sr, processed_folders)
    
    # Loop over each folder’s full data
    for (audio_features, facial_data) in raw_examples:
        # Determine a unique folder name from the cache filename (assume caching preserved name)
        # (Here we simply use the folder name stored in the file name of one of the npz files.
        #  Adjust if needed.)
        folder_name = "folder_" + str(len(dataset_index))  # fallback if no proper name available
        
        # (If you want to extract the folder name from the cache file path, you could modify load_data.)
        # For demonstration, we assume each raw_example corresponds to one folder.
        audio_bin_file = os.path.join(bin_dir, f"{folder_name}_audio.npy")
        facial_bin_file = os.path.join(bin_dir, f"{folder_name}_facial.npy")
        
        # Save the entire arrays as .npy binary files if not already present or if forced.
        if force_reprocess or (not os.path.exists(audio_bin_file) or not os.path.exists(facial_bin_file)):
            np.save(audio_bin_file, audio_features)
            np.save(facial_bin_file, facial_data)
            print(f"Saved binary files for folder '{folder_name}'")
        
        # Both arrays should have the same number of frames.
        T = audio_features.shape[0]
        if T < micro_batch_size:
            # Skip if not enough frames for a mini-batch.
            continue
        
        # Compute the number of regular segments (using a sliding window of stride 1)
        num_regular_segments = T - micro_batch_size + 1
        
        # If T is not a multiple of micro_batch_size, we add one extra segment using reflection.
        add_reflection = (T % micro_batch_size != 0)
        
        # Record an index entry for each regular segment.
        for start in range(0, num_regular_segments):
            dataset_index.append((audio_bin_file, facial_bin_file, start, False))
        
        # Record the extra (reflection) segment if needed.
        if add_reflection:
            # For the reflection segment we use start = T - micro_batch_size.
            dataset_index.append((audio_bin_file, facial_bin_file, T - micro_batch_size, True))
    
    # Optionally, save the global index to disk for faster reloading later.
    index_file = os.path.join(bin_dir, "dataset_index.pkl")
    with open(index_file, "wb") as f:
        pickle.dump(dataset_index, f)
    print(f"Saved global dataset index with {len(dataset_index)} segments to {index_file}")
    
    return dataset_index

# =============================================================================
# DATASET CLASS: Lazy-Loading Mini-Batches from Binary Files on Demand
#
# This dataset class loads only the index in memory and then uses it to open the
# corresponding .npy files (with memory mapping) and slice out a micro-batch.
# It replicates your original sliding window (plus reflection) logic.
# =============================================================================

class AudioFacialDataset(Dataset):
    def __init__(self, config, preload_index=True, force_reprocess=False):
        """
        Parameters:
          config         - dictionary with keys: 'root_dir', 'sr', 'micro_batch_size', etc.
          preload_index  - if True, load the index from disk (if available) instead of recomputing.
          force_reprocess- if True, ignore existing .npy files/index and reprocess.
        """
        self.config = config
        self.root_dir = config['root_dir']
        self.micro_batch_size = config['micro_batch_size']
        
        # Location for the binary files and index.
        self.bin_dir = os.path.join(self.root_dir, "processed_bin")
        self.index_file = os.path.join(self.bin_dir, "dataset_index.pkl")
        
        # Load or build the index.
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
        # Get the info for the requested segment.
        audio_file, facial_file, start, reflection_flag = self.dataset_index[idx]
        micro_batch_size = self.micro_batch_size
        
        # Load the audio and facial arrays via memory mapping.
        audio_data = np.load(audio_file, mmap_mode='r')
        facial_data = np.load(facial_file, mmap_mode='r')
        T = audio_data.shape[0]  # total number of frames
        
        if not reflection_flag:
            # Regular segment: simply slice out the window.
            audio_segment = audio_data[start : start + micro_batch_size]
            facial_segment = facial_data[start : start + micro_batch_size]
        else:
            # Reflection segment: for the last window, if there are not enough frames,
            # we fill the remainder by reflecting (mirroring) the available frames.
            segment_audio = audio_data[start : T]
            segment_facial = facial_data[start : T]
            # Initialize arrays with zeros.
            audio_segment = np.zeros((micro_batch_size, audio_data.shape[1]), dtype=audio_data.dtype)
            facial_segment = np.zeros((micro_batch_size, facial_data.shape[1]), dtype=facial_data.dtype)
            len_seg = segment_audio.shape[0]
            # Copy available frames.
            audio_segment[:len_seg] = segment_audio
            facial_segment[:len_seg] = segment_facial
            missing = micro_batch_size - len_seg
            # Reflect available frames to fill in.
            ref_audio = np.flip(segment_audio, axis=0)
            ref_facial = np.flip(segment_facial, axis=0)
            audio_segment[len_seg:] = ref_audio[:missing]
            facial_segment[len_seg:] = ref_facial[:missing]
        
        # Convert to torch tensors.
        return (torch.tensor(audio_segment, dtype=torch.float32),
                torch.tensor(facial_segment, dtype=torch.float32))
    
    @staticmethod
    def collate_fn(batch):
        # Collate function that pads sequences in the batch.
        src_batch, trg_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=0)
        return src_batch, trg_batch

# =============================================================================
# DATALOADER PREPARATION FUNCTIONS
#
# These functions create a DataLoader (or train/validation split) using our lazy
# dataset. Multiple workers (and prefetching) ensure the next batch is read from
# disk while the current one is processed.
# =============================================================================

def prepare_dataloader_with_split(config, val_split=0.01):
    dataset = AudioFacialDataset(config)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=AudioFacialDataset.collate_fn,
        num_workers=4,         # adjust as needed
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=AudioFacialDataset.collate_fn,
        num_workers=4,
        prefetch_factor=2
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader

def prepare_dataloader(config):
    dataset = AudioFacialDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=AudioFacialDataset.collate_fn,
        num_workers=4,
        prefetch_factor=2
    )
    return dataset, dataloader

'''
