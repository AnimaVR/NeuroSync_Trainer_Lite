import numpy as np

from utils.audio.load_audio import load_and_preprocess_audio, load_audio_from_bytes
from utils.audio.extraction.extract_features_utils import  smooth_features, extract_mfcc_features, extract_autocorrelation_features

def extract_audio_features(audio_input, sr=88200, from_bytes=False):
    if from_bytes:
        y, sr = load_audio_from_bytes(audio_input, sr)
    else:
        y, sr = load_and_preprocess_audio(audio_input, sr)
    
    # frame_length = int(0.01667 * sr)  # Frame length set to 0.01667 seconds (~60 fps)
    frame_length = int(0.03333 * sr)
    hop_length = frame_length // 2  # 2x overlap for smoother transitions
    min_frames = 9  # Minimum number of frames needed for delta calculation

    num_frames = (len(y) - frame_length) // hop_length + 1

    if num_frames < min_frames:
        print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
        return None, None

    combined_features = extract_and_combine_features(y, sr, frame_length, hop_length)
    
    return combined_features, y

def extract_and_combine_features(y, sr, frame_length, hop_length, apply_smoothing=False, include_autocorr=True):
   
    all_features = []
    
    # 1) MFCC as baseline
    mfcc_features, _ = extract_mfcc_features(y, sr, frame_length, hop_length)
    all_features.append(mfcc_features)
    
    # 2) Autocorrelation
    if include_autocorr:
        autocorr_features = extract_autocorrelation_features(
            y, sr, frame_length, hop_length
        )
        all_features.append(autocorr_features)
    
    combined_features = np.hstack(all_features)

    if apply_smoothing:
        combined_features = smooth_features(combined_features)

    return combined_features
