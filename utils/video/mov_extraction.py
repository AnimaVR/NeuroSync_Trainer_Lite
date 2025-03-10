# mov_extraction.pyvisio

import subprocess
import os

from config import training_config as config

def find_files(folder_path):
    mov_path, mp4_path, wav_path = None, None, None
    facial_csv_path, facial_json_path, other_csv_path = None, None, None

    # Always set audio_features_csv_path to the expected path, even if it doesn't exist
    audio_features_csv_path = os.path.join(folder_path, 'audio_features.csv')

    for file in os.listdir(folder_path):
        # if file.endswith('.mov'):
        #     mov_path = os.path.join(folder_path, file)
        # elif file.endswith('.mp4'):
        #     mp4_path = os.path.join(folder_path, file)
        if file.endswith('.wav'):
            wav_path = os.path.join(folder_path, file)
        elif file.endswith('.csv'):
            if 'iPhone_cal' in file:
                facial_csv_path = os.path.join(folder_path, file)
            else:
                other_csv_path = os.path.join(folder_path, file)
        elif file.endswith('.json'):
            facial_json_path = os.path.join(folder_path, file)
    # Return audio_features_csv_path regardless of its existence
    return mov_path, mp4_path, wav_path, facial_csv_path, facial_json_path, audio_features_csv_path, other_csv_path

def get_audio(video_path, wav_path, folder_path):
    if video_path:
        audio_path = extract_audio(video_path, folder_path)
       
        return audio_path
    else:
        return wav_path

def extract_audio(video_path, output_dir):
    sr = config['sr']
    ffmpeg_path = config['ffmpeg_path']
    audio_path = os.path.join(output_dir, 'audio.wav')
    
    # Check if the audio file already exists
    if os.path.exists(audio_path):
        print(f"Audio already exists at {audio_path}")
        return audio_path
    
    try:
        command = [
            ffmpeg_path,
            '-i', video_path,
            '-ac', '1',
            '-ar', str(sr),
            '-y',
            audio_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from {video_path}: {e.stderr.decode('utf-8')}")
        return None

