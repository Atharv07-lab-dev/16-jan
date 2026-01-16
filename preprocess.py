"""
MSR-VTT Data Preprocessing Pipeline
Extracts frames, filters indoor captions, builds vocabulary
Run after downloading dataset: kaggle datasets download -d khoahunhtngng/msrvtt
"""

import pandas as pd
import cv2
import os
import numpy as np
import nltk
from collections import Counter
from tqdm import tqdm
import torch

# Download required NLTK data
nltk.download('punkt')

# Configuration
DATA_DIR = 'msrvtt'
FRAME_DIR = 'frames'
NUM_FRAMES = 8
INDOOR_KEYWORDS = ['chair', 'table', 'door', 'room', 'kitchen', 'window', 
                   'sofa', 'bed', 'floor', 'wall', 'cabinet', 'shelf']

def parse_captions(file_path):
    """Parse MSR-VTT caption file into DataFrame"""
    print(f"Parsing captions from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            video_id = parts[0]
            for caption in parts[1:]:
                if caption.strip():
                    data.append({'video_id': video_id, 'caption': caption.strip()})
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} caption pairs")
    return df

def filter_indoor_captions(df, keywords):
    """Filter captions containing indoor navigation keywords"""
    print("Filtering for indoor-relevant captions...")
    pattern = '|'.join(keywords)
    filtered_df = df[df['caption'].str.contains(pattern, case=False, na=False)]
    print(f"Retained {len(filtered_df)} indoor-focused captions ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df

def extract_frames(video_path, num_frames=8):
    """Extract evenly-spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    for i in range(num_frames):
        frame_idx = int(i * total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Resize to 224x224 and convert to RGB
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    return np.array(frames[:num_frames])

def process_videos(df, video_dir, output_dir, num_frames=8):
    """Extract and save frames for all videos in DataFrame"""
    os.makedirs(output_dir, exist_ok=True)
    
    unique_videos = df['video_id'].unique()
    print(f"Processing {len(unique_videos)} unique videos...")
    
    successful = 0
    failed = []
    
    for video_id in tqdm(unique_videos):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        output_path = os.path.join(output_dir, f"{video_id}.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            successful += 1
            continue
        
        if not os.path.exists(video_path):
            failed.append(video_id)
            continue
        
        frames = extract_frames(video_path, num_frames)
        if frames is not None:
            np.save(output_path, frames)
            successful += 1
        else:
            failed.append(video_id)
    
    print(f"Successfully processed: {successful}/{len(unique_videos)}")
    if failed:
        print(f"Failed videos: {len(failed)}")
        with open('failed_videos.txt', 'w') as f:
            f.write('\n'.join(failed))
    
    return successful, failed

def build_vocab(df, min_freq=5):
    """Build vocabulary from captions"""
    print("Building vocabulary...")
    counter = Counter()
    
    for caption in tqdm(df['caption']):
        tokens = nltk.word_tokenize(caption.lower())
        counter.update(tokens)
    
    # Special tokens
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # Add frequent words
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)} tokens")
    return vocab

def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("MSR-VTT PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Parse captions
    train_df = parse_captions(os.path.join(DATA_DIR, 'train_caps.txt'))
    val_df = parse_captions(os.path.join(DATA_DIR, 'val_caps.txt'))
    
    # Step 2: Filter for indoor navigation
    train_df = filter_indoor_captions(train_df, INDOOR_KEYWORDS)
    val_df = filter_indoor_captions(val_df, INDOOR_KEYWORDS)
    
    # Step 3: Build vocabulary
    vocab = build_vocab(train_df, min_freq=5)
    torch.save(vocab, 'vocab.pth')
    print("Saved vocabulary to vocab.pth")
    
    # Step 4: Process videos and extract frames
    print("\nProcessing training videos...")
    process_videos(train_df, os.path.join(DATA_DIR, 'video_train'), 
                  os.path.join(FRAME_DIR, 'train'), NUM_FRAMES)
    
    print("\nProcessing validation videos...")
    process_videos(val_df, os.path.join(DATA_DIR, 'video_val'), 
                  os.path.join(FRAME_DIR, 'val'), NUM_FRAMES)
    
    # Step 5: Save processed DataFrames
    # Remove videos that failed processing
    if os.path.exists('failed_videos.txt'):
        with open('failed_videos.txt', 'r') as f:
            failed = set(f.read().splitlines())
        train_df = train_df[~train_df['video_id'].isin(failed)]
        val_df = val_df[~val_df['video_id'].isin(failed)]
    
    train_df.to_csv('train_processed.csv', index=False)
    val_df.to_csv('val_processed.csv', index=False)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Vocabulary size: {len(vocab)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
