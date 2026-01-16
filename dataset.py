"""
PyTorch Dataset and DataLoader for MSR-VTT Video Captioning
Handles frame loading, tokenization, and batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nltk
from torch.nn.utils.rnn import pad_sequence
import os

class VideoCaptionDataset(Dataset):
    def __init__(self, csv_path, vocab, frame_dir, max_frames=8, max_caption_len=20):
        """
        Args:
            csv_path: path to processed CSV (train_processed.csv / val_processed.csv)
            vocab: vocabulary dictionary {token: id}
            frame_dir: directory containing .npy frame files
            max_frames: number of frames per video
            max_caption_len: maximum caption length (for truncation)
        """
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.frame_dir = frame_dir
        self.max_frames = max_frames
        self.max_caption_len = max_caption_len
        
        # Create inverse vocab for debugging
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Filter out videos without frames
        valid_indices = []
        for idx in range(len(self.df)):
            video_id = self.df.iloc[idx]['video_id']
            frame_path = os.path.join(frame_dir, f"{video_id}.npy")
            if os.path.exists(frame_path):
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset loaded: {len(self.df)} samples with valid frames")
    
    def __len__(self):
        return len(self.df)
    
    def tokenize(self, caption):
        """Convert caption to token IDs"""
        # Tokenize and lowercase
        tokens = nltk.word_tokenize(caption.lower())
        
        # Add special tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Truncate if too long
        if len(tokens) > self.max_caption_len:
            tokens = tokens[:self.max_caption_len - 1] + ['<eos>']
        
        # Convert to IDs
        token_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        
        return torch.LongTensor(token_ids)
    
    def __getitem__(self, idx):
        """Get a single training sample"""
        row = self.df.iloc[idx]
        video_id = row['video_id']
        caption = row['caption']
        
        # Load pre-extracted frames
        frame_path = os.path.join(self.frame_dir, f"{video_id}.npy")
        frames = np.load(frame_path)  # [num_frames, 224, 224, 3]
        
        # Ensure correct number of frames
        if frames.shape[0] < self.max_frames:
            # Pad with zeros if needed
            padding = np.zeros((self.max_frames - frames.shape[0], 224, 224, 3), dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)
        frames = frames[:self.max_frames]
        
        # Convert to tensor and normalize to [0, 1]
        frames = torch.FloatTensor(frames) / 255.0
        # Permute to [num_frames, 3, 224, 224] (PyTorch format)
        frames = frames.permute(0, 3, 1, 2)
        
        # Tokenize caption
        caption_tensor = self.tokenize(caption)
        
        return {
            'video_id': video_id,
            'frames': frames,
            'caption': caption_tensor,
            'caption_text': caption
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable-length captions
    Pads captions to same length within batch
    """
    video_ids = [item['video_id'] for item in batch]
    frames = torch.stack([item['frames'] for item in batch])  # [batch, num_frames, 3, 224, 224]
    captions = [item['caption'] for item in batch]
    caption_texts = [item['caption_text'] for item in batch]
    
    # Pad captions to max length in batch
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return {
        'video_ids': video_ids,
        'frames': frames,
        'captions': captions_padded,
        'caption_texts': caption_texts
    }

def create_dataloaders(vocab, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders
    Args:
        vocab: vocabulary dictionary
        batch_size: batch size for training
        num_workers: number of data loading workers
    Returns:
        train_loader, val_loader
    """
    train_dataset = VideoCaptionDataset(
        csv_path='train_processed.csv',
        vocab=vocab,
        frame_dir='frames/train',
        max_frames=8,
        max_caption_len=20
    )
    
    val_dataset = VideoCaptionDataset(
        csv_path='val_processed.csv',
        vocab=vocab,
        frame_dir='frames/val',
        max_frames=8,
        max_caption_len=20
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataset
    import torch
    
    vocab = torch.load('vocab.pth')
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    dataset = VideoCaptionDataset(
        csv_path='train_processed.csv',
        vocab=vocab,
        frame_dir='frames/train'
    )
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"Video ID: {sample['video_id']}")
    print(f"Frames shape: {sample['frames'].shape}")
    print(f"Caption: {sample['caption_text']}")
    print(f"Caption tokens: {sample['caption'][:10]}...")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(vocab, batch_size=4, num_workers=0)
    
    print(f"\nDataLoader test:")
    batch = next(iter(train_loader))
    print(f"Batch frames shape: {batch['frames'].shape}")
    print(f"Batch captions shape: {batch['captions'].shape}")
    print(f"Sample captions: {batch['caption_texts'][:2]}")
