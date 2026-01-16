"""
Training Script for Video Captioning Model
Includes training loop, validation, checkpointing, and logging
Expected runtime: 6-8 hours on single GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
from tqdm import tqdm
import numpy as np

from model import VideoCaptioningModel
from dataset import create_dataloaders

# Configuration
class Config:
    # Model parameters
    embed_dim = 512
    hidden_dim = 1024
    dropout = 0.3
    num_frames = 8
    
    # Training parameters
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-5
    grad_clip = 1.0
    teacher_forcing_ratio = 1.0  # Start with full teacher forcing
    teacher_forcing_decay = 0.95  # Decay per epoch
    
    # Checkpointing
    save_dir = 'checkpoints'
    save_every = 5  # Save every N epochs
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4
    
    # Mixed precision training
    use_amp = True  # Automatic Mixed Precision for faster training

def calculate_accuracy(predictions, targets, pad_idx=0):
    """Calculate token-level accuracy, ignoring padding"""
    pred_tokens = predictions.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def train_epoch(model, dataloader, optimizer, criterion, scaler, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    # Adjust teacher forcing ratio
    tf_ratio = config.teacher_forcing_ratio * (config.teacher_forcing_decay ** epoch)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(config.device)  # [B, 8, 3, 224, 224]
        captions = batch['captions'].to(config.device)  # [B, max_len]
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.use_amp:
            with autocast():
                outputs = model(frames, captions, teacher_forcing_ratio=tf_ratio)
                # outputs: [B, max_len, vocab_size]
                # Reshape for loss calculation
                outputs_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
                targets_flat = captions[:, 1:].reshape(-1)
                loss = criterion(outputs_flat, targets_flat)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames, captions, teacher_forcing_ratio=tf_ratio)
            outputs_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets_flat = captions[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        # Calculate metrics
        accuracy = calculate_accuracy(outputs[:, 1:], captions[:, 1:])
        total_loss += loss.item()
        total_accuracy += accuracy
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'tf': f'{tf_ratio:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy

def validate(model, dataloader, criterion, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            frames = batch['frames'].to(config.device)
            captions = batch['captions'].to(config.device)
            
            outputs = model(frames, captions, teacher_forcing_ratio=1.0)
            outputs_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets_flat = captions[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            accuracy = calculate_accuracy(outputs[:, 1:], captions[:, 1:])
            total_loss += loss.item()
            total_accuracy += accuracy
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': vars(config)
    }
    filepath = os.path.join(config.save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def main():
    config = Config()
    
    print("=" * 60)
    print("VIDEO CAPTIONING TRAINING")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Mixed precision: {config.use_amp}")
    print("=" * 60)
    
    # Create checkpoint directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Load vocabulary
    vocab = torch.load('vocab.pth')
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        vocab, 
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = VideoCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_frames=config.num_frames
    ).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, config,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, config,
                'best_model.pth'
            )
            print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.num_epochs-1, train_losses[-1], val_losses[-1],
        config, 'model_final.pth'
    )
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
