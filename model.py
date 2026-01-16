"""
S2VT (Sequence to Sequence Video to Text) Model
Encoder: ResNet152 + Bidirectional LSTM
Decoder: LSTM with Attention
"""

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

class VideoCaptioningModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=1024, 
                 dropout=0.3, num_frames=8):
        super(VideoCaptioningModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        # Video encoder: ResNet152 pretrained on ImageNet
        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        # Freeze early layers
        for param in list(self.cnn.parameters())[:-10]:
            param.requires_grad = False
        
        # Project CNN features to embedding dimension
        self.cnn_projection = nn.Linear(2048, embed_dim)
        
        # Video temporal encoder (bidirectional LSTM)
        self.video_lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if hidden_dim > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Text decoder components
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,  # Concatenate with video context
            hidden_dim, 
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode_video(self, video_frames):
        """
        Encode video frames into temporal features
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]
        Returns:
            video_features: [batch_size, num_frames, hidden_dim*2]
        """
        batch_size, num_frames, c, h, w = video_frames.shape
        
        # Extract CNN features for each frame
        cnn_features = []
        for t in range(num_frames):
            # [batch_size, 3, 224, 224] -> [batch_size, 2048, 1, 1]
            feat = self.cnn(video_frames[:, t])
            # [batch_size, 2048, 1, 1] -> [batch_size, 2048]
            feat = feat.view(batch_size, -1)
            # [batch_size, 2048] -> [batch_size, embed_dim]
            feat = self.cnn_projection(feat)
            cnn_features.append(feat)
        
        # Stack features: [batch_size, num_frames, embed_dim]
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # Apply temporal LSTM
        video_features, _ = self.video_lstm(cnn_features)
        # video_features: [batch_size, num_frames, hidden_dim*2]
        
        return video_features
    
    def forward(self, video_frames, captions, teacher_forcing_ratio=1.0):
        """
        Forward pass for training
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]
            captions: [batch_size, max_caption_len]
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: [batch_size, max_caption_len, vocab_size]
        """
        batch_size = video_frames.size(0)
        max_len = captions.size(1)
        
        # Encode video
        video_features = self.encode_video(video_frames)
        # video_features: [batch_size, num_frames, hidden_dim*2]
        
        # Apply self-attention over video frames
        video_context, _ = self.attention(
            video_features, 
            video_features, 
            video_features
        )
        # video_context: [batch_size, num_frames, hidden_dim*2]
        
        # Initialize decoder hidden state
        decoder_hidden = None
        
        # Prepare outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(video_frames.device)
        
        # First input is <sos> token
        decoder_input = captions[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode step by step
        for t in range(1, max_len):
            # Embed current token
            embedded = self.text_embedding(decoder_input)  # [batch_size, 1, embed_dim]
            
            # Average pool video context for this step
            pooled_video = video_context.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim*2]
            
            # Concatenate embedded token with video context
            decoder_input_combined = torch.cat([embedded, pooled_video], dim=2)
            
            # LSTM step
            output, decoder_hidden = self.text_lstm(decoder_input_combined, decoder_hidden)
            
            # Predict next token
            prediction = self.fc_out(self.dropout(output))  # [batch_size, 1, vocab_size]
            outputs[:, t] = prediction.squeeze(1)
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = captions[:, t].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(dim=2)
        
        return outputs
    
    def generate(self, video_frames, max_len=20, start_token=1, end_token=2):
        """
        Generate caption for inference
        Args:
            video_frames: [1, num_frames, 3, 224, 224]
            max_len: maximum caption length
            start_token: <sos> token id
            end_token: <eos> token id
        Returns:
            caption_ids: [max_len] tensor of token ids
        """
        self.eval()
        with torch.no_grad():
            # Encode video
            video_features = self.encode_video(video_frames)
            video_context, _ = self.attention(video_features, video_features, video_features)
            
            # Initialize
            decoder_input = torch.LongTensor([[start_token]]).to(video_frames.device)
            decoder_hidden = None
            caption_ids = [start_token]
            
            # Generate tokens
            for _ in range(max_len - 1):
                embedded = self.text_embedding(decoder_input)
                pooled_video = video_context.mean(dim=1, keepdim=True)
                decoder_input_combined = torch.cat([embedded, pooled_video], dim=2)
                
                output, decoder_hidden = self.text_lstm(decoder_input_combined, decoder_hidden)
                prediction = self.fc_out(output)
                
                # Get most likely token
                token_id = prediction.argmax(dim=2).item()
                caption_ids.append(token_id)
                
                # Stop at end token
                if token_id == end_token:
                    break
                
                decoder_input = torch.LongTensor([[token_id]]).to(video_frames.device)
            
            return torch.LongTensor(caption_ids)

if __name__ == "__main__":
    # Test model
    model = VideoCaptioningModel(vocab_size=5000)
    video = torch.randn(2, 8, 3, 224, 224)
    captions = torch.randint(0, 5000, (2, 15))
    
    outputs = model(video, captions)
    print(f"Output shape: {outputs.shape}")  # [2, 15, 5000]
    
    # Test generation
    generated = model.generate(video[:1])
    print(f"Generated caption: {generated}")
