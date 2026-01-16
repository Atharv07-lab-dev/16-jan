# NavEyes - Offline Spatial Navigation Assistant

**Empowering 285 million visually impaired individuals worldwide with AI-powered navigation**

## 🎯 Overview

NavEyes is a production-ready video captioning system that provides real-time spatial navigation cues for visually impaired users. The system combines:

- **S2VT (Sequence to Sequence Video to Text)** - Deep learning model for scene understanding
- **YOLOv8** - Object detection for spatial awareness
- **MSR-VTT Dataset** - 10,000 annotated videos for training
- **Streamlit Dashboard** - User-friendly web interface

### Key Features

✅ **100% Offline** - No internet connection required  
✅ **Real-time Processing** - Sub-second inference times  
✅ **Spatial Awareness** - Direction-based cues (left/right/center/ahead)  
✅ **Voice Guidance** - Text-to-speech integration ready  
✅ **Indoor Optimized** - Trained on indoor navigation scenarios  
✅ **Privacy Focused** - All processing on-device  

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Streamlit Dashboard](#streamlit-dashboard)
6. [Project Structure](#project-structure)
7. [Model Architecture](#model-architecture)
8. [Performance Metrics](#performance-metrics)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 20GB+ free disk space
- Kaggle API credentials

### Quick Setup

```bash
# Clone or download all project files
# Then run the setup script:
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv naveyes_env
source naveyes_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# 4. Download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Create directories
mkdir -p checkpoints frames/train frames/val msrvtt
```

---

## 📊 Dataset Preparation

### 1. Download MSR-VTT Dataset

```bash
# Setup Kaggle API (one-time)
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (18GB)
kaggle datasets download -d khoahunhtngng/msrvtt
unzip msrvtt.zip -d msrvtt
```

### 2. Preprocess Data

```bash
python preprocess.py
```

**What this does:**
- Parses 130K+ caption pairs from `train_caps.txt`
- Filters for indoor navigation keywords (~25K samples)
- Extracts 8 frames per video
- Builds vocabulary (10K tokens)
- Saves processed data to `train_processed.csv`, `val_processed.csv`

**Expected output:**
```
Training samples: ~25,000
Validation samples: ~2,000
Vocabulary size: ~10,000
Processing time: 2-3 hours
```

---

## 🎓 Training

### Start Training

```bash
python train.py
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Adjust based on GPU memory |
| Epochs | 20 | Convergence at ~15 epochs |
| Learning Rate | 1e-4 | Adam optimizer |
| GPU | Required | CUDA-enabled GPU |
| Time | 6-8 hours | Single GPU training |

### Monitor Progress

Training will output:
```
Epoch 1/20
Train Loss: 2.345 | Train Acc: 0.234
Val Loss:   2.123 | Val Acc:   0.267
✓ New best model saved
```

### Checkpoints

Models are saved to `checkpoints/`:
- `best_model.pth` - Best validation loss
- `checkpoint_epoch_N.pth` - Every 5 epochs
- `model_final.pth` - Final model

---

## 🔍 Inference

### Single Image Testing

```bash
# Basic inference
python inference.py --image test.jpg

# With visualization
python inference.py --image test.jpg --visualize
```

### Live Webcam Mode

```bash
python inference.py --video
```

**Controls:**
- Press `c` to capture and analyze frame
- Press `q` to quit

### Example Output

```
🎯 Navigation Cue:
   Left: Person sitting on chair - chair detected

📝 Caption: Person sitting on chair.
🧭 Direction: Left
🎯 Primary Object: chair
📊 Objects Detected: 3
✓ Confidence: 85%
```

---

## 🖥️ Streamlit Dashboard

### Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

### Features

1. **Camera Input** - Real-time frame capture
2. **Scene Analysis** - AI-powered captioning
3. **Spatial Detection** - Object localization
4. **Navigation Cues** - Direction-based guidance
5. **Performance Metrics** - FPS, latency, confidence
6. **Visualization** - Bounding box overlay

### Usage

1. Click camera button
2. Point at surroundings
3. Receive instant navigation cue
4. Listen to voice guidance (TTS)

---

## 📁 Project Structure

```
naveyes/
├── preprocess.py          # Data preprocessing pipeline
├── model.py               # S2VT model architecture
├── dataset.py             # PyTorch dataset/dataloader
├── train.py               # Training script
├── spatial_detection.py   # YOLOv8 integration
├── streamlit_app.py       # Web dashboard
├── inference.py           # Standalone testing
├── requirements.txt       # Dependencies
├── setup.sh              # Setup automation
├── README.md             # This file
│
├── checkpoints/          # Model checkpoints
│   ├── best_model.pth
│   └── model_final.pth
│
├── frames/               # Extracted video frames
│   ├── train/
│   └── val/
│
├── msrvtt/               # Dataset
│   ├── video_train/
│   ├── video_val/
│   ├── train_caps.txt
│   └── val_caps.txt
│
└── vocab.pth             # Token vocabulary
```

---

## 🏗️ Model Architecture

### S2VT Video Captioning Model

```
Input: [Batch, 8 frames, 3, 224, 224]
    ↓
ResNet152 (pretrained) × 8 frames
    ↓ [Batch, 8, 2048]
Linear Projection
    ↓ [Batch, 8, 512]
Bidirectional LSTM
    ↓ [Batch, 8, 2048]
Multi-Head Attention (8 heads)
    ↓ [Batch, 8, 2048]
Text Decoder LSTM
    ↓ [Batch, MaxLen, 1024]
Linear Output Layer
    ↓ [Batch, MaxLen, VocabSize]
Output: Token probabilities
```

### Parameters

- **Total Parameters:** ~95M
- **Trainable Parameters:** ~85M
- **ResNet152:** Partially frozen (fine-tune last 10 layers)

---

## 📈 Performance Metrics

### Expected Results

| Metric | Value |
|--------|-------|
| Training Loss | 1.8 - 2.0 |
| Validation Loss | 1.9 - 2.1 |
| Token Accuracy | 60-65% |
| Inference Time | 0.5-1.0s (GPU) |
| FPS | 1-2 (real-time) |

### Hardware Requirements

**Training:**
- GPU: NVIDIA RTX 3060+ (12GB+ VRAM)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+

**Inference:**
- CPU: Moderate (Intel i5+)
- RAM: 8GB+
- GPU: Optional (speeds up 10x)

---

## 🎯 Usage Examples

### Example 1: Indoor Navigation

```python
from inference import NavEyesInference

engine = NavEyesInference()
result = engine.process_image('living_room.jpg')

print(result['navigation_cue']['cue'])
# Output: "Left: Chair and table visible - chair detected"
```

### Example 2: Batch Processing

```python
import os
from inference import NavEyesInference

engine = NavEyesInference()
image_dir = 'test_images/'

for img_file in os.listdir(image_dir):
    result = engine.process_image(os.path.join(image_dir, img_file))
    print(f"{img_file}: {result['navigation_cue']['cue']}")
```

### Example 3: Custom Video Source

```python
engine = NavEyesInference()
engine.process_video_stream(source='rtsp://camera_ip/stream')
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size in train.py
batch_size = 8  # Instead of 16
```

**2. Dataset not found**
```bash
# Verify dataset structure
ls msrvtt/video_train  # Should show .mp4 files
```

**3. Model not loading**
```bash
# Check checkpoint exists
ls checkpoints/best_model.pth
```

**4. Slow inference**
```bash
# Use GPU if available
python inference.py --image test.jpg
# Check: "Models loaded on cuda"
```

---

## 📚 References

- **MSR-VTT Dataset:** [arXiv:1609.06828](https://arxiv.org/abs/1609.06828)
- **S2VT Model:** [arXiv:1505.00487](https://arxiv.org/abs/1505.00487)
- **YOLOv8:** [Ultralytics Docs](https://docs.ultralytics.com)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Implement full TTS integration (pyttsx3/gTTS)
- [ ] Add support for outdoor navigation
- [ ] Optimize for mobile deployment
- [ ] Multi-language support
- [ ] Real-time video streaming
- [ ] Edge device optimization (Jetson Nano, Raspberry Pi)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Microsoft Research for MSR-VTT dataset
- Anthropic for Claude AI assistance
- Ultralytics for YOLOv8
- PyTorch and Streamlit communities

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting)
2. Review code comments
3. Test with provided examples

---

**Built with ❤️ for accessibility**

*NavEyes v1.0 - Empowering independence through AI*
