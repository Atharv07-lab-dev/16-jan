#!/bin/bash

# NavEyes Setup Script
# Complete environment setup and data preparation

echo "========================================"
echo "NavEyes Setup Script"
echo "========================================"

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv naveyes_env
source naveyes_env/bin/activate

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Download NLTK data
echo ""
echo "Step 3: Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Step 4: Download YOLOv8 model
echo ""
echo "Step 4: Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Step 5: Setup Kaggle (if not already configured)
echo ""
echo "Step 5: Kaggle setup..."
echo "Please ensure you have kaggle.json in ~/.kaggle/"
echo "Download from: https://www.kaggle.com/settings"
read -p "Press Enter when ready to continue..."

# Create necessary directories
echo ""
echo "Step 6: Creating project directories..."
mkdir -p checkpoints
mkdir -p frames/train
mkdir -p frames/val
mkdir -p msrvtt

# Step 7: Download MSR-VTT dataset
echo ""
echo "Step 7: Downloading MSR-VTT dataset (18GB)..."
echo "This may take 30-60 minutes depending on your connection..."
read -p "Proceed with download? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    kaggle datasets download -d khoahunhtngng/msrvtt
    echo "Extracting dataset..."
    unzip -q msrvtt.zip -d msrvtt
    rm msrvtt.zip
    echo "Dataset downloaded and extracted!"
else
    echo "Skipping dataset download. You can run it manually later:"
    echo "kaggle datasets download -d khoahunhtngng/msrvtt"
fi

# Step 8: Preprocess data
echo ""
echo "Step 8: Preprocessing data..."
read -p "Run preprocessing now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python preprocess.py
    echo "Preprocessing complete!"
else
    echo "Skipping preprocessing. Run manually with: python preprocess.py"
fi

# Step 9: Setup complete
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Project structure:"
echo "  ├── preprocess.py          # Data preprocessing"
echo "  ├── model.py                # Model architecture"
echo "  ├── dataset.py              # DataLoader"
echo "  ├── train.py                # Training script"
echo "  ├── spatial_detection.py    # YOLOv8 integration"
echo "  ├── streamlit_app.py        # Web dashboard"
echo "  ├── requirements.txt        # Dependencies"
echo "  ├── checkpoints/            # Model checkpoints"
echo "  ├── frames/                 # Extracted frames"
echo "  └── msrvtt/                 # Dataset"
echo ""
echo "Next steps:"
echo "  1. Train model: python train.py (6-8 hours on GPU)"
echo "  2. Launch dashboard: streamlit run streamlit_app.py"
echo ""
echo "To activate environment in future sessions:"
echo "  source naveyes_env/bin/activate"
echo ""
echo "========================================"
