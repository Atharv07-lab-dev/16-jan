"""
NavEyes - Streamlit Production Dashboard
Real-time spatial navigation assistant for visually impaired users
100% offline | Real-time inference | TTS integration
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import nltk
import time
from io import BytesIO
import base64

# Import custom modules
from model import VideoCaptioningModel
from spatial_detection import SpatialDetector

# Page configuration
st.set_page_config(
    page_title="NavEyes - Spatial Navigation",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        font-size: 1.2rem;
    }
    .nav-cue {
        font-size: 2rem;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .direction-left {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .direction-right {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .direction-center {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .direction-ahead {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_caption_model():
    """Load trained video captioning model"""
    try:
        vocab = torch.load('vocab.pth', map_location='cpu')
        model = VideoCaptioningModel(
            vocab_size=len(vocab),
            embed_dim=512,
            hidden_dim=1024,
            dropout=0.3,
            num_frames=8
        )
        
        # Load best checkpoint
        checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create inverse vocabulary for decoding
        inv_vocab = {v: k for k, v in vocab.items()}
        
        return model, vocab, inv_vocab
    except Exception as e:
        st.error(f"Error loading caption model: {e}")
        return None, None, None

@st.cache_resource
def load_spatial_model():
    """Load YOLOv8 spatial detection model"""
    try:
        detector = SpatialDetector(model_size='n', conf_threshold=0.3)
        return detector
    except Exception as e:
        st.error(f"Error loading spatial model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Resize
    img_resized = cv2.resize(np.array(image), target_size)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor [1, 1, 3, 224, 224] (batch, frames, channels, H, W)
    img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def decode_caption(token_ids, inv_vocab):
    """Decode token IDs to text"""
    tokens = []
    for token_id in token_ids:
        if token_id == 2:  # <eos>
            break
        if token_id in [0, 1]:  # <pad>, <sos>
            continue
        token = inv_vocab.get(token_id, '<unk>')
        if token != '<unk>':
            tokens.append(token)
    
    caption = ' '.join(tokens)
    # Basic cleanup
    caption = caption.capitalize()
    if caption and caption[-1] not in '.!?':
        caption += '.'
    
    return caption

def generate_caption(model, image_tensor, inv_vocab, max_len=20):
    """Generate caption from image"""
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            max_len=max_len,
            start_token=1,  # <sos>
            end_token=2     # <eos>
        )
        caption = decode_caption(caption_ids.tolist(), inv_vocab)
    
    return caption

def text_to_speech_placeholder(text):
    """Placeholder for TTS - returns audio signal info"""
    # In production, integrate with pyttsx3 or gTTS
    return f"🔊 Audio: \"{text}\""

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Minimum confidence for object detection"
    )
    
    enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
    show_detections = st.checkbox("Show Detection Boxes", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    st.metric("Frames Processed", st.session_state.frame_count)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **NavEyes** assists visually impaired individuals with:
    - Real-time scene understanding
    - Spatial object detection
    - Directional navigation cues
    - Voice guidance
    
    **100% Offline** | **No Internet Required**
    """)

# Main content
st.markdown('<p class="main-header">🦯 NavEyes</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Offline Spatial Navigation Assistant for the Visually Impaired</p>', unsafe_allow_html=True)

# Load models
with st.spinner("Loading AI models..."):
    caption_model, vocab, inv_vocab = load_caption_model()
    spatial_detector = load_spatial_model()

if caption_model is None or spatial_detector is None:
    st.error("Failed to load models. Please ensure model files are in the correct location.")
    st.stop()

st.success("✓ Models loaded successfully")

# Update detector confidence
spatial_detector.conf_threshold = confidence_threshold

# Camera input
st.markdown("### 📸 Camera Input")
camera_image = st.camera_input("Point camera at your surroundings")

if camera_image is not None:
    # Update counter
    st.session_state.frame_count += 1
    
    # Load image
    image = Image.open(camera_image).convert('RGB')
    img_array = np.array(image)
    
    with st.spinner("🔍 Analyzing scene..."):
        start_time = time.time()
        
        # Generate caption
        image_tensor = preprocess_image(image)
        caption = generate_caption(caption_model, image_tensor, inv_vocab)
        
        # Spatial detection
        detections = spatial_detector.detect_objects(img_array)
        nav_cue_data = spatial_detector.get_navigation_cue(img_array, caption)
        
        processing_time = time.time() - start_time
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📷 Scene View")
        
        if show_detections and len(detections) > 0:
            annotated_img = spatial_detector.visualize_detections(img_array, detections)
            st.image(annotated_img, use_column_width=True)
        else:
            st.image(image, use_column_width=True)
        
        # Processing info
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**Processing Time:** {processing_time:.2f}s")
        st.markdown(f"**Objects Detected:** {nav_cue_data['object_count']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Navigation Guidance")
        
        # Main navigation cue
        direction = nav_cue_data['direction']
        cue_class = f"direction-{direction}"
        st.markdown(
            f'<div class="nav-cue {cue_class}">{nav_cue_data["cue"]}</div>',
            unsafe_allow_html=True
        )
        
        # Details
        st.markdown("##### 📋 Scene Details")
        st.markdown(f"**Caption:** {caption}")
        st.markdown(f"**Direction:** {direction.title()}")
        
        if nav_cue_data['primary_object']:
            st.markdown(f"**Primary Object:** {nav_cue_data['primary_object']}")
        
        if nav_cue_data['all_objects']:
            objects_str = ', '.join(nav_cue_data['all_objects'][:5])
            st.markdown(f"**Detected Objects:** {objects_str}")
        
        st.markdown(f"**Confidence:** {nav_cue_data['confidence']:.0%}")
        
        # TTS
        if enable_tts:
            st.markdown("---")
            st.markdown("##### 🔊 Voice Guidance")
            tts_text = nav_cue_data['cue']
            st.info(text_to_speech_placeholder(tts_text))
            
            # Audio playback simulation
            st.markdown("*In production, audio would play automatically*")
    
    # Performance metrics
    st.markdown("---")
    st.markdown("### 📈 Performance Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("FPS", f"{1/processing_time:.1f}")
    
    with metric_col2:
        st.metric("Latency", f"{processing_time*1000:.0f}ms")
    
    with metric_col3:
        st.metric("Detection Conf.", f"{nav_cue_data['confidence']:.0%}")
    
    with metric_col4:
        st.metric("Objects", nav_cue_data['object_count'])

else:
    # Instructions
    st.info("👆 Click the camera button above to start navigation assistance")
    
    # Demo information
    st.markdown("---")
    st.markdown("### 🎯 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Capture")
        st.markdown("Take a photo of your surroundings using the camera")
    
    with col2:
        st.markdown("#### 2️⃣ Analyze")
        st.markdown("AI models detect objects and generate scene descriptions")
    
    with col3:
        st.markdown("#### 3️⃣ Navigate")
        st.markdown("Receive spatial cues (left/right/center/ahead) with voice guidance")
    
    st.markdown("---")
    st.markdown("### 🌟 Key Features")
    
    features = [
        "✓ **100% Offline** - No internet connection required",
        "✓ **Real-time Processing** - Sub-second response times",
        "✓ **Spatial Awareness** - Direction-based navigation cues",
        "✓ **Voice Guidance** - Text-to-speech for accessibility",
        "✓ **Indoor Navigation** - Optimized for indoor environments",
        "✓ **Privacy Focused** - All processing happens on device"
    ]
    
    for feature in features:
        st.markdown(feature)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">NavEyes v1.0 | Empowering 285M visually impaired individuals worldwide 🌍</p>',
    unsafe_allow_html=True
)
