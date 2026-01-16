"""
Standalone Inference Script
Test trained model on single images without Streamlit
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os

from model import VideoCaptioningModel
from spatial_detection import SpatialDetector

class NavEyesInference:
    def __init__(self, model_path='checkpoints/best_model.pth', vocab_path='vocab.pth'):
        """Initialize inference engine"""
        print("Loading models...")
        
        # Load vocabulary
        self.vocab = torch.load(vocab_path, map_location='cpu')
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Load caption model
        self.caption_model = VideoCaptioningModel(
            vocab_size=len(self.vocab),
            embed_dim=512,
            hidden_dim=1024,
            dropout=0.3,
            num_frames=8
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        self.caption_model.load_state_dict(checkpoint['model_state_dict'])
        self.caption_model.eval()
        
        # Load spatial detector
        self.spatial_detector = SpatialDetector(model_size='n')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.caption_model.to(self.device)
        
        print(f"Models loaded on {self.device}")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image_path
            image = Image.fromarray(img_array)
        
        # Resize and normalize
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 224, 224]
        
        return img_tensor.to(self.device), img_array
    
    def decode_caption(self, token_ids):
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id == 2:  # <eos>
                break
            if token_id in [0, 1]:  # <pad>, <sos>
                continue
            token = self.inv_vocab.get(token_id, '<unk>')
            if token != '<unk>':
                tokens.append(token)
        
        caption = ' '.join(tokens)
        caption = caption.capitalize()
        if caption and caption[-1] not in '.!?':
            caption += '.'
        
        return caption
    
    def generate_caption(self, image_tensor):
        """Generate caption from image"""
        with torch.no_grad():
            caption_ids = self.caption_model.generate(
                image_tensor,
                max_len=20,
                start_token=1,
                end_token=2
            )
            caption = self.decode_caption(caption_ids.tolist())
        
        return caption
    
    def process_image(self, image_path, visualize=False):
        """Complete processing pipeline"""
        # Preprocess
        image_tensor, img_array = self.preprocess_image(image_path)
        
        # Generate caption
        caption = self.generate_caption(image_tensor)
        
        # Spatial analysis
        detections = self.spatial_detector.detect_objects(img_array)
        nav_cue = self.spatial_detector.get_navigation_cue(img_array, caption)
        
        # Visualize if requested
        if visualize and len(detections) > 0:
            annotated_img = self.spatial_detector.visualize_detections(img_array, detections)
            cv2.imshow('NavEyes Detection', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'caption': caption,
            'navigation_cue': nav_cue,
            'detections': detections
        }
    
    def process_video_stream(self, source=0):
        """Process live video stream (webcam)"""
        print(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Press 'q' to quit, 'c' to capture frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display current frame
            cv2.imshow('NavEyes Live', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nProcessing frame...")
                result = self.process_image(frame_rgb)
                
                print("\n" + "="*60)
                print("NAVIGATION CUE")
                print("="*60)
                print(f"{result['navigation_cue']['cue']}")
                print("="*60)
                print(f"Caption: {result['caption']}")
                print(f"Direction: {result['navigation_cue']['direction']}")
                print(f"Primary Object: {result['navigation_cue']['primary_object']}")
                print(f"Objects Detected: {result['navigation_cue']['object_count']}")
                print(f"Confidence: {result['navigation_cue']['confidence']:.2%}")
                print("="*60 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='NavEyes Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', action='store_true', help='Use webcam for live inference')
    parser.add_argument('--source', type=int, default=0, help='Video source (default: 0 for webcam)')
    parser.add_argument('--visualize', action='store_true', help='Show detection boxes')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='vocab.pth', 
                       help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = NavEyesInference(model_path=args.model, vocab_path=args.vocab)
    
    if args.video:
        # Live video processing
        engine.process_video_stream(source=args.source)
    
    elif args.image:
        # Single image processing
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        print(f"Processing image: {args.image}")
        result = engine.process_image(args.image, visualize=args.visualize)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\n🎯 Navigation Cue:")
        print(f"   {result['navigation_cue']['cue']}")
        print(f"\n📝 Caption:")
        print(f"   {result['caption']}")
        print(f"\n🧭 Direction: {result['navigation_cue']['direction'].title()}")
        print(f"🎯 Primary Object: {result['navigation_cue']['primary_object']}")
        print(f"📊 Objects Detected: {result['navigation_cue']['object_count']}")
        print(f"✓ Confidence: {result['navigation_cue']['confidence']:.0%}")
        
        if result['navigation_cue']['all_objects']:
            print(f"\n📦 All Objects: {', '.join(result['navigation_cue']['all_objects'])}")
        
        print("="*60 + "\n")
    
    else:
        print("Please specify --image <path> or --video")
        print("Examples:")
        print("  python inference.py --image test.jpg --visualize")
        print("  python inference.py --video")

if __name__ == "__main__":
    main()
