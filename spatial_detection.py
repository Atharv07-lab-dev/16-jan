"""
Spatial Awareness Module using YOLOv8
Detects objects and provides directional cues (left/center/right/ahead)
"""

import numpy as np
import cv2
from ultralytics import YOLO
import torch

class SpatialDetector:
    def __init__(self, model_size='n', conf_threshold=0.3):
        """
        Initialize YOLOv8 spatial detector
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            conf_threshold: minimum confidence for detections
        """
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.conf_threshold = conf_threshold
        
        # Indoor navigation relevant classes (COCO dataset)
        self.indoor_classes = {
            56: 'chair', 60: 'dining table', 57: 'couch', 59: 'bed',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 73: 'book', 74: 'clock',
            75: 'vase', 39: 'bottle', 41: 'cup', 42: 'fork',
            43: 'knife', 44: 'spoon', 45: 'bowl'
        }
        
    def detect_objects(self, image):
        """
        Detect objects in image
        Args:
            image: numpy array [H, W, 3] in RGB
        Returns:
            detections: list of dicts with {class, confidence, bbox, position}
        """
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()
                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                
                # Calculate center position
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Get class name
                class_name = self.model.names[cls_id]
                
                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': bbox,
                    'center': (center_x, center_y),
                    'is_indoor': cls_id in self.indoor_classes
                })
        
        return detections
    
    def get_spatial_direction(self, image, detections=None):
        """
        Determine spatial direction based on object positions
        Args:
            image: numpy array [H, W, 3]
            detections: optional pre-computed detections
        Returns:
            direction: 'left', 'center', 'right', or 'ahead'
            primary_object: name of most prominent object
            confidence: direction confidence score
        """
        if detections is None:
            detections = self.detect_objects(image)
        
        # Filter for indoor objects
        indoor_detections = [d for d in detections if d['is_indoor']]
        
        if len(indoor_detections) == 0:
            return 'ahead', None, 0.0
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Find most prominent object (largest area × confidence)
        primary_detection = max(
            indoor_detections,
            key=lambda d: (d['bbox'][2] - d['bbox'][0]) * 
                         (d['bbox'][3] - d['bbox'][1]) * 
                         d['confidence']
        )
        
        # Calculate weighted center of all objects
        weighted_x = 0
        total_weight = 0
        
        for det in indoor_detections:
            area = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            weight = area * det['confidence']
            weighted_x += det['center'][0] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_x /= total_weight
        else:
            weighted_x = img_width / 2
        
        # Determine direction based on position
        # Divide image into zones
        left_boundary = img_width * 0.33
        right_boundary = img_width * 0.67
        
        if weighted_x < left_boundary:
            direction = 'left'
        elif weighted_x > right_boundary:
            direction = 'right'
        else:
            direction = 'center'
        
        # Calculate confidence based on how far from center
        center_dist = abs(weighted_x - img_width / 2) / (img_width / 2)
        confidence = min(1.0, center_dist * 1.5)  # Scale up for stronger signal
        
        return direction, primary_detection['class'], confidence
    
    def get_navigation_cue(self, image, caption=None):
        """
        Generate complete navigation cue combining detection and caption
        Args:
            image: numpy array [H, W, 3]
            caption: optional text caption from video model
        Returns:
            navigation_cue: formatted string for TTS
        """
        detections = self.detect_objects(image)
        direction, primary_obj, confidence = self.get_spatial_direction(image, detections)
        
        # Count objects
        num_objects = len([d for d in detections if d['is_indoor']])
        
        # Build cue
        if primary_obj:
            if caption:
                cue = f"{direction.title()}: {caption} - {primary_obj} detected"
            else:
                cue = f"{direction.title()}: {primary_obj} detected"
        else:
            if caption:
                cue = f"{direction.title()}: {caption}"
            else:
                cue = f"{direction.title()}: Path clear"
        
        return {
            'cue': cue,
            'direction': direction,
            'primary_object': primary_obj,
            'object_count': num_objects,
            'confidence': confidence,
            'all_objects': [d['class'] for d in detections if d['is_indoor']]
        }
    
    def visualize_detections(self, image, detections):
        """
        Draw bounding boxes on image for debugging
        Args:
            image: numpy array [H, W, 3]
            detections: list of detection dicts
        Returns:
            annotated_image: image with boxes drawn
        """
        img = image.copy()
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            color = (0, 255, 0) if det['is_indoor'] else (255, 0, 0)
            
            # Draw box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img

if __name__ == "__main__":
    # Test spatial detector
    detector = SpatialDetector(model_size='n')
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect
    detections = detector.detect_objects(test_image)
    print(f"Detected {len(detections)} objects")
    
    # Get navigation cue
    nav_cue = detector.get_navigation_cue(
        test_image, 
        caption="person sitting on chair"
    )
    print(f"\nNavigation cue: {nav_cue['cue']}")
    print(f"Direction: {nav_cue['direction']}")
    print(f"Primary object: {nav_cue['primary_object']}")
    print(f"Confidence: {nav_cue['confidence']:.2f}")
