# -*- coding: utf-8 -*-
"""
Test script for YOLO12 animal detection model using Kinect camera
"""
import sys
import io

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import cv2
import numpy as np
from pathlib import Path
from openni import openni2
from ultralytics import YOLO

# Check OpenCV GUI support
def check_opencv_gui():
    """Check if OpenCV has GUI support"""
    try:
        # Try to create a test window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("test")
        return True
    except cv2.error as e:
        if "not implemented" in str(e).lower() or "gtk" in str(e).lower():
            return False
        raise
    except Exception:
        return False

# Configuration
OPENNI2_PATH = "C:/Program Files/OpenNI2/Redist"
CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold for detections
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text
BACKGROUND_COLOR = (0, 0, 0)  # Black background for text
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 0),    # Dark Green
]

def initialize_kinect():
    """Initialize OpenNI2 and Kinect device"""
    print("Initializing OpenNI2...")
    
    # Try multiple paths for OpenNI2
    openni2_paths = [
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
        os.path.join(os.environ.get("ProgramFiles", ""), "OpenNI2", "Redist"),
    ]
    
    openni2_initialized = False
    for path in openni2_paths:
        if os.path.exists(path):
            try:
                openni2.initialize(path)
                openni2_initialized = True
                print(f"[OK] OpenNI2 initialized from: {path}")
                break
            except Exception as e:
                print(f"[WARNING] Failed to initialize from {path}: {e}")
                continue
    
    if not openni2_initialized:
        raise RuntimeError("Could not initialize OpenNI2. Please check installation.")
    
    # Open device
    print("Opening Kinect device...")
    try:
        device = openni2.Device.open_any()
        print("[OK] Kinect device opened")
    except Exception as e:
        raise RuntimeError(f"Could not open Kinect device: {e}")
    
    # Check sensors
    if not device.has_sensor(openni2.SENSOR_COLOR):
        raise RuntimeError("Kinect device does not have color sensor")
    
    print("[OK] Color sensor available")
    return device

def create_color_stream(device):
    """Create and start color stream"""
    print("Creating color stream...")
    try:
        color_stream = device.create_color_stream()
        color_stream.start()
        print("[OK] Color stream started")
        return color_stream
    except Exception as e:
        raise RuntimeError(f"Could not start color stream: {e}")

def find_model():
    """Find the trained model in various possible locations"""
    base_dir = Path(__file__).parent.parent
    
    # List of possible model paths to try
    possible_paths = [
        # Primary location - best model from training
        base_dir / 'yolo_training' / 'runs' / 'yolo12_animal_detection' / 'weights' / 'best.pt',
        # Alternative - last checkpoint
        base_dir / 'yolo_training' / 'runs' / 'yolo12_animal_detection' / 'weights' / 'last.pt',
        # Exports folder
        base_dir / 'yolo_training' / 'exports' / 'best.pt',
        # Direct in runs folder (if structure is different)
        base_dir / 'runs' / 'yolo12_animal_detection' / 'weights' / 'best.pt',
        base_dir / 'runs' / 'yolo12_animal_detection' / 'weights' / 'last.pt',
        # Nested runs/detect structure (common when training from different locations)
        base_dir / 'runs' / 'detect' / 'runs' / 'detect' / 'yolo12_animal_detection' / 'weights' / 'best.pt',
        base_dir / 'runs' / 'detect' / 'runs' / 'detect' / 'yolo12_animal_detection' / 'weights' / 'last.pt',
        base_dir / 'runs' / 'detect' / 'yolo12_animal_detection' / 'weights' / 'best.pt',
        base_dir / 'runs' / 'detect' / 'yolo12_animal_detection' / 'weights' / 'last.pt',
    ]
    
    print("Searching for trained model...")
    for model_path in possible_paths:
        if model_path.exists():
            print(f"[OK] Found model at: {model_path}")
            return model_path
    
    # If not found in specific paths, do a broader search
    print("\nSearching more broadly in runs directories...")
    runs_dirs = [
        base_dir / 'runs',
        base_dir / 'yolo_training' / 'runs',
    ]
    
    for runs_dir in runs_dirs:
        if runs_dir.exists():
            # Search for any best.pt or last.pt in weights folders
            for weights_file in runs_dir.rglob('weights/best.pt'):
                print(f"[OK] Found model at: {weights_file}")
                return weights_file
            for weights_file in runs_dir.rglob('weights/last.pt'):
                print(f"[OK] Found model at: {weights_file}")
                return weights_file
    
    # If no model found, show all searched paths
    print("\n[ERROR] Model not found in any of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nPlease train the model first using the Jupyter notebook:")
    print("  jupyter notebook yolo_training/train_yolo12.ipynb")
    
    raise FileNotFoundError("Trained model not found. Please train the model first.")

def load_model():
    """Load YOLO12 model"""
    model_path = find_model()
    
    try:
        print(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
        print("[OK] Model loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")

def draw_detections(frame, results, class_names):
    """Draw bounding boxes and labels on frame"""
    annotated_frame = frame.copy()
    detection_count = 0
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class and confidence
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Get class name
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            
            # Choose color based on class
            color = COLORS[cls % len(COLORS)]
            
            # Draw bounding box (thicker for high confidence)
            thickness = 3 if conf > 0.7 else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - baseline - 10),
                (x1 + label_width + 5, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, y1 - baseline - 5),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS
            )
            
            detection_count += 1
    
    return annotated_frame, detection_count

def main():
    """Main function"""
    print("=" * 60)
    print("YOLO12 Animal Detection - Kinect Camera Test")
    print("=" * 60)
    
    # Check OpenCV GUI support
    print("\nChecking OpenCV GUI support...")
    if not check_opencv_gui():
        print("\n" + "="*60)
        print("ERROR: OpenCV GUI is not available!")
        print("="*60)
        print("\nOpenCV was installed without GUI support.")
        print("This is required to display the camera feed.")
        print("\nSOLUTION:")
        print("1. Uninstall current OpenCV:")
        print("   pip uninstall opencv-python opencv-contrib-python")
        print("\n2. Reinstall with GUI support:")
        print("   pip install opencv-python")
        print("\n3. If you're in a virtual environment, make sure to:")
        print("   - Activate the virtual environment first")
        print("   - Then run the pip install command")
        print("\n4. After reinstalling, run this script again.")
        print("="*60)
        return
    
    print("[OK] OpenCV GUI support available")
    
    device = None
    color_stream = None
    
    try:
        # Initialize Kinect
        device = initialize_kinect()
        color_stream = create_color_stream(device)
        
        # Load model
        model = load_model()
        
        # Get class names from model
        class_names = model.names
        print(f"\nDetecting {len(class_names)} classes:")
        for idx, name in class_names.items():
            print(f"  {idx}: {name}")
        
        print("\n" + "=" * 60)
        print("Starting detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        print("=" * 60)
        
        # FPS calculation
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        current_confidence = CONFIDENCE_THRESHOLD
        
        while True:
            # Read frame from Kinect
            frame = color_stream.read_frame()
            if frame is None:
                continue
            
            # Convert frame to numpy array
            rgb_data = np.frombuffer(
                frame.get_buffer_as_uint8(),
                dtype=np.uint8
            ).reshape(480, 640, 3)
            
            # Convert RGB to BGR for OpenCV
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            
            # Flip horizontally (mirror effect)
            bgr_data = cv2.flip(bgr_data, 1)
            
            # Run inference
            inference_start = time.time()
            results = model(bgr_data, conf=current_confidence, imgsz=640, verbose=False)
            inference_time = (time.time() - inference_start) * 1000  # in ms
            
            # Draw detections
            annotated_frame, detection_count = draw_detections(bgr_data, results, class_names)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                elapsed = time.time() - fps_start_time
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
            
            # Draw info overlay
            info_y = 30
            line_height = 25
            
            # FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, info_y),
                FONT,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Detection count
            info_y += line_height
            cv2.putText(
                annotated_frame,
                f"Detections: {detection_count}",
                (10, info_y),
                FONT,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Confidence threshold
            info_y += line_height
            cv2.putText(
                annotated_frame,
                f"Confidence: {current_confidence:.2f}",
                (10, info_y),
                FONT,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Inference time
            info_y += line_height
            cv2.putText(
                annotated_frame,
                f"Inference: {inference_time:.1f}ms",
                (10, info_y),
                FONT,
                0.7,
                (255, 0, 255),
                2
            )
            
            # Display frame (with error handling for OpenCV GUI issues)
            try:
                cv2.imshow("YOLO12 Animal Detection - Kinect", annotated_frame)
            except cv2.error as e:
                if "not implemented" in str(e).lower() or "gtk" in str(e).lower():
                    print("\n" + "="*60)
                    print("ERROR: OpenCV GUI not available!")
                    print("="*60)
                    print("\nThis usually means OpenCV was installed without GUI support.")
                    print("\nSOLUTION: Reinstall opencv-python:")
                    print("  pip uninstall opencv-python opencv-contrib-python")
                    print("  pip install opencv-python")
                    print("\nOr if you need headless (no GUI) version:")
                    print("  pip install opencv-python-headless")
                    print("="*60)
                    raise
                else:
                    raise
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('+') or key == ord('='):
                current_confidence = min(0.95, current_confidence + 0.05)
                print(f"Confidence threshold increased to: {current_confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                current_confidence = max(0.05, current_confidence - 0.05)
                print(f"Confidence threshold decreased to: {current_confidence:.2f}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        if color_stream is not None:
            try:
                color_stream.stop()
                print("[OK] Color stream stopped")
            except:
                pass
        
        if device is not None:
            try:
                openni2.unload()
                print("[OK] OpenNI2 unloaded")
            except:
                pass
        
        # Safely destroy windows
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
            # Ignore errors when destroying windows (common if window was never created)
            if "not implemented" not in str(e).lower():
                print(f"[WARNING] Error closing windows: {e}")
        
        print("Done!")

if __name__ == "__main__":
    main()



