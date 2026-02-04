import cv2
import numpy as np
from ultralytics import YOLO
import os

# Model path
MODEL_PATH = r"runs\detect\runs\detect\yolo12_animal_detection\weights\best.pt"

# Category Mapping based on USER provided classes
CATEGORY_MAP = {
    # Animals
    "Cat": "Escenario 1",
    "Dog": "Escenario 1",
    "Pig": "Escenario 1",
    "bird": "Escenario 1",
    "cow": "Escenario 1",
    "duck": "Escenario 1",
    "hen": "Escenario 1",
    "horse": "Escenario 1",
    "sheep": "Escenario 1",
    
    # Vehicles
    "Skateboard": "Escenario 2",
    "Train": "Escenario 2",
    "Van": "Escenario 2",
    "bike": "Escenario 2",
    "bus": "Escenario 2",
    "car": "Escenario 2",
    "motorbike": "Escenario 2",
    
    # Food
    "apple": "Escenario 3",
    "banana": "Escenario 3",
    "cake": "Escenario 3",
    "hamburger": "Escenario 3",
    "hot dog": "Escenario 3",
    "milkshakes": "Escenario 3",
    "orange": "Escenario 3",
    "pizza": "Escenario 3",
    "potatoes": "Escenario 3",
    "soda": "Escenario 3",
    "spaghetti": "Escenario 3",
    "strawberry": "Escenario 3",
    
    # Clothes
    "cap": "Escenario 4",
    "glasses": "Escenario 4",
    "pants": "Escenario 4",
    "sandals": "Escenario 4",
    "shirt": "Escenario 4",
    "shoes": "Escenario 4",
    "socks": "Escenario 4",
    "sweater": "Escenario 4",
    
    # School Supplies
    "backpack": "Escenario 5",
    "book": "Escenario 5",
    "colors": "Escenario 5",
    "eraser": "Escenario 5",
    "notebook": "Escenario 5",
    "pencil": "Escenario 5",
    "ruler": "Escenario 5"
}

class ObjectDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.names = self.model.names
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect(self, frame, conf=0.3):
        if self.model is None:
            return []
        
        results = self.model(frame, conf=conf, verbose=False)
        detections = []
        
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                category = CATEGORY_MAP.get(label, None)
                
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "category": category
                })
        
        return detections

    def map_coordinates(self, x, y, coordenadas):
        """
        Maps camera coordinates to projection coordinates.
        camera: 640x480 -> projection: 1280x800
        """
        xw_min = coordenadas["xw_min"]
        xw_max = coordenadas["xw_max"]
        yw_min = coordenadas["yw_min"]
        yw_max = coordenadas["yw_max"]
        xv_min = coordenadas["xv_min"]
        xv_max = coordenadas["xv_max"]
        yv_min = coordenadas["yv_min"]
        yv_max = coordenadas["yv_max"]
        
        # Simple linear mapping (as seen in EXPLICACION_SISTEMA.md fallbacks)
        # Note: If homography is available and needed, it should be passed here.
        sx = float(xv_max - xv_min) / (xw_max - xw_min)
        sy = float(yv_max - yv_min) / (yw_max - yw_min)
        
        x_proj = xv_min + (x - xw_min) * sx
        y_proj = yv_min + (y - yw_min) * sy
        
        return int(x_proj), int(y_proj)

def draw_shine_effect(screen, x, y, width, height, color=(255, 255, 255)):
    """
    Draws a pulse/shine effect around the detected object.
    """
    # Simple glowing border for now
    overlay = screen.copy()
    cv2.rectangle(overlay, (x-5, y-5), (x+width+5, y+height+5), color, 5)
    cv2.addWeighted(overlay, 0.4, screen, 0.6, 0, screen)
