import cv2
import numpy as np
from ultralytics import YOLO
import os

from src.core.calibration import map_camera_to_viewport

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


def preprocess_frame_for_detection(bgr_image):
    """
    Atenúa luces altas y reflejos en figuras brillantes y reparte mejor el contraste
    antes de pasar el frame a YOLO (sin cambiar la resolución ni el layout BGR).
    """
    if bgr_image is None or bgr_image.size == 0:
        return bgr_image

    x = bgr_image.astype(np.float32) / 255.0
    # Compresión tipo Reinhard suave: reduce píxeles quemados sin aplastar sombras
    x = x / (1.0 + 0.35 * x)
    bgr = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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

        frame = preprocess_frame_for_detection(frame)

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

    def map_coordinates(self, x, y, coordenadas, view_width=1280, view_height=800):
        """
        Pasa coordenadas del frame de color 640×480 (flipped) al viewport.
        Si ``coordenadas`` incluye ``homography_matrix``, usa la misma corrección
        que la detección de toques; si no, escalado lineal sobre xv_/xw_.
        """
        return map_camera_to_viewport(x, y, coordenadas, view_width, view_height)

def draw_shine_effect(screen, x, y, width, height, color=(255, 255, 255)):
    """
    Draws a pulse/shine effect around the detected object.
    """
    # Simple glowing border for now
    overlay = screen.copy()
    cv2.rectangle(overlay, (x-5, y-5), (x+width+5, y+height+5), color, 5)
    cv2.addWeighted(overlay, 0.4, screen, 0.6, 0, screen)
