"""
Detection utilities for MagicboARd
"""
import cv2
import numpy as np


def detect_color_and_shape(image, min_contour_area=250):
    """
    Detecta colores y formas en una imagen.
    
    Args:
        image: Imagen BGR de OpenCV
        min_contour_area: Área mínima del contorno para considerar válido
    
    Returns:
        Lista de tuplas (shape, color_name, contour)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        "Rojo": ([170, 50, 50], [180, 255, 255]),
        "Verde": ([40, 50, 50], [90, 255, 255]),
        "Azul": ([90, 50, 50], [130, 255, 255]),
        "Amarillo": ([20, 100, 100], [30, 255, 255]),
        "Naranja": ([0, 100, 100], [10, 255, 255]),
        "Morado": ([130, 50, 50], [160, 255, 255]),
    }

    detected_shapes = []

    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                shape = None
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) > 4:
                    shape = "círculo"
                elif len(approx) == 4:
                    shape = "cuadrado"
                elif len(approx) == 3:
                    shape = "triángulo"

                detected_shapes.append((shape, color_name, cnt))

    return detected_shapes

