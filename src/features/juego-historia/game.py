"""
Juego historia feature (placeholder)
"""
import cv2
import numpy as np
import os
import sys
from openni import openni2

# Get project root and feature root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
feature_root = os.path.dirname(__file__)
assets_path = os.path.join(feature_root, 'assets')
images_path = os.path.join(assets_path, 'images')
sounds_path = os.path.join(assets_path, 'sounds')
config_path = os.path.join(feature_root, 'config')

# Add project root to path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import font utilities
from src.core.font_utils import put_text_ubuntu


def juego_historia(device, coordenadas=None, dmax_map=None, dmin_map=None):
    """
    Placeholder for the juego-historia game feature.
    
    Args:
        device: OpenNI2 device
        coordenadas: Calibration coordinates (optional)
        dmax_map: Maximum depth map (optional)
        dmin_map: Minimum depth map (optional)
    """
    print("Juego Historia - Placeholder")
    print("This feature is not yet implemented.")
    
    # Create a simple placeholder screen
    view_width = 1280
    view_height = 800
    screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Draw placeholder text using Ubuntu font
    text = "Juego Historia - Coming Soon"
    font_scale = 2.0
    thickness = 4
    # Get text size for centering (using approximate size)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_x = (view_width - text_size[0]) // 2
    text_y = view_height // 2
    
    put_text_ubuntu(screen, text, (text_x, text_y), font_scale, (0, 255, 255), thickness)
    
    # Show the screen
    cv2.namedWindow("Juego Historia", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Juego Historia", 1920, 0)
    cv2.setWindowProperty("Juego Historia", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Juego Historia", screen)
    
    # Wait for key press
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

