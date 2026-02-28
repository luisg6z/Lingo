"""
Circular button component for MagicboARd games.
Provides reusable circular buttons with text using Ubuntu font.
"""
import cv2
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.font_utils import get_ubuntu_font, put_text_ubuntu

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def draw_circular_button(
    screen: np.ndarray,
    center_x: int, center_y: int, radius: int,
    text: str,
    bg_color: tuple = (0, 200, 0),  # Green default
    border_color: tuple = (255, 255, 255),
    border_thickness: int = 3,
    text_color: tuple = (255, 255, 255),
    font_scale: float = 1.3,
    bold: bool = True,
    shadow: bool = True
) -> dict:
    """
    Draw a circular button with text on the screen.
    
    Args:
        screen: OpenCV image (numpy array) to draw on
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Circle radius
        text: Text to display on button
        bg_color: Background color (BGR tuple)
        border_color: Border color (BGR tuple)
        border_thickness: Border thickness in pixels
        text_color: Text color (BGR tuple)
        font_scale: Font scale factor
        bold: Whether to use bold font
        shadow: Whether to draw text shadow
    
    Returns:
        Dictionary with button bounds and text position:
        {
            'center_x': int, 'center_y': int, 'radius': int,
            'text_x': int, 'text_y': int,
            'bounds': {'x1': int, 'y1': int, 'x2': int, 'y2': int}
        }
    """
    # Draw circle shadow
    if shadow:
        cv2.circle(screen, (center_x + 3, center_y + 3), radius, (0, 0, 0), -1)
    
    # Draw circle with fill
    cv2.circle(screen, (center_x, center_y), radius, bg_color, -1)
    # Draw circle border
    cv2.circle(screen, (center_x, center_y), radius, border_color, border_thickness)
    
    # Calculate text size and position for centering
    font_button = cv2.FONT_HERSHEY_DUPLEX
    thickness_button = 5 if bold else 3
    
    # Get text size using PIL for accurate centering with Ubuntu font
    text_width = 0
    text_height = 0
    bbox_top = 0
    bbox_bottom = 0
    
    if PIL_AVAILABLE:
        try:
            font = get_ubuntu_font(font_scale=font_scale, bold=bold)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]  # Top of text (usually negative for fonts with ascenders)
                bbox_bottom = bbox[3]  # Bottom of text (baseline area)
            except AttributeError:
                bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]
                bbox_bottom = bbox[3]
        except:
            text_size, _ = cv2.getTextSize(text, font_button, font_scale, thickness_button)
            text_width = text_size[0]
            text_height = text_size[1]
            bbox_top = 0
            bbox_bottom = text_height
    else:
        text_size, _ = cv2.getTextSize(text, font_button, font_scale, thickness_button)
        text_width = text_size[0]
        text_height = text_size[1]
        bbox_top = 0
        bbox_bottom = text_height
    
    # Center text in circle
    # Horizontal centering: center_x - half of text width
    text_x = center_x - text_width // 2
    
    # Vertical centering: PIL uses y as baseline (the bottom line where text sits)
    # The bbox gives us: (left, top, right, bottom) where:
    # - top is usually negative (for ascenders like 'H')
    # - bottom is near 0 when drawing from baseline
    # To center vertically, we want the visual center at center_y
    # Visual center ≈ baseline - abs(top)/2
    # So: center_y = baseline - abs(top)/2  =>  baseline = center_y + abs(top)/2
    # But we want to move UP, so we subtract instead of add
    if PIL_AVAILABLE and bbox_top < 0:
        # Use bbox top offset to calculate proper baseline position
        # Subtract more to move text higher up
        text_y = center_y - abs(bbox_top) - text_height // 3
    else:
        # Fallback: subtract more of height to move text up more
        text_y = center_y - text_height // 2  # Move text up more
    
    # Draw text with shadow if enabled
    if shadow:
        put_text_ubuntu(screen, text, (text_x + 2, text_y + 2), 
                       font_scale, (0, 0, 0), thickness_button + 1, bold=bold)
    
    # Draw main text
    put_text_ubuntu(screen, text, (text_x, text_y), 
                   font_scale, text_color, thickness_button, bold=bold)
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'radius': radius,
        'text_x': text_x,
        'text_y': text_y,
        'bounds': {
            'x1': center_x - radius,
            'y1': center_y - radius,
            'x2': center_x + radius,
            'y2': center_y + radius
        }
    }


def is_point_in_circular_button(x: int, y: int, button_bounds: dict) -> bool:
    """
    Check if a point is inside a circular button.
    
    Args:
        x: X coordinate of point
        y: Y coordinate of point
        button_bounds: Dictionary with 'center_x', 'center_y', 'radius' keys
    
    Returns:
        True if point is inside button, False otherwise
    """
    import math
    center_x = button_bounds['center_x']
    center_y = button_bounds['center_y']
    radius = button_bounds['radius']
    
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius

