"""
Rectangular button component for MagicboARd games.
Provides reusable rectangular buttons with text using Ubuntu font.
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


def draw_rectangular_button(
    screen: np.ndarray,
    x: int, y: int, width: int, height: int,
    text: str,
    bg_color: tuple = (0, 200, 0),  # Green default
    border_color: tuple = (255, 255, 255),
    border_thickness: int = 3,
    text_color: tuple = (255, 255, 255),
    font_scale: float = 1.0,
    bold: bool = True,
    shadow: bool = True,
    fill: bool = True  # Whether to fill the rectangle
) -> dict:
    """
    Draw a rectangular button with text on the screen.
    
    Args:
        screen: OpenCV image (numpy array) to draw on
        x: Top-left x coordinate of button
        y: Top-left y coordinate of button
        width: Button width
        height: Button height
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
            'x1': int, 'y1': int, 'x2': int, 'y2': int,
            'text_x': int, 'text_y': int
        }
    """
    # Draw button rectangle with fill (if fill is True)
    if fill:
        cv2.rectangle(screen, (x, y), (x + width, y + height), bg_color, -1)
    # Draw button border
    cv2.rectangle(screen, (x, y), (x + width, y + height), border_color, border_thickness)
    
    # Calculate text size and position for centering
    font_button = cv2.FONT_HERSHEY_DUPLEX
    thickness_button = 3 if bold else 2
    
    # Get text size using PIL for accurate centering with Ubuntu font
    bbox_top = 0
    if PIL_AVAILABLE:
        try:
            font = get_ubuntu_font(font_scale=font_scale, bold=bold)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]  # Top of bounding box (can be negative for ascenders)
            except AttributeError:
                bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]
        except:
            text_size, _ = cv2.getTextSize(text, font_button, font_scale, thickness_button)
            text_width = text_size[0]
            text_height = text_size[1]
            bbox_top = 0
    else:
        text_size, _ = cv2.getTextSize(text, font_button, font_scale, thickness_button)
        text_width = text_size[0]
        text_height = text_size[1]
        bbox_top = 0
    
    # Center text in button
    text_x = x + (width - text_width) // 2
    # Center vertically: use bbox_top to adjust for proper vertical centering
    if bbox_top < 0:
        text_y = y + height // 2 + abs(bbox_top) // 2 - text_height // 2
    else:
        text_y = y + height // 2 - text_height // 2
    
    # Draw text with shadow if enabled
    if shadow:
        put_text_ubuntu(screen, text, (text_x + 2, text_y + 2), 
                       font_scale, (0, 0, 0), thickness_button + 1, bold=bold)
    
    # Draw main text
    put_text_ubuntu(screen, text, (text_x, text_y), 
                   font_scale, text_color, thickness_button, bold=bold)
    
    return {
        'x1': x,
        'y1': y,
        'x2': x + width,
        'y2': y + height,
        'text_x': text_x,
        'text_y': text_y
    }


def is_point_in_rectangular_button(x: int, y: int, button_bounds: dict) -> bool:
    """
    Check if a point is inside a rectangular button.
    
    Args:
        x: X coordinate of point
        y: Y coordinate of point
        button_bounds: Dictionary with 'x1', 'y1', 'x2', 'y2' keys
    
    Returns:
        True if point is inside button, False otherwise
    """
    return (button_bounds['x1'] <= x <= button_bounds['x2'] and
            button_bounds['y1'] <= y <= button_bounds['y2'])

