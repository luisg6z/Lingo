"""
Font utilities for the Lingo project.
Provides Ubuntu font loading across the project.
"""
import os
import sys
from PIL import ImageFont


def _get_project_root():
    """
    Get the project root directory.
    
    Returns:
        str: Path to project root directory
    """
    # Get the directory where this file is located
    current_file = os.path.abspath(__file__)
    # Go up from src/core/font_utils.py to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root


def get_ubuntu_font_path():
    """
    Get the path to Ubuntu font file.
    First checks the resources/fonts folder in the project, then searches
    common font directories on Windows, Linux, and macOS.
    
    Returns:
        str: Path to Ubuntu font file, or None if not found
    """
    # Common Ubuntu font filenames
    ubuntu_fonts = [
        "Ubuntu-Regular.ttf",
        "Ubuntu-Bold.ttf",
        "Ubuntu-Light.ttf",
        "Ubuntu-Medium.ttf",
    ]
    
    # First, check the resources/fonts folder in the project
    project_root = _get_project_root()
    resources_fonts_dir = os.path.join(project_root, "resources", "fonts")
    
    if os.path.exists(resources_fonts_dir):
        for font_name in ubuntu_fonts:
            font_path = os.path.join(resources_fonts_dir, font_name)
            if os.path.exists(font_path):
                return font_path
    
    # If not found in resources, search system directories
    search_dirs = []
    
    if sys.platform == "win32":
        # Windows font directories
        windir = os.environ.get("WINDIR", "C:/Windows")
        search_dirs.extend([
            os.path.join(windir, "Fonts"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
        ])
    elif sys.platform.startswith("linux"):
        # Linux font directories
        data_home = os.environ.get("XDG_DATA_HOME")
        if not data_home:
            data_home = os.path.expanduser("~/.local/share")
        xdg_dirs = [data_home]
        
        data_dirs = os.environ.get("XDG_DATA_DIRS")
        if not data_dirs:
            data_dirs = "/usr/local/share:/usr/share"
        xdg_dirs.extend(data_dirs.split(":"))
        
        search_dirs.extend([os.path.join(xdg_dir, "fonts") for xdg_dir in xdg_dirs])
        search_dirs.extend([
            os.path.expanduser("~/.fonts"),
            "/usr/share/fonts/truetype",
            "/usr/share/fonts/TTF",
        ])
    elif sys.platform == "darwin":
        # macOS font directories
        search_dirs.extend([
            "/Library/Fonts",
            "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ])
    
    # Search for Ubuntu font in system directories
    for font_name in ubuntu_fonts:
        for directory in search_dirs:
            if not directory:
                continue
            font_path = os.path.join(directory, font_name)
            if os.path.exists(font_path):
                return font_path
    
    return None


def get_ubuntu_font(font_scale=1.0, size=None, bold=False):
    """
    Get Ubuntu font object for PIL/Pillow.
    
    Args:
        font_scale: Scale factor for font size (default: 1.0)
        size: Direct font size in pixels (overrides font_scale if provided)
        bold: If True, try to load Ubuntu-Bold, otherwise Ubuntu-Regular
    
    Returns:
        ImageFont: Ubuntu font object, or default font if Ubuntu not found
    """
    font_path = get_ubuntu_font_path()
    
    if font_path:
        try:
            # Calculate font size
            if size is None:
                # Default size calculation based on scale
                # FONT_HERSHEY_DUPLEX with scale 0.9 renders visually as ~22-24px
                font_size = max(int(22 * font_scale), 16)
            else:
                font_size = size
            
            # Try to get bold version if requested
            if bold:
                font_dir = os.path.dirname(font_path)
                # Try Ubuntu-Bold first, then Ubuntu-Medium
                for variant in ["Ubuntu-Bold.ttf", "Ubuntu-Medium.ttf"]:
                    bold_path = os.path.join(font_dir, variant)
                    if os.path.exists(bold_path):
                        font_path = bold_path
                        break
            
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Warning: Could not load Ubuntu font from {font_path}: {e}")
            return ImageFont.load_default()
    else:
        print("Warning: Ubuntu font not found. Using default font.")
        return ImageFont.load_default()


def get_ubuntu_font_path_for_customtkinter():
    """
    Get Ubuntu font path for use with customtkinter.
    
    Returns:
        str: Font family name "Ubuntu" or fallback to system default
    """
    font_path = get_ubuntu_font_path()
    if font_path:
        return "Ubuntu"
    return None


def put_text_ubuntu(img, text, position, font_scale, color, thickness, line_type=None, bold=False):
    """
    Render text using Ubuntu font from resources/fonts.
    This is a replacement for cv2.putText that uses Ubuntu font.
    
    Args:
        img: OpenCV image (numpy array)
        text: Text string to render
        position: (x, y) tuple for text position
        font_scale: Font scale factor (similar to cv2 font scale)
        color: BGR color tuple
        thickness: Line thickness
        line_type: OpenCV line type (defaults to cv2.LINE_AA)
        bold: If True, use bold Ubuntu font variant
    
    Returns:
        None (modifies img in-place)
    """
    import cv2
    import numpy as np
    
    if line_type is None:
        line_type = cv2.LINE_AA
    
    try:
        from PIL import Image, ImageDraw
        
        # Convert OpenCV image to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Get Ubuntu font with bold support
        font = get_ubuntu_font(font_scale=font_scale, bold=bold)
        
        # Get text bounding box for positioning
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, 0, 0)
        
        # Convert color from BGR to RGB
        color_rgb = (color[2], color[1], color[0])
        
        # Draw text
        x, y = position
        draw.text((x, y), text, fill=color_rgb, font=font)
        
        # Convert back to OpenCV format
        img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img[:] = img_result[:]
    except Exception as e:
        # Fallback to OpenCV if PIL fails
        # Use a default font face for fallback
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font_face, font_scale, color, thickness, line_type)


def calculate_word_positions(text, start_x, start_y, font_scale, max_width, bold=False):
    """
    Calculate the exact positions (start and end x coordinates) for each word in the text.
    Handles word wrapping and uses PIL for accurate text measurement.

    Args:
        text: The text string to analyze
        start_x: Starting x coordinate
        start_y: Starting y coordinate (baseline)
        font_scale: Font scale factor
        max_width: Maximum width before wrapping to next line
        bold: Whether to use bold font

    Returns:
        List of dicts with keys: 'word', 'x_start', 'x_end', 'y', 'line_index'
    """
    words = text.split()
    word_positions = []
    x_current = start_x
    y_current = start_y
    line_index = 0

    try:
        from PIL import Image, ImageDraw

        font = get_ubuntu_font(font_scale=font_scale, bold=bold)
        img_pil = Image.new('RGB', (100, 100), (0, 0, 0))
        draw = ImageDraw.Draw(img_pil)

        for word in words:
            word_with_space = word + " "
            try:
                bbox = draw.textbbox((0, 0), word_with_space, font=font)
                word_width = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font.getbbox(word_with_space) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                word_width = bbox[2] - bbox[0]

            if x_current + word_width > max_width and x_current > start_x:
                y_current += int(font_scale * 35)
                x_current = start_x
                line_index += 1

            word_start = x_current
            word_end = x_current + word_width
            word_positions.append({
                'word': word,
                'x_start': word_start,
                'x_end': word_end,
                'y': y_current,
                'line_index': line_index
            })
            x_current = word_end

    except Exception:
        import cv2
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        for word in words:
            word_with_space = word + " "
            (word_width, word_height), baseline = cv2.getTextSize(word_with_space, font_face, font_scale, 2)
            if x_current + word_width > max_width and x_current > start_x:
                y_current += word_height + 5
                x_current = start_x
                line_index += 1
            word_start = x_current
            word_end = x_current + word_width
            word_positions.append({
                'word': word,
                'x_start': word_start,
                'x_end': word_end,
                'y': y_current,
                'line_index': line_index
            })
            x_current = word_end

    return word_positions

