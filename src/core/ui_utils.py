"""
UI utilities for MagicboARd
"""
import cv2
import numpy as np
import os
import pygame


def scale_to_videobeam(image, source_width=1280, source_height=800, videobeam_width=1920, videobeam_height=1080):
    """
    Escala una imagen de la resolución fuente a la resolución del videobeam.
    
    Args:
        image: Imagen a escalar (numpy array)
        source_width: Ancho de la imagen fuente (default: 1280)
        source_height: Alto de la imagen fuente (default: 800)
        videobeam_width: Ancho del videobeam (default: 1920)
        videobeam_height: Alto del videobeam (default: 1080)
    
    Returns:
        Imagen escalada a la resolución del videobeam
    """
    if image is None or image.size == 0:
        return image
    
    # Escalar la imagen para que llene toda la pantalla del videobeam
    scaled_image = cv2.resize(image, (videobeam_width, videobeam_height), interpolation=cv2.INTER_LINEAR)
    return scaled_image


def draw_logo(screen, view_width=1280, view_height=800):
    """
    Dibuja el logo en la parte superior de la pantalla.
    
    Args:
        screen: Pantalla donde dibujar (numpy array)
        view_width: Ancho de la vista (default: 1280)
        view_height: Alto de la vista (default: 800)
    
    Returns:
        bool: True si se cargó el logo, False si se usó texto fallback
    """
    logo_loaded_local = False
    logo_image_local = None
    
    # Get path relative to project root
    logo_png_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images", "logo.png")
    logo_svg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images", "logo.svg")
    
    try:
        # Intentar cargar como PNG primero
        if os.path.exists(logo_png_path):
            logo_image_local = cv2.imread(logo_png_path, cv2.IMREAD_UNCHANGED)
            if logo_image_local is not None:
                logo_loaded_local = True
        # Intentar cargar SVG usando pygame
        elif os.path.exists(logo_svg_path):
            try:
                logo_surface = pygame.image.load(logo_svg_path)
                logo_string = pygame.image.tostring(logo_surface, "RGBA")
                logo_np = np.frombuffer(logo_string, np.uint8)
                logo_image_local = logo_np.reshape((logo_surface.get_height(), logo_surface.get_width(), 4))
                logo_image_local = cv2.cvtColor(logo_image_local, cv2.COLOR_RGBA2BGRA)
                logo_loaded_local = True
            except Exception:
                logo_loaded_local = False
    except Exception:
        logo_loaded_local = False
    
    if logo_loaded_local and logo_image_local is not None:
        logo_height = 120
        if len(logo_image_local.shape) == 3:
            original_height, original_width = logo_image_local.shape[:2]
        else:
            original_height, original_width = logo_image_local.shape[0], logo_image_local.shape[1]
        
        aspect_ratio = original_width / original_height
        logo_width = int(logo_height * aspect_ratio)
        logo_resized = cv2.resize(logo_image_local, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
        logo_x = (view_width - logo_width) // 2
        logo_y = 30
        
        if logo_x >= 0 and logo_y >= 0 and logo_x + logo_width <= view_width and logo_y + logo_height <= view_height:
            if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:
                alpha = logo_resized[:, :, 3] / 255.0
                for c in range(3):
                    screen[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width, c] = (
                        alpha * logo_resized[:, :, c] + (1 - alpha) * screen[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width, c]
                    )
            elif len(logo_resized.shape) == 3:
                screen[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width] = logo_resized[:, :, :3]
            else:
                logo_bgr = cv2.cvtColor(logo_resized, cv2.COLOR_GRAY2BGR)
                screen[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width] = logo_bgr
        return True
    else:
        # Dibujar texto como fallback
        logo_text = "MagicboARd"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.5
        thickness = 4
        text_size, _ = cv2.getTextSize(logo_text, font, font_scale, thickness)
        text_x = (view_width - text_size[0]) // 2
        text_y = 80
        cv2.putText(screen, logo_text, (text_x + 3, text_y + 3), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(screen, logo_text, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), thickness)
        return False

