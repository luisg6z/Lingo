# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import random
import json
import threading
import pygame
import speech_recognition as sr
import pyaudio
from sentence_transformers import SentenceTransformer
import pyttsx3
import sys
import logging

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.features.confetti import ConfettiSystem
from src.core.font_utils import get_ubuntu_font
from src.core.calibration import map_depth_roi_to_viewport
from src.components import draw_circular_button, draw_rectangular_button, is_point_in_circular_button, is_point_in_rectangular_button

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Advertencia: PIL/Pillow no está disponible. Los caracteres acentuados pueden no mostrarse correctamente.")


def put_text_safe(img, text, position, font_face, font_scale, color, thickness, line_type=cv2.LINE_AA, bold=False):
    """
    Función auxiliar para renderizar texto usando Ubuntu font de resources/fonts.
    Siempre intenta usar Ubuntu font cuando PIL está disponible.
    Modifica la imagen in-place.
    
    Args:
        bold: If True, use bold Ubuntu font variant
    """
    # Siempre usar PIL con Ubuntu font si está disponible
    if PIL_AVAILABLE:
        try:
            # Convertir imagen OpenCV a PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Always use Ubuntu font from resources/fonts
            try:
                if get_ubuntu_font:
                    font = get_ubuntu_font(font_scale=font_scale, bold=bold)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Obtener el tamaño real del texto con PIL para centrado correcto
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width_pil = bbox[2] - bbox[0]
            
            # Obtener tamaño con OpenCV para comparar y ajustar centrado
            text_size_cv, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_width_cv = text_size_cv[0]
            
            # Ajustar posición x para centrado correcto basado en el ancho real de PIL
            x, y = position
            # Calcular la diferencia y ajustar para mantener el centrado
            width_diff = text_width_pil - text_width_cv
            x = x - width_diff // 2
            
            # Renderizar texto
            # Convertir color BGR a RGB
            color_rgb = (color[2], color[1], color[0])
            draw.text((x, y), text, fill=color_rgb, font=font)
            
            # Convertir de vuelta a OpenCV y copiar a la imagen original
            img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img[:] = img_result[:]
        except Exception as e:
            print(f"Error al renderizar texto con PIL: {e}, usando OpenCV")
            # Fallback a OpenCV
            cv2.putText(img, text, position, font_face, font_scale, color, thickness, line_type)
    else:
        # Fallback a OpenCV si PIL no está disponible
        cv2.putText(img, text, position, font_face, font_scale, color, thickness, line_type)


def _overlay_lingo_bien_title(screen, char_x, char_y, char_new_width, char_new_height):
    """Dibuja «¡Muy Bien!» encima del confeti (mismo estilo que create_end_screen)."""
    texto_a_dibujar = "¡Muy Bien!"
    font_message = cv2.FONT_HERSHEY_DUPLEX
    font_scale_message = 1.5
    thickness_message = 3
    message_color = (0, 255, 0)
    message_y = char_y + char_new_height + 20
    try:
        from PIL import Image, ImageDraw
        font_ubuntu = get_ubuntu_font(font_scale=font_scale_message, bold=True)
        img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            bbox = draw.textbbox((0, 0), texto_a_dibujar, font=font_ubuntu)
            message_width = bbox[2] - bbox[0]
        except AttributeError:
            bbox = font_ubuntu.getbbox(texto_a_dibujar) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
            message_width = bbox[2] - bbox[0]
    except Exception:
        text_size_message, _ = cv2.getTextSize(texto_a_dibujar, font_message, font_scale_message, thickness_message)
        message_width = text_size_message[0]
    message_x = char_x + (char_new_width - message_width) // 2 - 80
    put_text_safe(screen, texto_a_dibujar, (message_x + 2, message_y + 2),
                  font_message, font_scale_message, (0, 0, 0), thickness_message + 1, bold=True)
    put_text_safe(screen, texto_a_dibujar, (message_x, message_y),
                  font_message, font_scale_message, message_color, thickness_message, bold=True)


def create_end_screen(image_path, message_text, show_confetti, original_image, original_image_pos,
                      view_width, view_height, confetti_system=None,
                      draw_celebration_confetti=True, draw_lingo_bien_title=True):
    """
    Crea una pantalla genérica de fin de juego que puede mostrar cualquier imagen y mensaje.

    Args:
        image_path: Ruta a la imagen del personaje (ej: "images/LingoBien.png" o "images/LingoMal.png")
        message_text: Texto opcional a mostrar debajo de la imagen del personaje
        show_confetti: Boolean para habilitar/deshabilitar confetti
        original_image: Imagen original del juego para mostrar en el fondo
        original_image_pos: Tupla (img_x, img_y, new_width, new_height) con la posición de la imagen original
        view_width, view_height: Dimensiones de la pantalla
        confetti_system: Sistema de confetti (opcional, solo si show_confetti=True)
        draw_celebration_confetti: Si False, no dibuja confetti (útil para raster estático + animación aparte).
        draw_lingo_bien_title: Si False, no dibuja el título «¡Muy Bien!» sobre LingoBien (p. ej. para dibujarlo tras el confeti).

    Returns:
        tuple: (screen, button_salir_x, button_siguiente_x, button_y, button_width, button_height,
                salir_button_bounds, siguiente_button_bounds, char_x, char_y, char_new_width, char_new_height)
    """
    img_x, img_y, new_width, new_height = original_image_pos
    
    # Crear pantalla con opacidad negra
    screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    ratio_col = np.linspace(0.0, 1.0, view_height, dtype=np.float32)[:, np.newaxis]
    screen[:, :, 0] = (255 * (0.8 - 0.3 * ratio_col)).astype(np.uint8)
    screen[:, :, 1] = (200 * (0.5 + 0.3 * ratio_col)).astype(np.uint8)
    screen[:, :, 2] = (255 * (0.3 + 0.4 * ratio_col)).astype(np.uint8)
    
    # Dibujar imagen original con opacidad negra (0.3 para oscurecer más)
    opacity = 0.3
    if len(original_image.shape) == 3 and original_image.shape[2] == 4:
        # Imagen con canal alfa
        alpha_original = original_image[:, :, 3] / 255.0
        img_bgr = original_image[:, :, :3]
        for c in range(3):
            screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                (1 - opacity) * screen[img_y:img_y+new_height, img_x:img_x+new_width, c] +
                opacity * (alpha_original * img_bgr[:, :, c] + (1 - alpha_original) * screen[img_y:img_y+new_height, img_x:img_x+new_width, c])
            )
    else:
        for c in range(3):
            screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                (1 - opacity) * screen[img_y:img_y+new_height, img_x:img_x+new_width, c] +
                opacity * original_image[:, :, c]
            )
    
    # Inicializar variables para la posición de la imagen del personaje
    char_y = 120
    char_new_height = 0
    char_x = 0
    char_new_width = 0
    
    # Cargar y mostrar imagen del personaje
    if os.path.exists(image_path):
        character_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if character_img is not None:
            # Redimensionar imagen del personaje para que quepa bien
            char_height, char_width = character_img.shape[:2]
            max_char_width = int(view_width * 0.6)
            max_char_height = int(view_height * 0.4)
            
            char_aspect = char_width / char_height
            if char_aspect > (max_char_width / max_char_height):
                char_new_width = max_char_width
                char_new_height = int(max_char_width / char_aspect)
            else:
                char_new_height = max_char_height
                char_new_width = int(max_char_height * char_aspect)
            
            character_resized = cv2.resize(character_img, (char_new_width, char_new_height), interpolation=cv2.INTER_AREA)
            
            # Posicionar imagen del personaje centrado (bajado un poco)
            char_x = (view_width - char_new_width) // 2
            char_y = 120
            
            # Dibujar imagen del personaje (manejar transparencia)
            if len(character_resized.shape) == 3 and character_resized.shape[2] == 4:
                alpha_char = character_resized[:, :, 3] / 255.0
                img_char_bgr = character_resized[:, :, :3]
                for c in range(3):
                    screen[char_y:char_y+char_new_height, char_x:char_x+char_new_width, c] = (
                        alpha_char * img_char_bgr[:, :, c] + (1 - alpha_char) * screen[char_y:char_y+char_new_height, char_x:char_x+char_new_width, c]
                    )
            else:
                screen[char_y:char_y+char_new_height, char_x:char_x+char_new_width] = character_resized[:, :, :3]
            
            # Si la imagen es LingoBien, mostrar "¡Muy Bien!" debajo automáticamente
            if "LingoBien" in image_path and message_text is None:
                message_text = "¡Muy Bien!"
    
    # Dibujar mensaje de texto si se proporciona (pero NO para LingoBien, se dibuja después del confetti)
    if message_text and "LingoBien" not in image_path:
        font_message = cv2.FONT_HERSHEY_DUPLEX
        font_scale_message = 1.5
        thickness_message = 3
        text_size_message, baseline = cv2.getTextSize(message_text, font_message, font_scale_message, thickness_message)
        text_width_message = text_size_message[0]
        text_height_message = text_size_message[1] + baseline
        text_x_message = (view_width - text_width_message) // 2
        # Posicionar el mensaje debajo de la imagen del personaje
        text_y_message = 120 + int(view_height * 0.4) + 60
        
        # Dibujar fondo para el mensaje (semi-transparente negro)
        padding = 20
        rect_x1 = max(0, text_x_message - padding)
        rect_y1 = max(0, text_y_message - text_height_message - padding)
        rect_x2 = min(view_width, text_x_message + text_width_message + padding)
        rect_y2 = min(view_height, text_y_message + padding)
        
        # Solo dibujar si las coordenadas son válidas
        if rect_x2 > rect_x1 and rect_y2 > rect_y1:
            # Dibujar fondo semi-transparente (mezclar con el fondo existente)
            overlay = screen.copy()
            cv2.rectangle(overlay, 
                         (rect_x1, rect_y1),
                         (rect_x2, rect_y2),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, screen, 0.3, 0, screen)
            
            # Dibujar borde blanco
            cv2.rectangle(screen,
                         (rect_x1, rect_y1),
                         (rect_x2, rect_y2),
                         (255, 255, 255), 2)
        
        # Dibujar texto del mensaje
        put_text_safe(screen, message_text, (text_x_message + 2, text_y_message + 2),
                     font_message, font_scale_message, (0, 0, 0), thickness_message + 1)  # Sombra negra
        put_text_safe(screen, message_text, (text_x_message, text_y_message),
                     font_message, font_scale_message, (255, 255, 255), thickness_message)  # Texto blanco
    
    # Dibujar dos botones: "Salir" (izquierda) y "Siguiente" (derecha)
    button_width = 250
    button_height = 80
    button_spacing = 30
    center_x = view_width // 2
    button_y = view_height - 180  # Subidos más arriba
    
    # Botón "Salir" (izquierda) - Rojo como en el juego de clasificación
    button_salir_x = center_x - button_width - button_spacing // 2
    
    # Draw "Salir" button using component
    salir_button_bounds = draw_rectangular_button(
        screen,
        button_salir_x, button_y, button_width, button_height,
        "Salir",
        bg_color=(0, 0, 200),  # Red
        border_color=(255, 255, 255),
        border_thickness=3,
        text_color=(255, 255, 255),
        font_scale=1.0,
        bold=True,
        shadow=False
    )
    
    # Botón "Siguiente" (derecha) - Verde como en el juego de clasificación
    button_siguiente_x = center_x + button_spacing // 2
    
    # Draw "Siguiente" button using component
    siguiente_button_bounds = draw_rectangular_button(
        screen,
        button_siguiente_x, button_y, button_width, button_height,
        "Siguiente",
        bg_color=(0, 200, 0),  # Green
        border_color=(255, 255, 255),
        border_thickness=3,
        text_color=(255, 255, 255),
        font_scale=1.0,
        bold=True,
        shadow=False
    )
    
    if show_confetti and confetti_system is not None and draw_celebration_confetti:
        confetti_system.draw(screen)

    if draw_lingo_bien_title and "LingoBien" in image_path and char_new_height > 0:
        _overlay_lingo_bien_title(screen, char_x, char_y, char_new_width, char_new_height)

    return (
        screen,
        button_salir_x,
        button_siguiente_x,
        button_y,
        button_width,
        button_height,
        salir_button_bounds,
        siguiente_button_bounds,
        char_x,
        char_y,
        char_new_width,
        char_new_height,
    )


def draw_close_card(screen, view_width, view_height, elevated=False):
    """
    Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca.
    Similar a la función en niveles_clasificacion.py
    
    Args:
        screen: Pantalla donde dibujar
        view_width: Ancho de la vista
        view_height: Alto de la vista
        elevated: Si True, la card se dibuja elevada (efecto de levantarse)
    """
    # Posición base del lado derecho
    base_card_radius = 50
    card_margin_x = 180
    card_margin_y = 80
    
    # Efecto de elevación si está elevada
    elevation_offset = 0
    scale_factor = 1.0
    shadow_offset_base = 5
    
    if elevated:
        elevation_offset = -15  # Mover hacia arriba
        scale_factor = 1.08  # Aumentar tamaño ligeramente
        shadow_offset_base = 10  # Sombra más grande cuando está elevada
    
    card_radius = int(base_card_radius * scale_factor)
    card_center = (view_width - card_margin_x - int(base_card_radius * scale_factor), 
                   card_margin_y + int(base_card_radius * scale_factor) + elevation_offset)
    
    # Sombra suave (múltiples capas para efecto infantil)
    shadow_offset = int(shadow_offset_base * scale_factor)
    for i in range(3, 0, -1):
        shadow_alpha = i / 3.0 * 0.3
        shadow_color = tuple(int(c * shadow_alpha) for c in (100, 0, 0))
        offset = shadow_offset + (3 - i)
        cv2.circle(screen, 
                  (card_center[0] + offset, card_center[1] + offset), 
                  card_radius, shadow_color, -1)
    
    # Gradiente rojo pastel (simulado con círculos concéntricos)
    # Hacer el color más brillante si está elevada
    color_intensity = 1.15 if elevated else 1.0
    base_red_light = int(100 * color_intensity)
    base_red_medium = int(50 * color_intensity)
    base_red_dark = int(30 * color_intensity)
    # Limitar valores a 255
    base_red_light = min(255, base_red_light)
    base_red_medium = min(255, base_red_medium)
    base_red_dark = min(255, base_red_dark)
    
    # Círculo exterior más claro
    cv2.circle(screen, card_center, card_radius, (base_red_light, base_red_light, 255), -1)  # Rojo pastel claro
    # Círculo interior más intenso
    cv2.circle(screen, card_center, int(card_radius * 0.85), (base_red_medium, base_red_medium, 255), -1)  # Rojo pastel medio
    # Círculo más interno
    cv2.circle(screen, card_center, int(card_radius * 0.7), (base_red_dark, base_red_dark, 255), -1)  # Rojo más intenso
    
    # Borde blanco suave (estilo infantil)
    cv2.circle(screen, card_center, card_radius, (255, 255, 255), 4)
    cv2.circle(screen, card_center, card_radius - 2, (200, 200, 200), 2)
    
    # Dibujar la X blanca en el centro
    x_size = int(card_radius * 0.5)
    thickness = 5
    # Sombra de la X
    cv2.line(screen, 
            (card_center[0] - x_size + 2, card_center[1] - x_size + 2), 
            (card_center[0] + x_size + 2, card_center[1] + x_size + 2), 
            (150, 150, 150), thickness)
    cv2.line(screen, 
            (card_center[0] - x_size + 2, card_center[1] + x_size + 2), 
            (card_center[0] + x_size + 2, card_center[1] - x_size + 2), 
            (150, 150, 150), thickness)
    # X blanca principal
    cv2.line(screen, 
            (card_center[0] - x_size, card_center[1] - x_size), 
            (card_center[0] + x_size, card_center[1] + x_size), 
            (255, 255, 255), thickness)
    cv2.line(screen, 
            (card_center[0] - x_size, card_center[1] + x_size), 
            (card_center[0] + x_size, card_center[1] - x_size), 
            (255, 255, 255), thickness)
    
    return card_center, card_radius


def draw_logo(screen, view_width, view_height):
    """
    Dibuja el logo en la parte superior de la pantalla.
    Similar a la función en main.py
    """
    logo_loaded_local = False
    logo_image_local = None
    
    try:
        # Intentar cargar como PNG primero
        if os.path.exists("images/logo.png"):
            logo_image_local = cv2.imread("images/logo.png", cv2.IMREAD_UNCHANGED)
            if logo_image_local is not None:
                logo_loaded_local = True
        # Intentar cargar SVG usando pygame
        elif os.path.exists("images/logo.svg"):
            try:
                logo_surface = pygame.image.load("images/logo.svg")
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
        put_text_safe(screen, logo_text, (text_x + 3, text_y + 3), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        put_text_safe(screen, logo_text, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), thickness)
        return False


def juego_absurdos_reconocimiento_voz(device, coordenadas, dmax_map, dmin_map, model=None, existing_window_name=None):
    """
    Juego de reconocimiento de voz para absurdos lógicos.
    Muestra una imagen aleatoria y el usuario debe describir qué está mal usando voz.

    Args:
        existing_window_name: Si se indica (ej. "Menú de Juegos"), se reutiliza esa ventana
            en el videobeam. Al salir con X no se cierra la ventana para que el menú
            pueda actualizarla y mostrarse de inmediato.
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('absurdos_visuales.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800
    
    # Resolución del videobeam (segunda pantalla)
    VIDEOBEAM_WIDTH = 1920
    VIDEOBEAM_HEIGHT = 1080
    
    def scale_to_videobeam(image, source_width=1280, source_height=800):
        """Escala una imagen de la resolución fuente a la resolución del videobeam."""
        if image is None or image.size == 0:
            return image
        return cv2.resize(image, (VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Cargar configuración de absurdos
    json_path = "absurdos-visuales/absurdos_logicos/config/absurdos.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        absurdos_list = datos.get("absurdos", [])
        if not absurdos_list:
            print("Error: No se encontraron absurdos en el archivo JSON")
            return
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {json_path} no es un JSON válido")
        return
    
    # Usar ventana existente (videobeam) o crear una nueva
    window_name = existing_window_name if existing_window_name else "Juego de Reconocimiento de Voz"
    use_shared_window = bool(existing_window_name)
    window_exists = False
    if use_shared_window:
        try:
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if prop >= 0:
                window_exists = True
        except Exception:
            pass
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.waitKey(50)

    def _close_window_if_own():
        """Cierra la ventana solo si es propia del juego (no la compartida con el menú)."""
        if not use_shared_window:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    # Variable para controlar si es la primera imagen (solo TTS en la primera)
    primera_imagen = True
    
    # Lista para rastrear los absurdos ya mostrados (usando el nombre de la imagen como identificador único)
    absurdos_mostrados = []
    
    # Bucle principal del juego - continuar hasta que el usuario presione "Volver al Menú"
    while True:
        # Reconfigurar ventana en cada iteración para asegurar tamaño correcto
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        
        # Filtrar absurdos disponibles (excluir los ya mostrados)
        absurdos_disponibles = [
            absurdo for absurdo in absurdos_list 
            if absurdo.get("imagen", "") not in absurdos_mostrados
        ]
        
        # Si todos los absurdos han sido mostrados, reiniciar la lista
        if not absurdos_disponibles:
            print("Todos los absurdos han sido mostrados. Reiniciando lista...")
            absurdos_mostrados = []
            absurdos_disponibles = absurdos_list
        
        # Seleccionar un absurdo aleatorio de los disponibles
        absurdo_actual = random.choice(absurdos_disponibles)
        imagen_nombre = absurdo_actual.get("imagen", "")
        
        # Agregar el absurdo actual a la lista de mostrados
        if imagen_nombre:
            absurdos_mostrados.append(imagen_nombre)
        
        # Construir ruta de imagen
        assets_path = "absurdos-visuales/absurdos_logicos/assets"
        ruta_imagen = os.path.join(assets_path, imagen_nombre)
        
        # Función para normalizar nombres de archivo (remover extensiones, guiones, guiones bajos, espacios, convertir a minúsculas)
        def normalizar_nombre(nombre):
            # Remover extensión
            nombre_sin_ext = os.path.splitext(nombre)[0]
            # Normalizar: convertir a minúsculas, reemplazar guiones y guiones bajos con nada, remover espacios
            nombre_normalizado = nombre_sin_ext.lower().replace("-", "").replace("_", "").replace(" ", "")
            return nombre_normalizado
        
        # Intentar encontrar la imagen con matching flexible
        if not os.path.exists(ruta_imagen):
            # Intentar con guiones en lugar de guiones bajos
            imagen_nombre_alt = imagen_nombre.replace("_", "-")
            ruta_imagen = os.path.join(assets_path, imagen_nombre_alt)
        
        # Si aún no existe, buscar archivos similares usando normalización
        if not os.path.exists(ruta_imagen):
            imagen_normalizada = normalizar_nombre(imagen_nombre)
            mejor_coincidencia = None
            mejor_puntuacion = 0
            
            # Obtener lista de archivos en el directorio
            if os.path.exists(assets_path):
                archivos_disponibles = os.listdir(assets_path)
                for archivo in archivos_disponibles:
                    # Solo procesar archivos de imagen
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        archivo_normalizado = normalizar_nombre(archivo)
                        
                        # Si los nombres normalizados son iguales
                        if imagen_normalizada == archivo_normalizado:
                            mejor_coincidencia = archivo
                            mejor_puntuacion = 100
                            break
                        
                        # Si uno contiene al otro (coincidencia parcial)
                        if imagen_normalizada in archivo_normalizado or archivo_normalizado in imagen_normalizada:
                            # Calcular puntuación basada en la longitud de la coincidencia
                            len_menor = min(len(imagen_normalizada), len(archivo_normalizado))
                            len_mayor = max(len(imagen_normalizada), len(archivo_normalizado))
                            puntuacion = (len_menor / len_mayor) * 100
                            if puntuacion > mejor_puntuacion:
                                mejor_coincidencia = archivo
                                mejor_puntuacion = puntuacion
                        
                        # Calcular similitud por caracteres comunes (método de Jaccard)
                        caracteres_imagen = set(imagen_normalizada)
                        caracteres_archivo = set(archivo_normalizado)
                        caracteres_comunes = caracteres_imagen & caracteres_archivo
                        caracteres_totales = caracteres_imagen | caracteres_archivo
                        
                        if len(caracteres_totales) > 0:
                            similitud_jaccard = len(caracteres_comunes) / len(caracteres_totales) * 100
                            # También considerar la longitud de la subsecuencia común más larga
                            # Extraer palabras clave comunes (palabras de 3+ caracteres)
                            palabras_imagen = [imagen_normalizada[i:i+3] for i in range(len(imagen_normalizada)-2)]
                            palabras_archivo = [archivo_normalizado[i:i+3] for i in range(len(archivo_normalizado)-2)]
                            palabras_comunes = set(palabras_imagen) & set(palabras_archivo)
                            
                            if len(palabras_comunes) > 0:
                                puntuacion_palabras = len(palabras_comunes) / max(len(palabras_imagen), len(palabras_archivo)) * 100
                                puntuacion_final = (similitud_jaccard * 0.6 + puntuacion_palabras * 0.4)
                            else:
                                puntuacion_final = similitud_jaccard
                            
                            if puntuacion_final > mejor_puntuacion and puntuacion_final > 40:  # Al menos 40% de similitud
                                mejor_coincidencia = archivo
                                mejor_puntuacion = puntuacion_final
            
            if mejor_coincidencia:
                ruta_imagen = os.path.join(assets_path, mejor_coincidencia)
                print(f"Imagen encontrada por coincidencia: {mejor_coincidencia} (buscando: {imagen_nombre})")
        
        if not os.path.exists(ruta_imagen):
            print(f"Error: No se encontró la imagen {imagen_nombre} en {assets_path}")
            print(f"Archivos disponibles: {', '.join([f for f in os.listdir(assets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])}")
            continue  # Continuar con el siguiente absurdo en el bucle while True
        
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
            continue  # Continuar con el siguiente absurdo
        
        # Crear pantalla del videobeam
        videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        
        # Crear degradado de colores (mismo estilo que otros juegos)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            videobeam_screen[y, :] = [b, g, r]
        
        # Dibujar logo en la parte superior
        draw_logo(videobeam_screen, view_width, view_height)
        
        # Dibujar botón X (cerrar) en la esquina superior derecha
        close_card_center, close_card_radius = draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
        
        # Redimensionar imagen para que quepa en la pantalla (más pequeña y arriba del botón)
        max_img_width = int(view_width * 0.5)  # Reducido de 0.7 a 0.5
        max_img_height = int(view_height * 0.4)  # Reducido de 0.6 a 0.4
        
        img_height, img_width = imagen.shape[:2]
        aspect_ratio = img_width / img_height
        
        if aspect_ratio > (max_img_width / max_img_height):
            new_width = max_img_width
            new_height = int(max_img_width / aspect_ratio)
        else:
            new_height = max_img_height
            new_width = int(max_img_height * aspect_ratio)
        
        imagen_resized = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calcular posición del botón primero (para posicionar la imagen arriba de él)
        circle_radius = 60
        circle_center_x = view_width // 2
        espacio_para_franja = 100  # Espacio necesario para la franja "Escuchando..." (50 arriba + 50 abajo)
        # Posición del botón: cerca de la parte inferior pero dejando espacio para la franja
        circle_center_y = view_height - espacio_para_franja - circle_radius
        
        # Posicionar la imagen arriba del botón con espacio
        espacio_entre_imagen_y_boton = 40  # Espacio entre la imagen y el botón
        img_y = circle_center_y - circle_radius - espacio_entre_imagen_y_boton - new_height
        # Asegurar que la imagen no se salga por arriba (dejar espacio para el logo)
        logo_height = 120
        logo_y = 30
        espacio_desde_logo = 20  # Espacio mínimo entre logo e imagen
        min_img_y = logo_y + logo_height + espacio_desde_logo
        img_y = max(img_y, min_img_y)  # Asegurar que esté debajo del logo
        
        # Centrar horizontalmente
        img_x = (view_width - new_width) // 2
        
        # Calcular dimensiones de la card (con padding alrededor de la imagen)
        card_padding = 20  # Espacio alrededor de la imagen dentro de la card
        card_width = new_width + (card_padding * 2)
        card_height = new_height + (card_padding * 2)
        card_x = img_x - card_padding
        card_y = img_y - card_padding
        
        # Dibujar sombra simple de la card (sin efectos de relieve, solo sombra)
        shadow_offset = 8
        shadow_color = (40, 40, 40)  # Gris oscuro para la sombra
        cv2.rectangle(
            videobeam_screen,
            (card_x + shadow_offset, card_y + shadow_offset),
            (card_x + card_width + shadow_offset, card_y + card_height + shadow_offset),
            shadow_color,
            -1
        )
        
        # Dibujar card blanca (fondo de la imagen)
        card_color = (255, 255, 255)  # Blanco
        cv2.rectangle(
            videobeam_screen,
            (card_x, card_y),
            (card_x + card_width, card_y + card_height),
            card_color,
            -1
        )
        
        # Dibujar imagen (manejar transparencia si existe)
        if len(imagen_resized.shape) == 3 and imagen_resized.shape[2] == 4:
            # Imagen con canal alfa
            alpha = imagen_resized[:, :, 3] / 255.0
            img_bgr = imagen_resized[:, :, :3]
            for c in range(3):
                videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                    alpha * img_bgr[:, :, c] + (1 - alpha) * videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width, c]
                )
        else:
            videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width] = imagen_resized[:, :, :3]
        
        # Extraer coordenadas para touch detection (necesarias para detectar el botón)
        xw_min = coordenadas["xw_min"]
        xw_max = coordenadas["xw_max"]
        yw_min = coordenadas["yw_min"]
        yw_max = coordenadas["yw_max"]
        xv_min = coordenadas["xv_min"]
        xv_max = coordenadas["xv_max"]
        yv_min = coordenadas["yv_min"]
        yv_max = coordenadas["yv_max"]
        
        # Dibujar card circular en la parte inferior (debajo de la imagen) - SIEMPRE visible
        # La posición del círculo ya fue calculada arriba para posicionar la imagen correctamente
        
        # Draw circular "Hablar" button using component
        hablar_button_bounds = draw_circular_button(
            videobeam_screen,
            circle_center_x, circle_center_y, circle_radius,
            "Hablar",
            bg_color=(0, 200, 0),  # Green
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.3,
            bold=True,
            shadow=True
        )
        
        # Mostrar pantalla con el botón circular (siempre visible)
        videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
        cv2.imshow(window_name, videobeam_screen_scaled)
        cv2.waitKey(50)
        
        # Inicializar TTS solo si es la primera imagen
        tts_finished = threading.Event()
        boton_habilitado = False  # El botón solo se habilita después del TTS
        
        if primera_imagen:
            try:
                engine_tts = pyttsx3.init()
                engine_tts.setProperty('rate', 150)  # Velocidad del habla
                # Intentar configurar voz en español
                voices = engine_tts.getProperty('voices')
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                        engine_tts.setProperty('voice', voice.id)
                        break
            except Exception as e:
                print(f"Advertencia: No se pudo inicializar TTS: {e}")
                engine_tts = None
            
            # Decir instrucción en un hilo separado (solo en la primera imagen)
            def decir_instruccion():
                if engine_tts:
                    try:
                        engine_tts.say("Ahora habla, di qué está mal")
                        engine_tts.runAndWait()
                    except Exception as e:
                        print(f"Error al decir instrucción: {e}")
                    finally:
                        tts_finished.set()
                else:
                    tts_finished.set()
            
            tts_thread = threading.Thread(target=decir_instruccion, daemon=True)
            tts_thread.start()
            
            # Esperar a que termine de hablar el TTS, actualizando la pantalla mientras tanto
            while not tts_finished.is_set():
                videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                cv2.imshow(window_name, videobeam_screen_scaled)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                cv2.waitKey(10)
                tts_finished.wait(timeout=0.1)  # Verificar cada 100ms
            
            # Esperar un poco más para asegurar que el audio se haya limpiado
            time.sleep(0.5)
            
            # Marcar que ya no es la primera imagen y habilitar el botón
            primera_imagen = False
            boton_habilitado = True
            
            # Actualizar pantalla para mostrar que el botón está habilitado
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            cv2.imshow(window_name, videobeam_screen_scaled)
            cv2.waitKey(10)
        else:
            # Si no es la primera imagen, el botón está habilitado inmediatamente
            boton_habilitado = True
        
        # Inicializar reconocimiento de voz (pero no escuchar todavía)
        try:
            recognizer = sr.Recognizer()
            # Configurar reconocedor para mejor precisión y tiempos más acotados
            recognizer.energy_threshold = 300  # Umbral de energía inicial
            recognizer.dynamic_energy_threshold = True  # Ajustar dinámicamente
            # Requerir 5 segundos de silencio sostenido antes de cortar la frase
            recognizer.pause_threshold = 5.0
            recognizer.phrase_threshold = 0.3
            # Mantener un poco de audio sin voz para no recortar el final
            recognizer.non_speaking_duration = 0.5
            recognizer.operation_timeout = None  # Sin timeout en operaciones
            
            # Índice PyAudio (no es el “puerto USB” físico). Cambiar con variable de entorno LINGO_MIC_DEVICE_INDEX.
            mic_index = int(os.environ.get("LINGO_MIC_DEVICE_INDEX", "2"))
            try:
                microphone = sr.Microphone(device_index=mic_index)
                print(f"Usando micrófono (device_index={mic_index}). Para otro mic: set LINGO_MIC_DEVICE_INDEX=N (ver índices con python test_microphone_pyaudio.py)")
            except Exception as e:
                print(f"Error al inicializar micrófono (device_index={mic_index}): {e}")
                try:
                    microphone = sr.Microphone()
                    print("Usando micrófono por defecto del sistema (sin índice).")
                except Exception as e2:
                    print(f"No se pudo abrir micrófono por defecto: {e2}")
                    _close_window_if_own()
                    continue
        except OSError as e:
            if "PyAudio" in str(e) or "pyaudio" in str(e).lower():
                print("Error: PyAudio no está instalado. Instalando...")
                print("Por favor ejecuta: uv pip install pyaudio")
                print("O si eso no funciona, instala desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            else:
                print(f"Error al inicializar reconocimiento de voz: {e}")
            _close_window_if_own()
            continue
        except Exception as e:
            print(f"Error al inicializar reconocimiento de voz: {e}")
            print("Asegúrate de que PyAudio esté instalado: uv pip install pyaudio")
            _close_window_if_own()
            continue
        
        # Función para detectar si se tocó el botón circular
        def detectar_boton_circular_tocado(cx_roi, cy_roi):
            # cx_roi y cy_roi son coordenadas relativas a depth_roi
            x_viewport, y_viewport = map_depth_roi_to_viewport(
                cx_roi, cy_roi, coordenadas, view_width=view_width, view_height=view_height
            )
            
            # Verificar si el toque está dentro del círculo usando component helper
            if is_point_in_circular_button(x_viewport, y_viewport, hablar_button_bounds):
                return True
            return False
        
        # Definir text_y para evitar errores (usado más adelante)
        text_y = view_height - 50
        
        # Cargar modelo ANTES del bucle de intentos (solo una vez)
        modelo_cargado_correctamente = True
        if model is None:
            try:
                # Usar modelo multilingüe que soporta español
                print("Cargando modelo de sentence-transformers...")
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ Modelo de sentence-transformers cargado correctamente")
            except Exception as e:
                print(f"Error al cargar modelo de sentence-transformers: {e}")
                print("Intentando cargar modelo nuevamente...")
                try:
                    # Reintentar con manejo de errores de red
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            print("✓ Modelo de sentence-transformers cargado correctamente")
                            break
                        except Exception as e2:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 2  # Esperar 2, 4, 6 segundos
                                print(f"Intento {attempt + 1} fallido. Reintentando en {wait_time} segundos...")
                                time.sleep(wait_time)
                            else:
                                print(f"Error al descargar/cargar modelo después de {max_retries} intentos: {e2}")
                                print("El juego continuará pero no podrá evaluar respuestas correctamente.")
                                _close_window_if_own()
                                modelo_cargado_correctamente = False
                except Exception as e3:
                    print(f"Error crítico al cargar modelo: {e3}")
                    _close_window_if_own()
                    modelo_cargado_correctamente = False
        else:
            print("✓ Usando modelo de sentence-transformers pre-cargado")
        
        # Si no se pudo cargar el modelo, continuar con el siguiente absurdo
        if not modelo_cargado_correctamente:
            continue
        
        # Cargar respuestas correctas ANTES del bucle (solo una vez por imagen)
        respuestas_json_path = "absurdos-visuales/absurdos_logicos/config/respuestas_correctas.json"
        respuestas_correctas = []
        try:
            with open(respuestas_json_path, "r", encoding="utf-8") as f:
                datos_respuestas = json.load(f)
                # Buscar respuestas para el absurdo actual
                absurdo_id = absurdo_actual.get("id")
                imagen_actual = absurdo_actual.get("imagen", "")
                for respuesta_item in datos_respuestas.get("respuestas", []):
                    if respuesta_item.get("id") == absurdo_id or respuesta_item.get("imagen") == imagen_actual:
                        respuestas_correctas = respuesta_item.get("respuestas_correctas", [])
                        break
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {respuestas_json_path}")
            respuestas_correctas = []
        except Exception as e:
            print(f"Error al cargar respuestas correctas: {e}")
            respuestas_correctas = []
        
        # Bucle de intentos para esta imagen (máximo 3 intentos)
        intentos_restantes = 3
        respuesta_correcta = False
        
        while intentos_restantes > 0 and not respuesta_correcta:
            # Iniciar streams para detección de toques
            rgb_stream = device.create_color_stream()
            depth_stream = device.create_depth_stream()
            rgb_stream.start()
            depth_stream.start()
            
            # Esperar a que se toque el botón circular
            boton_presionado = False
            button_pressed_flag = False
            audio = None
            
            print("Esperando a que presiones el botón circular para empezar a escuchar...")
            
            print("Listo. Presiona el botón para empezar a escuchar...")
            
            while True:
                frame = rgb_stream.read_frame()
                depth_frame = depth_stream.read_frame()
                
                if frame is None or depth_frame is None:
                    continue
                
                rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                depth_data = cv2.flip(depth_data, 1)
                depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
                
                # Crear máscara de toques
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Verificar si hay toques en el botón (solo si está habilitado)
                hay_toque_en_boton = False
                hay_toque_en_close = False
                current_screen = videobeam_screen.copy()
                
                # Primero verificar si hay toque en el botón X antes de dibujarlo
                # (para mostrar el efecto de elevación)
                close_card_touched = False
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])

                            x_viewport_touch, y_viewport_touch = map_depth_roi_to_viewport(
                                cx, cy, coordenadas, view_width=view_width, view_height=view_height
                            )

                            # Calcular posición del botón X para verificar toque
                            base_card_radius = 50
                            card_margin_x = 180
                            card_margin_y = 80
                            close_card_center_temp = (view_width - card_margin_x - base_card_radius, 
                                                     card_margin_y + base_card_radius)
                            
                            # Verificar si se tocó el botón X (cerrar)
                            distance_to_close = np.sqrt((x_viewport_touch - close_card_center_temp[0])**2 + 
                                                       (y_viewport_touch - close_card_center_temp[1])**2)
                            if distance_to_close <= base_card_radius:
                                close_card_touched = True
                                break
                
                # Dibujar botón X (cerrar) en la esquina superior derecha con efecto de elevación si se toca
                close_card_center, close_card_radius = draw_close_card(current_screen, view_width, view_height, elevated=close_card_touched)
                
                # SIEMPRE dibujar el botón en la pantalla (habilitado o no)
                if boton_habilitado:
                    # Botón habilitado - color verde normal
                    draw_circular_button(
                        current_screen,
                        circle_center_x, circle_center_y, circle_radius,
                        "Hablar",
                        bg_color=(0, 200, 0),  # Green
                        border_color=(255, 255, 255),
                        border_thickness=3,
                        text_color=(255, 255, 255),
                        font_scale=1.3,
                        bold=True,
                        shadow=True
                    )
                else:
                    # Botón no habilitado - color gris
                    draw_circular_button(
                        current_screen,
                        circle_center_x, circle_center_y, circle_radius,
                        "Hablar",
                        bg_color=(100, 100, 100),  # Gray
                        border_color=(255, 255, 255),
                        border_thickness=3,
                        text_color=(255, 255, 255),
                        font_scale=1.3,
                        bold=True,
                        shadow=True
                    )
                
                # Procesar toques (botón circular y botón X)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])

                            x_viewport_touch, y_viewport_touch = map_depth_roi_to_viewport(
                                cx, cy, coordenadas, view_width=view_width, view_height=view_height
                            )

                            # Verificar si se tocó el botón X (cerrar)
                            # Usar la posición base del botón (sin elevación) para la detección
                            base_card_radius = 50
                            card_margin_x = 180
                            card_margin_y = 80
                            close_card_center_base = (view_width - card_margin_x - base_card_radius, 
                                                     card_margin_y + base_card_radius)
                            
                            distance_to_close = np.sqrt((x_viewport_touch - close_card_center_base[0])**2 + 
                                                       (y_viewport_touch - close_card_center_base[1])**2)
                            if distance_to_close <= base_card_radius:
                                hay_toque_en_close = True
                                # Mostrar efecto de elevación y luego salir
                                print("Botón X (cerrar) presionado. Volviendo al menú principal...")
                                
                                # Redibujar con efecto de elevación (usar current_screen que ya tiene todos los elementos)
                                current_screen_elevated = current_screen.copy()
                                # Redibujar solo el botón X con efecto de elevación
                                draw_close_card(current_screen_elevated, view_width, view_height, elevated=True)
                                
                                # Mostrar el efecto de elevación brevemente
                                videobeam_screen_scaled_elevated = scale_to_videobeam(current_screen_elevated)
                                cv2.imshow(window_name, videobeam_screen_scaled_elevated)
                                cv2.waitKey(200)  # Mostrar el efecto por 200ms
                                
                                # Detener streams
                                try:
                                    rgb_stream.stop()
                                    depth_stream.stop()
                                except:
                                    pass
                                # No cerrar ventana si es la del menú (el menú la actualizará en el videobeam)
                                _close_window_if_own()
                                return  # Volver al menú principal
                            
                            # Verificar si se tocó el botón circular (solo si está habilitado)
                            if boton_habilitado and detectar_boton_circular_tocado(cx, cy):
                                hay_toque_en_boton = True
                                
                                if not button_pressed_flag:
                                        # Botón recién presionado - activar micrófono
                                        button_pressed_flag = True
                                        print("Botón presionado. Activando micrófono...")
                                        
                                        # Cambiar color del botón cuando se presiona (verde más oscuro)
                                        draw_circular_button(
                                            current_screen,
                                            circle_center_x, circle_center_y, circle_radius,
                                            "Hablar",
                                            bg_color=(0, 150, 0),  # Darker green
                                            border_color=(255, 255, 255),
                                            border_thickness=3,
                                            text_color=(255, 255, 255),
                                            font_scale=1.3,
                                            bold=True,
                                            shadow=True
                                        )
                                        
                                        # Mostrar solo el estado del botón presionado aquí.
                                        # El mensaje "Escuchando..." se mostrará más tarde,
                                        # cuando el micrófono esté realmente listo para grabar.
                                        videobeam_screen_scaled = scale_to_videobeam(current_screen)
                                        cv2.imshow(window_name, videobeam_screen_scaled)
                                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                        cv2.waitKey(200)
                                        
                                        # Detener streams antes de escuchar
                                        rgb_stream.stop()
                                        depth_stream.stop()
                                        
                                        # Ahora escuchar audio (no necesita mantener presionado)
                                        boton_presionado = True
                                        break
                
                # Si el botón fue presionado, salir del bucle de detección y empezar a escuchar
                if boton_presionado:
                    break
                
                # Mostrar pantalla (usar current_screen que tiene el estado actual del botón)
                videobeam_screen_scaled = scale_to_videobeam(current_screen)
                cv2.imshow(window_name, videobeam_screen_scaled)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                cv2.waitKey(10)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rgb_stream.stop()
                    depth_stream.stop()
                    _close_window_if_own()
                    return
            
            # Si el botón fue presionado, ahora sí escuchar audio
            audio = None
            
            if boton_presionado:
                # Ajustar para ruido ambiente rápidamente antes de escuchar
                print("Ajustando para ruido ambiente antes de escuchar...")
                try:
                    with microphone as source:
                        print("Calibrando micrófono (1 segundo)...")
                        recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Calibración
                        print(f"✓ Umbral de energía ajustado: {recognizer.energy_threshold}")
                except Exception as e:
                    print(f"⚠ Error al ajustar ruido ambiente: {e}")
                    # Continuar de todas formas
                
                # En este punto el micrófono ya está calibrado.
                # Ahora sí mostramos la franja "Escuchando..." en pantalla.
                listening_screen = videobeam_screen.copy()
                draw_circular_button(
                    listening_screen,
                    circle_center_x, circle_center_y, circle_radius,
                    "Hablar",
                    bg_color=(0, 150, 0),  # Darker green
                    border_color=(255, 255, 255),
                    border_thickness=3,
                    text_color=(255, 255, 255),
                    font_scale=1.3,
                    bold=True,
                    shadow=True
                )
                mensaje_escuchando = "Escuchando..."
                font_escuchando = cv2.FONT_HERSHEY_DUPLEX
                font_scale_escuchando = 0.9
                thickness_escuchando = 2
                text_size_escuchando, _ = cv2.getTextSize(
                    mensaje_escuchando,
                    font_escuchando,
                    font_scale_escuchando,
                    thickness_escuchando
                )
                text_x_escuchando = (view_width - text_size_escuchando[0]) // 2
                text_y_escuchando = circle_center_y + circle_radius + 50  # Debajo del botón
                color_fondo_escuchando = (200, 100, 0)  # Azul oscuro (en BGR)
                franja_y_inicio = text_y_escuchando - 25
                franja_y_fin = view_height - 1
                cv2.rectangle(
                    listening_screen,
                    (0, franja_y_inicio),
                    (view_width, franja_y_fin),
                    color_fondo_escuchando,
                    -1
                )
                cv2.rectangle(
                    listening_screen,
                    (0, franja_y_inicio),
                    (view_width, franja_y_fin),
                    (255, 255, 255),
                    2
                )
                put_text_safe(
                    listening_screen,
                    mensaje_escuchando,
                    (text_x_escuchando + 2, text_y_escuchando + 2),
                    font_escuchando,
                    font_scale_escuchando,
                    (0, 0, 0),
                    thickness_escuchando + 1
                )
                put_text_safe(
                    listening_screen,
                    mensaje_escuchando,
                    (text_x_escuchando, text_y_escuchando),
                    font_escuchando,
                    font_scale_escuchando,
                    (255, 255, 255),
                    thickness_escuchando
                )
                videobeam_screen_scaled = scale_to_videobeam(listening_screen)
                cv2.imshow(window_name, videobeam_screen_scaled)
                # No volvemos a tocar aquí el modo de ventana para evitar
                # recreaciones o parpadeos; ya está en fullscreen desde antes.
                cv2.waitKey(200)
                
                # Escuchar audio con detección de silencio
                print("=" * 50)
                print("INICIANDO CAPTURA DE AUDIO")
                print("=" * 50)
                print("Esperando audio (timeout: 7 segundos, límite de frase: 7 segundos)...")
                print("Habla ahora...")
                
                try:
                    with microphone as source:
                        # Escuchar hasta que detecte que terminó de hablar
                        audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)
                        if audio:
                            duracion = len(audio.frame_data) / audio.sample_rate
                            print(f"✓ Audio capturado exitosamente!")
                            print(f"  Duración: {duracion:.2f} segundos")
                            print(f"  Tamaño de datos: {len(audio.frame_data)} bytes")
                            print(f"  Sample rate: {audio.sample_rate} Hz")
                        else:
                            print("✗ No se capturó audio (audio es None)")
                            audio = None
                except sr.WaitTimeoutError:
                    print("✗ Tiempo de espera agotado. No se detectó ningún audio.")
                    print("  Verifica que:")
                    print("  - El micrófono esté funcionando")
                    print("  - El micrófono no esté silenciado")
                    print("  - Estés hablando lo suficientemente fuerte")
                    audio = None
                except Exception as e:
                    print(f"✗ Error al escuchar el micrófono: {e}")
                    print(f"  Tipo de error: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    audio = None
            else:
                # Si no se presionó el botón, continuar con el siguiente intento
                print("Botón no presionado. Intentando de nuevo...")
                intentos_restantes -= 1
                logger.debug(f"Intento decrementado (botón no presionado). Intentos restantes: {intentos_restantes}, Imagen: {imagen_nombre}")
                continue
        
            if not boton_presionado or audio is None:
                # Si no se presionó el botón o no se capturó audio, continuar con el siguiente intento
                print("No se capturó audio. Intentando de nuevo...")
                intentos_restantes -= 1
                logger.debug(f"Intento decrementado (no se capturó audio). Intentos restantes: {intentos_restantes}, Imagen: {imagen_nombre}")
                continue
        
            # Reconocer el audio
            if audio is not None:
                print("Procesando audio con Google Speech Recognition...")
                try:
                    # Usar configuración mejorada para mejor precisión
                    texto_reconocido = recognizer.recognize_google(
                        audio, 
                        language="es-ES",
                        show_all=False  # Solo devolver el mejor resultado
                    )
                    print(f"Texto reconocido: {texto_reconocido}")
                except sr.UnknownValueError:
                    print("No se pudo entender el audio. El audio podría estar muy bajo o no contener palabras claras.")
                    texto_reconocido = None
                except sr.RequestError as e:
                    print(f"Error al conectar con el servicio de reconocimiento: {e}")
                    print("Verifica tu conexión a internet.")
                    texto_reconocido = None
                except Exception as e:
                    print(f"Error inesperado en reconocimiento: {e}")
                    import traceback
                    traceback.print_exc()
                    texto_reconocido = None
            else:
                print("No se capturó audio. No se puede transcribir.")
                texto_reconocido = None
            
            # Evaluar respuesta usando sentence-transformers (DENTRO del bucle de intentos)
            es_correcta = False
            mensaje_resultado = ""
            SIMILARITY_THRESHOLD = 0.7
            
            if texto_reconocido and model:
                if respuestas_correctas:
                    try:
                        # Crear embeddings para el texto del usuario
                        texto_usuario_embedding = model.encode([texto_reconocido])[0]  # Obtener el vector único
                        
                        # Crear embeddings para todas las respuestas correctas
                        respuestas_embeddings = model.encode(respuestas_correctas)
                        
                        # Calcular similitud coseno entre el texto del usuario y cada respuesta correcta
                        # Usar numpy para calcular cosine similarity
                        def cosine_similarity_numpy(vec1, vec2_array):
                            """Calcula la similitud coseno entre un vector y un array de vectores usando numpy"""
                            # vec1: vector 1D, vec2_array: array 2D de vectores
                            dot_products = np.dot(vec2_array, vec1)
                            norm1 = np.linalg.norm(vec1)
                            norms2 = np.linalg.norm(vec2_array, axis=1)
                            return dot_products / (norm1 * norms2)
                        
                        similarities = cosine_similarity_numpy(texto_usuario_embedding, respuestas_embeddings)
                        
                        # Encontrar la similitud máxima
                        max_similarity = float(np.max(similarities))
                        
                        print(f"Similitud máxima encontrada: {max_similarity:.3f}")
                        
                        # Si la similitud máxima supera el umbral, la respuesta es correcta
                        if max_similarity >= SIMILARITY_THRESHOLD:
                            es_correcta = True
                            mensaje_resultado = "¡Correcto!"
                        else:
                            es_correcta = False
                            mensaje_resultado = "Inténtalo de nuevo"
                    except Exception as e:
                        print(f"Error al evaluar respuesta con sentence-transformers: {e}")
                        es_correcta = False
                        mensaje_resultado = "Error al evaluar"
                else:
                    print("Advertencia: No se encontraron respuestas correctas para comparar")
                    es_correcta = False
                    mensaje_resultado = "Inténtalo de nuevo"
            else:
                if not texto_reconocido:
                    mensaje_resultado = "No se detectó audio"
                else:
                    mensaje_resultado = "Error al cargar modelo"
            
            # Mostrar resultado (DENTRO del bucle de intentos)
            # TODOS los mensajes de feedback se muestran en el área inferior (misma posición que "Escuchando...")
            feedback_y_position = circle_center_y + circle_radius + 50  # Misma posición que "Escuchando..."
            
            # Determinar color de fondo según el tipo de feedback
            if es_correcta:
                # Verde para feedback positivo/correcto
                color_fondo = (0, 200, 0)  # Verde oscuro para el fondo
            elif "Error" in mensaje_resultado or "Error al cargar" in mensaje_resultado:
                # Naranja para errores del sistema
                color_fondo = (0, 140, 255)  # Naranja oscuro para el fondo
            elif "No se detectó" in mensaje_resultado:
                # Amarillo para advertencias
                color_fondo = (0, 200, 255)  # Amarillo oscuro para el fondo
            else:
                # Rojo para respuestas incorrectas
                color_fondo = (0, 0, 200)  # Rojo oscuro para el fondo
            
            font_resultado = cv2.FONT_HERSHEY_DUPLEX
            font_scale_resultado = 0.9
            thickness_resultado = 2
            
            text_size_resultado, _ = cv2.getTextSize(mensaje_resultado, font_resultado, font_scale_resultado, thickness_resultado)
            text_x_resultado = (view_width - text_size_resultado[0]) // 2
            text_y_resultado = feedback_y_position
            
            # Dibujar franja de fondo con color según el tipo de feedback (siempre en la parte inferior)
            franja_y_inicio = text_y_resultado - 25
            franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         color_fondo, -1)  # Fondo con color según feedback
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         (255, 255, 255), 2)  # Borde blanco
            
            # Texto siempre en blanco (usar función segura para caracteres acentuados)
            put_text_safe(videobeam_screen, mensaje_resultado, (text_x_resultado + 2, text_y_resultado + 2), 
                       font_resultado, font_scale_resultado, (0, 0, 0), thickness_resultado + 1)  # Sombra negra
            put_text_safe(videobeam_screen, mensaje_resultado, (text_x_resultado, text_y_resultado), 
                       font_resultado, font_scale_resultado, (255, 255, 255), thickness_resultado)  # Texto blanco
            
            # Dibujar botón X (cerrar) en la esquina superior derecha
            draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
            
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            cv2.imshow(window_name, videobeam_screen_scaled)
            # Asegurar que la ventana esté configurada correctamente
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
            cv2.waitKey(10)
            
            # Reproducir sonido de retroalimentación
            pygame.mixer.init()
            try:
                if es_correcta:
                    sonido_path = "sounds/correct.mp3"
                    if not os.path.exists(sonido_path):
                        sonido_path = "sounds/correcto.wav"
                else:
                    sonido_path = "sounds/incorrect.mp3"
                    if not os.path.exists(sonido_path):
                        sonido_path = "sounds/incorrecto.mp3"
                
                if os.path.exists(sonido_path):
                    try:
                        sonido = pygame.mixer.Sound(sonido_path)
                        sonido.play()
                        time.sleep(1.5)  # Esperar a que termine el sonido
                    except Exception as e:
                        print(f"Error al reproducir sonido: {e}")
            except Exception as e:
                print(f"Error al inicializar mixer: {e}")
            
            # Si la respuesta es correcta, mostrar pantalla de éxito
            if es_correcta:
                respuesta_correcta = True
                # Detener streams antes de mostrar pantalla de éxito
                try:
                    rgb_stream.stop()
                    depth_stream.stop()
                except:
                    pass
                
                # Extraer coordenadas para touch detection
                xw_min = coordenadas["xw_min"]
                xw_max = coordenadas["xw_max"]
                yw_min = coordenadas["yw_min"]
                yw_max = coordenadas["yw_max"]
                xv_min = coordenadas["xv_min"]
                xv_max = coordenadas["xv_max"]
                yv_min = coordenadas["yv_min"]
                yv_max = coordenadas["yv_max"]
                
                # Initialize confetti system with multiple bursts
                confetti_system = ConfettiSystem(view_width, view_height, num_particles=120)
                confetti_system.start(multiple_bursts=True, num_burst_points=3)
                
                # Raster estático una vez; en el bucle solo copia + confetti + título (evita reconstruir toda la pantalla).
                (
                    success_base,
                    button_salir_x,
                    button_siguiente_x,
                    button_y,
                    button_width,
                    button_height,
                    salir_button_bounds,
                    siguiente_button_bounds,
                    lingo_char_x,
                    lingo_char_y,
                    lingo_char_w,
                    lingo_char_h,
                ) = create_end_screen(
                    image_path="images/LingoBien.png",
                    message_text=None,
                    show_confetti=False,
                    original_image=imagen_resized,
                    original_image_pos=(img_x, img_y, new_width, new_height),
                    view_width=view_width,
                    view_height=view_height,
                    confetti_system=confetti_system,
                    draw_celebration_confetti=False,
                    draw_lingo_bien_title=False,
                )
                success_work = np.empty_like(success_base)
                np.copyto(success_work, success_base)
                confetti_system.draw(success_work)
                _overlay_lingo_bien_title(success_work, lingo_char_x, lingo_char_y, lingo_char_w, lingo_char_h)

                # Mostrar pantalla de éxito
                success_screen_scaled = scale_to_videobeam(success_work)
                cv2.imshow(window_name, success_screen_scaled)
                cv2.waitKey(10)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                
                # Iniciar streams para detección de toques
                rgb_stream = device.create_color_stream()
                depth_stream = device.create_depth_stream()
                rgb_stream.start()
                depth_stream.start()
                
                # Función para detectar toque en los botones
                def detectar_boton_tocado(cx_roi, cy_roi):
                    # cx_roi y cy_roi son coordenadas relativas a depth_roi
                    x_viewport, y_viewport = map_depth_roi_to_viewport(
                        cx_roi, cy_roi, coordenadas, view_width=view_width, view_height=view_height
                    )
                    
                    # Verificar botón "Salir" (izquierda)
                    if is_point_in_rectangular_button(x_viewport, y_viewport, salir_button_bounds):
                        return "menu"
                    # Verificar botón "Siguiente" (derecha)
                    elif is_point_in_rectangular_button(x_viewport, y_viewport, siguiente_button_bounds):
                        return "siguiente"
                    return None
                
                # Bucle de detección de toques
                button_pressed = False
                accion_seleccionada = None  # "siguiente", "menu", o None
                salir_bucle = False
                try:
                    while not salir_bucle:
                        confetti_system.update()
                        np.copyto(success_work, success_base)
                        confetti_system.draw(success_work)
                        _overlay_lingo_bien_title(success_work, lingo_char_x, lingo_char_y, lingo_char_w, lingo_char_h)

                        frame = rgb_stream.read_frame()
                        depth_frame = depth_stream.read_frame()
                        
                        if frame is None or depth_frame is None:
                            continue
                        
                        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                        depth_data = cv2.flip(depth_data, 1)
                        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
                        
                        # Crear máscara de toques
                        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                        kernel = np.ones((3, 3), np.uint8)
                        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Verificar si hay toques en esta iteración
                        hay_toque = False
                        boton_actual_tocado = None
                        
                        # Procesar toques
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 50:
                                M = cv2.moments(contour)
                                if M['m00'] != 0:
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    
                                    # Verificar si se tocó algún botón
                                    boton_tocado = detectar_boton_tocado(cx, cy)
                                    if boton_tocado:
                                        hay_toque = True
                                        boton_actual_tocado = boton_tocado
                                        
                                        # Solo procesar si no se había presionado antes
                                        if not button_pressed:
                                            button_pressed = True
                                            
                                            # Animación de botón presionado
                                            temp_screen = success_work.copy()
                                            
                                            if boton_tocado == "siguiente":
                                                print("Botón 'Siguiente' presionado")
                                                draw_rectangular_button(
                                                    temp_screen,
                                                    button_siguiente_x, button_y, button_width, button_height,
                                                    "Siguiente",
                                                    bg_color=(0, 150, 0),  # Darker green for pressed state
                                                    border_color=(255, 255, 255),
                                                    border_thickness=3,
                                                    text_color=(255, 255, 255),
                                                    font_scale=1.0,
                                                    bold=True,
                                                    shadow=False
                                                )
                                            elif boton_tocado == "menu":
                                                print("Botón 'Salir' presionado")
                                                draw_rectangular_button(
                                                    temp_screen,
                                                    button_salir_x, button_y, button_width, button_height,
                                                    "Salir",
                                                    bg_color=(0, 0, 150),  # Darker red for pressed state
                                                    border_color=(255, 255, 255),
                                                    border_thickness=3,
                                                    text_color=(255, 255, 255),
                                                    font_scale=1.0,
                                                    bold=True,
                                                    shadow=False
                                                )
                                            
                                            temp_screen_scaled = scale_to_videobeam(temp_screen)
                                            cv2.imshow(window_name, temp_screen_scaled)
                                            cv2.waitKey(200)
                                            
                                            # Detener streams y salir del bucle
                                            rgb_stream.stop()
                                            depth_stream.stop()
                                            
                                            if boton_tocado == "menu":
                                                # Volver al menú principal
                                                accion_seleccionada = "menu"
                                                _close_window_if_own()
                                                salir_bucle = True
                                                break
                                            elif boton_tocado == "siguiente":
                                                # Continuar con siguiente absurdo
                                                accion_seleccionada = "siguiente"
                                                _close_window_if_own()
                                                salir_bucle = True
                                                break
                        
                        # Resetear flag si no hay toque o si el toque no está en ningún botón
                        if button_pressed and (not hay_toque or boton_actual_tocado is None):
                            button_pressed = False
                        
                        success_screen_scaled = scale_to_videobeam(success_work)
                        cv2.imshow(window_name, success_screen_scaled)
                        # Asegurar que la ventana esté configurada correctamente en cada frame
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                        cv2.waitKey(10)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            rgb_stream.stop()
                            depth_stream.stop()
                            _close_window_if_own()
                            return
                finally:
                    # Asegurar que los streams se detengan
                    try:
                        rgb_stream.stop()
                        depth_stream.stop()
                    except:
                        pass
                
                # Verificar qué acción se seleccionó (después del bloque try-finally, dentro del if es_correcta)
                if accion_seleccionada == "menu":
                    _close_window_if_own()
                    return  # Volver al menú principal - salir completamente de la función
                elif accion_seleccionada == "siguiente":
                    break  # Salir del bucle de intentos y continuar con siguiente absurdo
            else:
                # Si la respuesta es incorrecta, manejar intentos
                intentos_restantes -= 1
                logger.debug(f"Intento decrementado (respuesta incorrecta). Intentos restantes: {intentos_restantes}, Imagen: {imagen_nombre}")
                
                # Detener streams antes de mostrar mensaje
                try:
                    rgb_stream.stop()
                    depth_stream.stop()
                except:
                    pass
                
                # Mostrar mensaje de intentos restantes (en la misma posición que "Escuchando...")
                if intentos_restantes > 0:
                    mensaje_intentos = f"Inténtalo de nuevo. Intentos restantes: {intentos_restantes}"
                    font_intentos = cv2.FONT_HERSHEY_DUPLEX
                    font_scale_intentos = 0.9
                    thickness_intentos = 2
                    text_size_intentos, _ = cv2.getTextSize(mensaje_intentos, font_intentos, font_scale_intentos, thickness_intentos)
                    text_x_intentos = (view_width - text_size_intentos[0]) // 2
                    text_y_intentos = circle_center_y + circle_radius + 50  # Misma posición que "Escuchando..."
                    
                    # Color naranja oscuro para el fondo de mensajes de reintento
                    color_fondo_intentos = (0, 140, 255)  # Naranja oscuro
                    
                    # Dibujar franja de fondo con color naranja (siempre en la parte inferior)
                    franja_y_inicio = text_y_intentos - 25
                    franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
                    cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                 color_fondo_intentos, -1)  # Fondo naranja
                    cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                 (255, 255, 255), 2)  # Borde blanco
                    
                    # Texto siempre en blanco (usar función segura para caracteres acentuados)
                    put_text_safe(videobeam_screen, mensaje_intentos, (text_x_intentos + 2, text_y_intentos + 2), 
                               font_intentos, font_scale_intentos, (0, 0, 0), thickness_intentos + 1)  # Sombra negra
                    put_text_safe(videobeam_screen, mensaje_intentos, (text_x_intentos, text_y_intentos), 
                               font_intentos, font_scale_intentos, (255, 255, 255), thickness_intentos)  # Texto blanco
                    
                    # Dibujar botón X (cerrar) en la esquina superior derecha
                    draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
                    
                    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                    cv2.imshow(window_name, videobeam_screen_scaled)
                    cv2.waitKey(10)
                    
                    # Esperar un momento antes de permitir otro intento
                    time.sleep(2)
                    
                    # Continuar con el siguiente intento (volver al inicio del bucle para la misma imagen)
                    # Reiniciar botón habilitado para el siguiente intento
                    boton_habilitado = True
                else:
                    # Se agotaron los intentos, salir del bucle para cambiar de imagen
                    logger.warning(
                        f"Usuario agotó todos los intentos. "
                        f"Imagen: {imagen_nombre}, "
                        f"Intentos restantes: {intentos_restantes}, "
                        f"Respuesta correcta: {respuesta_correcta}"
                    )
                    
                    # Detener cualquier stream que pueda estar activo
                    try:
                        if 'rgb_stream' in locals():
                            rgb_stream.stop()
                    except:
                        pass
                    try:
                        if 'depth_stream' in locals():
                            depth_stream.stop()
                    except:
                        pass
                    
                    # Salir del bucle de intentos para continuar con el siguiente absurdo
                    # (igual que cuando la respuesta es correcta)
                    break
            continue

