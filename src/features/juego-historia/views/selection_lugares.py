"""
Selection lugares view.
"""
import os
import sys
import cv2
import numpy as np
import pygame
import time

_views_dir = os.path.dirname(os.path.abspath(__file__))
_feature_dir = os.path.dirname(_views_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_feature_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.font_utils import put_text_ubuntu

from .helpers import historia_tts_speak
from .vista_final import mostrar_vista_final


def mostrar_seleccion_lugares(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de lugares con 6 cards.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        list, "MENU" o None: Lista de lugares seleccionados, "MENU" si se debe volver al menú, o None si se canceló
    """
    # Extraer coordenadas
    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]
    
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
        scaled_image = cv2.resize(image, (VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return scaled_image
    
    # Crear fondo
    lugares_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        lugares_screen[y, :] = [b, g, r]
    
    # Dibujar el logo (un poco más pequeño - 95% del tamaño original)
    def draw_logo_smaller_lugares(screen):
        logo_loaded_local = False
        logo_image_local = None
        
        try:
            if os.path.exists("images/logo.png"):
                logo_image_local = cv2.imread("images/logo.png", cv2.IMREAD_UNCHANGED)
                if logo_image_local is not None:
                    logo_loaded_local = True
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
            logo_height = 114  # 95% de 120 (casi imperceptible)
            if len(logo_image_local.shape) == 3:
                original_height, original_width = logo_image_local.shape[:2]
            else:
                original_height, original_width = logo_image_local.shape[0], logo_image_local.shape[1]
            
            aspect_ratio = original_width / original_height
            logo_width = int(logo_height * aspect_ratio)
            logo_resized = cv2.resize(logo_image_local, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
            logo_x = (view_width - logo_width) // 2
            logo_y = 20  # Subido un poco más (de 30 a 20)
            
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
        else:
            draw_logo_func(screen)
    
    draw_logo_smaller_lugares(lugares_screen)
    
    # Función para dibujar card redonda con flecha hacia la izquierda (estilo infantil) - en la parte superior izquierda
    def draw_back_card_lugares(screen, elevated=False):
        """
        Dibuja una card redonda con flecha hacia la izquierda en el centro, estilo infantil, azul con flecha blanca
        en la parte superior izquierda
        """
        # Posición base del lado izquierdo (parte superior)
        base_card_radius = 50
        card_margin_x = 120  # Movido más a la izquierda
        card_margin_y = 80  # Margen desde el borde superior
        
        # Efecto de elevación si está elevada
        elevation_offset = 0
        scale_factor = 1.0
        shadow_offset_base = 5
        
        if elevated:
            elevation_offset = -15  # Mover hacia arriba
            scale_factor = 1.08  # Aumentar tamaño ligeramente
            shadow_offset_base = 10  # Sombra más grande cuando está elevada
        
        card_radius = int(base_card_radius * scale_factor)
        card_center = (card_margin_x + int(base_card_radius * scale_factor), 
                       card_margin_y + int(base_card_radius * scale_factor) + elevation_offset)
        
        # Sombra suave (múltiples capas para efecto infantil)
        shadow_offset = int(shadow_offset_base * scale_factor)
        for i in range(3, 0, -1):
            shadow_alpha = i / 3.0 * 0.3
            shadow_color = tuple(int(c * shadow_alpha) for c in (0, 100, 100))  # Azul para la sombra
            offset = shadow_offset + (3 - i)
            cv2.circle(screen, 
                      (card_center[0] + offset, card_center[1] + offset), 
                      card_radius, shadow_color, -1)
        
        # Gradiente azul pastel (simulado con círculos concéntricos)
        color_intensity = 1.15 if elevated else 1.0
        base_blue_light = int(100 * color_intensity)
        base_blue_medium = int(50 * color_intensity)
        base_blue_dark = int(30 * color_intensity)
        base_blue_light = min(255, base_blue_light)
        base_blue_medium = min(255, base_blue_medium)
        base_blue_dark = min(255, base_blue_dark)
        
        # Círculo exterior más claro
        cv2.circle(screen, card_center, card_radius, (255, base_blue_light, base_blue_light), -1)
        # Círculo interior más intenso
        cv2.circle(screen, card_center, int(card_radius * 0.85), (255, base_blue_medium, base_blue_medium), -1)
        # Círculo más interno
        cv2.circle(screen, card_center, int(card_radius * 0.7), (255, base_blue_dark, base_blue_dark), -1)
        
        # Borde blanco suave (estilo infantil)
        cv2.circle(screen, card_center, card_radius, (255, 255, 255), 4)
        cv2.circle(screen, card_center, card_radius - 2, (200, 200, 200), 2)
        
        # Dibujar la flecha hacia la izquierda blanca en el centro
        arrow_size = int(card_radius * 0.4)
        thickness = 5
        
        # Punto de inicio de la flecha (punta)
        arrow_tip_x = card_center[0] - arrow_size
        arrow_tip_y = card_center[1]
        
        # Punto final de la flecha (cola)
        arrow_tail_x = card_center[0] + arrow_size
        arrow_tail_y = card_center[1]
        
        # Puntos para las dos líneas de la flecha (formando un triángulo)
        arrow_top_x = arrow_tail_x - arrow_size * 0.3
        arrow_top_y = arrow_tail_y - arrow_size * 0.5
        arrow_bottom_x = arrow_tail_x - arrow_size * 0.3
        arrow_bottom_y = arrow_tail_y + arrow_size * 0.5
        
        # Sombra de la flecha
        shadow_offset_arrow = 2
        cv2.line(screen, 
                (arrow_tip_x + shadow_offset_arrow, arrow_tip_y + shadow_offset_arrow), 
                (arrow_tail_x + shadow_offset_arrow, arrow_tail_y + shadow_offset_arrow), 
                (150, 150, 150), thickness)
        cv2.line(screen, 
                (arrow_tip_x + shadow_offset_arrow, arrow_tip_y + shadow_offset_arrow), 
                (int(arrow_top_x) + shadow_offset_arrow, int(arrow_top_y) + shadow_offset_arrow), 
                (150, 150, 150), thickness)
        cv2.line(screen, 
                (arrow_tip_x + shadow_offset_arrow, arrow_tip_y + shadow_offset_arrow), 
                (int(arrow_bottom_x) + shadow_offset_arrow, int(arrow_bottom_y) + shadow_offset_arrow), 
                (150, 150, 150), thickness)
        
        # Flecha blanca principal
        cv2.line(screen, 
                (arrow_tip_x, arrow_tip_y), 
                (arrow_tail_x, arrow_tail_y), 
                (255, 255, 255), thickness)
        cv2.line(screen, 
                (arrow_tip_x, arrow_tip_y), 
                (int(arrow_top_x), int(arrow_top_y)), 
                (255, 255, 255), thickness)
        cv2.line(screen, 
                (arrow_tip_x, arrow_tip_y), 
                (int(arrow_bottom_x), int(arrow_bottom_y)), 
                (255, 255, 255), thickness)
    
    # Función para dibujar card redonda con X (estilo infantil) - en la parte superior derecha
    def draw_close_card_lugares(screen, elevated=False):
        """
        Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca
        en la parte superior derecha
        """
        # Posición base del lado derecho (parte superior)
        base_card_radius = 50
        card_margin_x = 180
        card_margin_y = 80  # Margen desde el borde superior
        
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
        color_intensity = 1.15 if elevated else 1.0
        base_red_light = int(100 * color_intensity)
        base_red_medium = int(50 * color_intensity)
        base_red_dark = int(30 * color_intensity)
        base_red_light = min(255, base_red_light)
        base_red_medium = min(255, base_red_medium)
        base_red_dark = min(255, base_red_dark)
        
        # Círculo exterior más claro
        cv2.circle(screen, card_center, card_radius, (base_red_light, base_red_light, 255), -1)
        # Círculo interior más intenso
        cv2.circle(screen, card_center, int(card_radius * 0.85), (base_red_medium, base_red_medium, 255), -1)
        # Círculo más interno
        cv2.circle(screen, card_center, int(card_radius * 0.7), (base_red_dark, base_red_dark, 255), -1)
        
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
    
    # Variables para la card de retroceso (necesarias para la detección)
    back_card_radius_lugares = 50
    back_card_margin_x_lugares = 120  # Movido más a la izquierda
    back_card_margin_y_lugares = 80
    back_card_center_x_lugares = back_card_margin_x_lugares + back_card_radius_lugares
    back_card_center_y_lugares = back_card_margin_y_lugares + back_card_radius_lugares
    
    # Usar exactamente la misma lógica de detección que en selección de acciones
    back_card_detection_size_lugares = back_card_radius_lugares * 2.4
    back_card_detection_x_lugares = back_card_center_x_lugares - back_card_radius_lugares * 1.2
    back_card_detection_y_lugares = back_card_center_y_lugares - back_card_radius_lugares * 1.2
    back_card_detection_w_lugares = back_card_detection_size_lugares
    back_card_detection_h_lugares = back_card_detection_size_lugares
    
    # Función para detectar si se tocó la card de retroceso
    def detectar_back_card_touch_lugares(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de retroceso"""
        return (back_card_detection_x_lugares <= x_touch <= back_card_detection_x_lugares + back_card_detection_w_lugares and
                back_card_detection_y_lugares <= y_touch <= back_card_detection_y_lugares + back_card_detection_h_lugares)
    
    # Variables para la card de cerrar (necesarias para la detección)
    close_card_radius_lugares = 50
    close_card_margin_x_lugares = 180
    close_card_margin_y_lugares = 80
    close_card_center_x_lugares = view_width - close_card_margin_x_lugares - close_card_radius_lugares
    close_card_center_y_lugares = close_card_margin_y_lugares + close_card_radius_lugares
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_lugares = close_card_radius_lugares * 2.4
    close_card_detection_x_lugares = close_card_center_x_lugares - close_card_radius_lugares * 1.2
    close_card_detection_y_lugares = close_card_center_y_lugares - close_card_radius_lugares * 1.2
    close_card_detection_w_lugares = close_card_detection_size_lugares
    close_card_detection_h_lugares = close_card_detection_size_lugares
    
    # Función para detectar si se tocó la card de cerrar
    def detectar_close_card_touch_lugares(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar"""
        return (close_card_detection_x_lugares <= x_touch <= close_card_detection_x_lugares + close_card_detection_w_lugares and
                close_card_detection_y_lugares <= y_touch <= close_card_detection_y_lugares + close_card_detection_h_lugares)
    
    # Definir los 6 lugares
    # (Casa, Clinica, Escuela, Estacion-policia, Parque, Cocina)
    lugares = [
        {"nombre": "Casa", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Casa.png"},
        {"nombre": "Clinica", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Clinica.png"},
        {"nombre": "Escuela", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Escuela.png"},
        {"nombre": "Estacion-policia", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Estacion-Policia.png"},
        {"nombre": "Parque", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Parque.png"},
        {"nombre": "Cocina", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Cocina.png"}
    ]
    
    # Cargar imágenes de los lugares si existen
    lugar_images = {}
    for lugar in lugares:
        if "imagen" in lugar and os.path.exists(lugar["imagen"]):
            img = cv2.imread(lugar["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                lugar_images[lugar["nombre"]] = img
                print(f"✓ Imagen cargada para {lugar['nombre']}: {lugar['imagen']}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {lugar['nombre']}: {lugar['imagen']}")
    
    # Título
    titulo_texto = "Selecciona un lugar"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 120  # Posición del título
    
    # Sombra del título
    put_text_ubuntu(lugares_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    put_text_ubuntu(lugares_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Función para dibujar el texto de lugares seleccionados
    def draw_lugares_seleccionados(screen, num_seleccionados):
        """
        Dibuja el texto que muestra el número de lugares seleccionados
        """
        texto = f"Lugares seleccionados: {num_seleccionados}"
        font_scale = 1.2  # Aumentado de 0.9 a 1.2 (similar al juego de clasificación)
        thickness = 3  # Aumentado de 2 a 3 (similar al juego de clasificación)
        
        # Usar PIL para obtener tamaño preciso del texto con Ubuntu font
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font = get_ubuntu_font(font_scale=font_scale, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), texto, font=font)
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font.getbbox(texto) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                text_width = bbox[2] - bbox[0]
        except:
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size, _ = cv2.getTextSize(texto, font, font_scale, thickness)
            text_width = text_size[0]
        
        # Centrar texto horizontalmente
        text_x = (view_width - text_width) // 2
        text_y = 145  # Subido más (de 190 a 145)
        
        # Sombra del texto
        put_text_ubuntu(screen, texto, (text_x + 2, text_y + 2), 
                   font_scale, (0, 0, 0), thickness + 1, bold=True)
        # Texto principal
        put_text_ubuntu(screen, texto, (text_x, text_y), 
                   font_scale, (255, 255, 255), thickness, bold=True)
    
    # Dimensiones de las cards (más pequeñas para 6 cards en 2 filas de 3)
    card_width = 240  # Reducido de 280 a 240
    card_height = 210  # Reducido de 250 a 210
    card_spacing = 25  # Reducido de 30 a 25
    
    # Calcular posiciones (centradas, 3 cards por fila, 2 filas)
    total_width = 3 * card_width + 2 * card_spacing
    start_x = (view_width - total_width) // 2 - 30  # Movido un poco a la izquierda
    start_y = 180  # Subido un poco más (de 200 a 180)
    
    lugar_positions = {}
    for idx, lugar in enumerate(lugares):
        row = idx // 3
        col = idx % 3
        x = start_x + col * (card_width + card_spacing)
        y = start_y + row * (card_height + card_spacing)
        lugar_positions[lugar["nombre"]] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': lugar["color"],
            'imagen': lugar_images.get(lugar["nombre"])
        }
    
    # Función para dibujar las cards de lugares
    def draw_lugar_cards(screen, lugar_positions, selected_cards=None):
        if selected_cards is None:
            selected_cards = []
        for nombre, pos in lugar_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            color = pos['color']
            img = pos.get('imagen')
            
            # Determinar si está seleccionada
            is_selected = nombre in selected_cards
            
            # Efecto visual si está seleccionada: hacer la card más ancha (como en acciones)
            width_scale = 1.15 if is_selected else 1.0  # 15% más ancha cuando está seleccionada
            w_scaled = int(w * width_scale)
            # Centrar la card expandida
            x_scaled = x - (w_scaled - w) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            
            # Efecto visual si está seleccionada: borde verde
            border_color = (0, 255, 0) if is_selected else (100, 100, 100)  # Verde si seleccionada
            border_thickness = 5 if is_selected else 2
            
            # Dibujar sombra
            shadow_offset = 8
            shadow_color = (40, 40, 40)
            for i in range(3, 0, -1):
                shadow_alpha = i / 3.0
                shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
                offset_layer = shadow_offset + (3 - i)
                cv2.rectangle(screen, 
                            (x_scaled + offset_layer, y + offset_layer), 
                            (x_scaled + w_scaled + offset_layer, y + h + offset_layer), 
                            shadow_color_layer, -1)
            
            # Dibujar la imagen como fondo completo de la card si existe
            if img is not None:
                img_resized = cv2.resize(img, (w_scaled, h), interpolation=cv2.INTER_AREA)
                
                if x_scaled >= 0 and y >= 0 and x_scaled + w_scaled <= screen.shape[1] and y + h <= screen.shape[0]:
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        # Con transparencia
                        alpha = img_resized[:, :, 3] / 255.0
                        img_bgr = img_resized[:, :, :3]
                        for c in range(3):
                            screen[y:y+h, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y:y+h, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        # Sin transparencia
                        screen[y:y+h, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
            else:
                # Si no hay imagen, dibujar card con color
                cv2.rectangle(screen, (x_scaled, y), (x_scaled + w_scaled, y + h), color, -1)
            
            # Dibujar borde (verde si está seleccionada)
            cv2.rectangle(screen, (x_scaled, y), (x_scaled + w_scaled, y + h), border_color, border_thickness)
    
    # Función para dibujar el botón "Jugar"
    def draw_jugar_button(screen, elevated=False, blocked=False):
        """
        Dibuja un botón "Jugar" en la parte inferior de la pantalla
        """
        button_width = 200
        button_height = 60
        button_margin_bottom = 90  # Ajustado para bajar un poco el botón
        button_x = (view_width - button_width) // 2  # Centrado horizontalmente
        button_y = view_height - button_margin_bottom - button_height
        
        # Efecto de elevación si está elevado (solo si no está bloqueado)
        elevation_offset = 0
        scale_factor = 1.0
        shadow_offset_base = 5
        
        if elevated and not blocked:
            elevation_offset = -5
            scale_factor = 1.05
            shadow_offset_base = 8
        
        w_scaled = int(button_width * scale_factor)
        h_scaled = int(button_height * scale_factor)
        x_scaled = button_x - (w_scaled - button_width) // 2
        y_scaled = button_y + elevation_offset - (h_scaled - button_height) // 2
        
        # Asegurar que no se salga de los límites
        x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
        y_scaled = max(0, min(y_scaled, screen.shape[0] - h_scaled))
        
        # Dibujar sombra (más suave si está bloqueado)
        shadow_offset = int(shadow_offset_base * scale_factor)
        shadow_color = (40, 40, 40) if not blocked else (20, 20, 20)
        for i in range(3, 0, -1):
            shadow_alpha = i / 3.0
            shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
            offset_layer = shadow_offset + (3 - i)
            cv2.rectangle(screen, 
                        (x_scaled + offset_layer, y_scaled + offset_layer), 
                        (x_scaled + w_scaled + offset_layer, y_scaled + h_scaled + offset_layer), 
                        shadow_color_layer, -1)
        
        # Color del botón (gris si está bloqueado, verde si no)
        if blocked:
            button_color = (100, 100, 100)  # Gris cuando está bloqueado
            border_color = (80, 80, 80)  # Borde gris oscuro
            text_color = (150, 150, 150)  # Texto gris claro
        else:
            button_color = (100, 255, 100) if not elevated else (150, 255, 150)
            border_color = (0, 200, 0)
            text_color = (255, 255, 255)  # Texto blanco
        
        cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), button_color, -1)
        
        # Borde del botón
        border_thickness = 3
        cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
        
        # Texto "Jugar" - aumentado y centrado
        texto = "Jugar"
        font_scale = 1.5 * scale_factor  # Aumentado de 0.8 a 1.5
        thickness = 3  # Aumentado de 2 a 3 para más bold
        
        # Usar PIL para obtener tamaño preciso del texto con Ubuntu font
        bbox_top = 0
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font = get_ubuntu_font(font_scale=font_scale, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), texto, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]  # Top of bbox (usually negative for ascenders)
            except AttributeError:
                bbox = font.getbbox(texto) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                bbox_top = bbox[1]
        except:
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size, _ = cv2.getTextSize(texto, font, font_scale, thickness)
            text_width = text_size[0]
            text_height = text_size[1]
            bbox_top = 0
        
        # Centrar texto en el botón
        text_x = x_scaled + (w_scaled - text_width) // 2
        # PIL usa y como baseline. Para centrar verticalmente, necesitamos ajustar considerando el bbox_top
        # Mover el texto más arriba para que no sobresalga del botón
        if bbox_top < 0:
            # Hay ascenders, ajustar la posición moviendo más arriba
            # Restar más para subir el texto
            text_y = y_scaled + h_scaled // 2 + abs(bbox_top) // 2 - text_height // 2
        else:
            # Sin ascenders significativos, centrar normalmente pero más arriba
            text_y = y_scaled + h_scaled // 2 - text_height // 2
        
        # Sombra del texto
        put_text_ubuntu(screen, texto, (text_x + 2, text_y + 2), 
                   font_scale, (0, 0, 0), thickness + 1, bold=True)
        # Texto principal
        put_text_ubuntu(screen, texto, (text_x, text_y), 
                   font_scale, text_color, thickness, bold=True)
    
    # Variables para el botón "Jugar"
    jugar_button_width = 200
    jugar_button_height = 60
    jugar_button_margin_bottom = 90
    jugar_button_x = (view_width - jugar_button_width) // 2
    jugar_button_y = view_height - jugar_button_margin_bottom - jugar_button_height
    
    # Función para detectar si se tocó el botón "Jugar"
    def detectar_jugar_button_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área del botón Jugar"""
        return (jugar_button_x <= x_touch <= jugar_button_x + jugar_button_width and
                jugar_button_y <= y_touch <= jugar_button_y + jugar_button_height)
    
    # Dibujar las cards, la flecha de retroceso y la X (sin selección inicial)
    draw_lugar_cards(lugares_screen, lugar_positions, selected_cards=[])
    draw_lugares_seleccionados(lugares_screen, 0)  # Inicialmente 0 seleccionados
    draw_back_card_lugares(lugares_screen, elevated=False)
    draw_close_card_lugares(lugares_screen, elevated=False)
    draw_jugar_button(lugares_screen, elevated=False, blocked=True)  # Bloqueado inicialmente
    
    # Configurar ventana
    window_name = existing_window_name if existing_window_name else "Selección de Lugares"
    window_exists = False
    try:
        window_exists = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0
    except:
        pass
    
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    lugares_screen_scaled = scale_to_videobeam(lugares_screen)
    cv2.imshow(window_name, lugares_screen_scaled)
    
    # Iniciar streams de cámara para detección de toques
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    lugares_seleccionados = []  # Lista de lugares seleccionados
    jugar_elevated = False
    jugar_pressed = False
    jugar_press_frames = 0
    back_elevated_lugares = False
    close_elevated_lugares = False
    frame_count = 0
    initialization_delay = 10
    
    # Sistema de debounce temporal
    from collections import defaultdict
    touch_history = defaultdict(list)
    min_touch_frames = 2
    touch_persistence_threshold = 0.8
    min_touch_area = 100
    max_touch_area = 50000
    last_valid_touch_time = time.time()
    touch_cooldown = 0.15
    history_cleanup_interval = 30
    max_history_age = 1.0
    
    # Bucle principal de detección de toques
    while True:
        frame_count += 1
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        
        if frame is None or depth_frame is None:
            continue
        
        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]
        
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
        
        # Crear la máscara de toques
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        
        # Aplicar filtros
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
        kernel = np.ones((2, 2), np.uint8)
        touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
        touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)
        
        if frame_count < initialization_delay:
            # Redibujar la pantalla durante el delay
            temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
            for y in range(view_height):
                ratio = y / view_height
                r = int(255 * (0.3 + 0.4 * ratio))
                g = int(200 * (0.5 + 0.3 * ratio))
                b = int(255 * (0.8 - 0.3 * ratio))
                temp_screen[y, :] = [b, g, r]
            draw_logo_smaller_lugares(temp_screen)
            draw_lugar_cards(temp_screen, lugar_positions, selected_cards=lugares_seleccionados)
            draw_lugares_seleccionados(temp_screen, len(lugares_seleccionados))
            draw_back_card_lugares(temp_screen, elevated=False)
            draw_close_card_lugares(temp_screen, elevated=False)
            is_blocked = len(lugares_seleccionados) != 1
            draw_jugar_button(temp_screen, elevated=False, blocked=is_blocked)
            lugares_screen_scaled = scale_to_videobeam(temp_screen)
            cv2.imshow(window_name, lugares_screen_scaled)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        
        # Encontrar contornos
        contours, _ = cv2.findContours(touch_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Limpiar historial antiguo
        current_time = time.time()
        if frame_count % history_cleanup_interval == 0:
            for key in list(touch_history.keys()):
                touch_history[key] = [
                    touch for touch in touch_history[key] 
                    if current_time - touch[3] < max_history_age
                ]
                if not touch_history[key]:
                    del touch_history[key]
        
        # Procesar contornos
        valid_touches_this_frame = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_touch_area <= area <= max_touch_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Mapeo de coordenadas
                    x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                    
                    # Agregar a historial
                    touch_key = (x_touch // 25, y_touch // 25)
                    touch_history[touch_key].append((x_touch, y_touch, area, current_time))
                    
                    # Verificar persistencia
                    if len(touch_history[touch_key]) >= min_touch_frames:
                        if current_time - last_valid_touch_time > touch_cooldown:
                            recent_touches = touch_history[touch_key][-min_touch_frames:]
                            all_recent = all(current_time - touch[3] < 1.0 for touch in recent_touches)
                            
                            if all_recent and len(recent_touches) >= min_touch_frames:
                                areas = [touch[2] for touch in recent_touches]
                                avg_area = sum(areas) / len(areas)
                                
                                if min(areas) > 0:
                                    area_variance = max(areas) / min(areas)
                                    if area_variance < 3.5 and min_touch_area <= avg_area <= max_touch_area:
                                        valid_touches_this_frame.append((x_touch, y_touch, touch_key))
        
        # Procesar toques válidos
        for x_touch, y_touch, touch_key in valid_touches_this_frame:
            if touch_key in touch_history:
                del touch_history[touch_key]
            last_valid_touch_time = current_time
            
            # Verificar si se tocó el botón de retroceso (flecha)
            if detectar_back_card_touch_lugares(x_touch, y_touch):
                print("Card de retroceso tocada en selección de lugares")
                # Mostrar efecto de elevación en la flecha y volver a la vista de acciones
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_smaller_lugares(temp_screen)
                draw_lugar_cards(temp_screen, lugar_positions, selected_cards=lugares_seleccionados)
                draw_lugares_seleccionados(temp_screen, len(lugares_seleccionados))
                draw_back_card_lugares(temp_screen, elevated=True)
                draw_close_card_lugares(temp_screen, elevated=False)
                is_blocked = len(lugares_seleccionados) != 1
                draw_jugar_button(temp_screen, elevated=False, blocked=is_blocked)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver a la vista anterior (retornar "BACK" para indicar retroceso)
                rgb_stream.stop()
                depth_stream.stop()
                return "BACK"
            
            # Verificar si se tocó la X
            if detectar_close_card_touch_lugares(x_touch, y_touch):
                print("Card de cerrar tocada en selección de lugares")
                # Mostrar efecto de elevación en la X
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_smaller_lugares(temp_screen)
                draw_lugar_cards(temp_screen, lugar_positions, selected_cards=lugares_seleccionados)
                draw_lugares_seleccionados(temp_screen, len(lugares_seleccionados))
                draw_back_card_lugares(temp_screen, elevated=False)
                draw_close_card_lugares(temp_screen, elevated=True)
                is_blocked = len(lugares_seleccionados) != 1
                draw_jugar_button(temp_screen, elevated=False, blocked=is_blocked)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver al menú principal (retornar "MENU" para indicar que se debe volver al menú)
                rgb_stream.stop()
                depth_stream.stop()
                return "MENU"
            
            # Verificar si se está tocando el botón "Jugar"
            if detectar_jugar_button_touch(x_touch, y_touch):
                if len(lugares_seleccionados) == 1:  # Requiere exactamente 1 lugar
                    if not jugar_pressed:
                        # Iniciar el efecto de elevación
                        jugar_pressed = True
                        jugar_press_frames = 0
                        jugar_elevated = True
                        print("Botón Jugar presionado en selección de lugares")
                else:
                    jugar_elevated = False
                    jugar_pressed = False
                    jugar_press_frames = 0
                    print("Botón Jugar bloqueado - selecciona al menos un lugar")
            else:
                # Si no se está tocando el botón Jugar, resetear estado
                if jugar_pressed:
                    jugar_pressed = False
                    jugar_press_frames = 0
                jugar_elevated = False
                
                # Verificar si se seleccionó un lugar
                lugar_seleccionado_temp = None
                for nombre, pos in lugar_positions.items():
                    x, y = pos['x'], pos['y']
                    w, h = pos['width'], pos['height']
                    if x <= x_touch <= x + w and y <= y_touch <= y + h:
                        lugar_seleccionado_temp = nombre
                        break
                
                if lugar_seleccionado_temp:
                    # Toggle de selección de lugar (igual que otras vistas)
                    if lugar_seleccionado_temp in lugares_seleccionados:
                        lugares_seleccionados.remove(lugar_seleccionado_temp)
                        print(f"Lugar deseleccionado: {lugar_seleccionado_temp}")
                    else:
                        # Limitar a máximo 1 lugar
                        if len(lugares_seleccionados) < 1:
                            lugares_seleccionados.append(lugar_seleccionado_temp)
                            print(f"Lugar seleccionado: {lugar_seleccionado_temp}")
                            historia_tts_speak(lugar_seleccionado_temp, tipo="lugar")
                        else:
                            # Si ya hay un lugar seleccionado, reemplazarlo
                            lugares_seleccionados.clear()
                            lugares_seleccionados.append(lugar_seleccionado_temp)
                            print(f"Lugar seleccionado: {lugar_seleccionado_temp} (reemplazando selección anterior)")
                            historia_tts_speak(lugar_seleccionado_temp, tipo="lugar")
                    print(f"Lugares seleccionados: {lugares_seleccionados}")
                    # Continuar en el bucle para mostrar la selección actualizada
        
        # Procesar el botón "Jugar" si está presionado
        if jugar_pressed:
            jugar_press_frames += 1
            if jugar_press_frames >= 10:  # Después de 10 frames, procesar la acción
                print("Botón Jugar procesado - pasando a vista final")
                # Detener streams de lugares antes de ir a la vista final
                rgb_stream.stop()
                depth_stream.stop()
                
                # Retornar los lugares seleccionados (la función que llama manejará la vista final)
                return lugares_seleccionados
        
        # Redibujar la pantalla en cada frame
        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            temp_screen[y, :] = [b, g, r]
        draw_logo_smaller_lugares(temp_screen)
        draw_lugar_cards(temp_screen, lugar_positions, selected_cards=lugares_seleccionados)
        draw_lugares_seleccionados(temp_screen, len(lugares_seleccionados))
        draw_back_card_lugares(temp_screen, elevated=back_elevated_lugares)
        draw_close_card_lugares(temp_screen, elevated=close_elevated_lugares)
        # Bloquear botón si no hay exactamente 1 lugar seleccionado
        is_blocked = len(lugares_seleccionados) != 1
        draw_jugar_button(temp_screen, elevated=jugar_elevated, blocked=is_blocked)
        lugares_screen_scaled = scale_to_videobeam(temp_screen)
        cv2.imshow(window_name, lugares_screen_scaled)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Solo cerrar si se presiona 'q' explícitamente
            rgb_stream.stop()
            depth_stream.stop()
            return None
    
    # Detener streams y cerrar (esto no debería ejecutarse normalmente)
    rgb_stream.stop()
    depth_stream.stop()
    return lugares_seleccionados if lugares_seleccionados else None  # Retornar los lugares seleccionados (o None si no hay ninguno)

