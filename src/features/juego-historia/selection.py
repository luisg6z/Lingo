"""
Escenario selection for juego-historia
"""
import cv2
import numpy as np
import os
import time
import pygame
import json
from openni import openni2

# Get project root and add to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sys
sys.path.insert(0, project_root)

# Import core utilities
from src.core.ui_utils import scale_to_videobeam, draw_logo

# Resolución del videobeam (segunda pantalla)
VIDEOBEAM_WIDTH = 1920
VIDEOBEAM_HEIGHT = 1080


def mostrar_seleccion_escenarios_historia(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de escenarios para el juego de historia.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        str o None: Escenario seleccionado o None si se canceló
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
    
    # Crear fondo
    escenarios_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        escenarios_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(escenarios_screen, view_width, view_height)
    
    # Función para dibujar card redonda con X (estilo infantil)
    def draw_close_card(screen, elevated=False):
        """
        Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca
        
        Args:
            screen: Pantalla donde dibujar
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
    
    # Variables para la card de cerrar (necesarias para la detección)
    close_card_radius = 50
    close_card_margin_x = 180
    close_card_margin_y = 80
    close_card_center_x = view_width - close_card_margin_x - close_card_radius
    close_card_center_y = close_card_margin_y + close_card_radius
    
    # Crear un área rectangular de detección
    close_card_detection_size = close_card_radius * 2.4
    close_card_detection_x = close_card_center_x - close_card_radius * 1.2
    close_card_detection_y = close_card_center_y - close_card_radius * 1.2
    close_card_detection_w = close_card_detection_size
    close_card_detection_h = close_card_detection_size
    
    # Función para detectar si se tocó la card de cerrar
    def detectar_close_card_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar"""
        return (close_card_detection_x <= x_touch <= close_card_detection_x + close_card_detection_w and
                close_card_detection_y <= y_touch <= close_card_detection_y + close_card_detection_h)
    
    # Definir los 6 escenarios
    escenarios = [
        {"id": "1", "nombre": "Escenario 1", "color": (100, 200, 255)},  # Azul claro
        {"id": "2", "nombre": "Escenario 2", "color": (100, 255, 150)},  # Verde claro
        {"id": "3", "nombre": "Escenario 3", "color": (255, 150, 200)},  # Rosa claro
        {"id": "4", "nombre": "Escenario 4", "color": (255, 200, 100)},  # Naranja claro
        {"id": "5", "nombre": "Escenario 5", "color": (200, 100, 255)},  # Morado claro
        {"id": "6", "nombre": "Escenario 6", "color": (255, 255, 100)},  # Amarillo claro
    ]
    
    # Cargar imágenes de escenarios si existen
    escenario_images = {}
    for escenario in escenarios:
        # Buscar imagen en assets/images del feature
        image_path = os.path.join(project_root, "src", "features", "juego-historia", "assets", "images", f"Escenario{escenario['id']}.png")
        if not os.path.exists(image_path):
            # Fallback a images/ en la raíz
            image_path = os.path.join(project_root, "images", f"Escenario{escenario['id']}.png")
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                escenario_images[escenario['id']] = img
                print(f"✓ Imagen cargada para {escenario['nombre']}: {image_path}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {escenario['nombre']}: {image_path}")
        else:
            print(f"⚠ Archivo no encontrado: {image_path}")
    
    # Título
    titulo_texto = "Selecciona un escenario"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 180
    # Sombra del título
    cv2.putText(escenarios_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    cv2.putText(escenarios_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Dimensiones de las cards de escenarios
    card_width = 180
    card_height = 220
    card_spacing = 30
    
    # Calcular posiciones (6 cards en 2 filas de 3)
    cards_per_row = 3
    total_width = cards_per_row * card_width + (cards_per_row - 1) * card_spacing
    start_x = (view_width - total_width) // 2
    start_y = 250  # Debajo del título
    
    escenario_positions = {}
    for idx, escenario in enumerate(escenarios):
        row = idx // cards_per_row
        col = idx % cards_per_row
        x = start_x + col * (card_width + card_spacing)
        y = start_y + row * (card_height + card_spacing)
        escenario_positions[escenario['id']] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': escenario['color'],
            'nombre': escenario['nombre'],
            'imagen': escenario_images.get(escenario['id']),
            'seleccionado': False  # Estado de selección
        }
    
    # Función para dibujar las cards de escenarios
    def draw_escenario_cards(screen, escenario_positions, elevated_card=None):
        for escenario_id, pos in escenario_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            color = pos['color']
            seleccionado = pos['seleccionado']
            
            # Efecto de elevación si esta card está elevada o seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 8
            
            if elevated_card == escenario_id or seleccionado:
                elevation_offset = -20
                scale_factor = 1.05
                shadow_offset_base = 12
                # Hacer el color más brillante cuando está elevada/seleccionada
                color = tuple(min(255, int(c * 1.15)) for c in color)
            
            # Aplicar escala
            w_scaled = int(w * scale_factor)
            h_scaled = int(h * scale_factor)
            x_scaled = x - (w_scaled - w) // 2
            y_scaled = y + elevation_offset - (h_scaled - h) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, screen.shape[0] - h_scaled))
            
            # Dibujar sombra
            shadow_offset = int(shadow_offset_base * scale_factor)
            shadow_color = (40, 40, 40)
            for i in range(3, 0, -1):
                shadow_alpha = i / 3.0
                shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
                offset_layer = shadow_offset + (3 - i)
                cv2.rectangle(screen, 
                            (x_scaled + offset_layer, y_scaled + offset_layer), 
                            (x_scaled + w_scaled + offset_layer, y_scaled + h_scaled + offset_layer), 
                            shadow_color_layer, -1)
            
            # Dibujar la imagen si existe
            if pos['imagen'] is not None:
                img = pos['imagen'].copy()
                img_resized = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                
                if x_scaled >= 0 and y_scaled >= 0 and x_scaled + w_scaled <= screen.shape[1] and y_scaled + h_scaled <= screen.shape[0]:
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        alpha = img_resized[:, :, 3] / 255.0
                        img_bgr = img_resized[:, :, :3]
                        for c in range(3):
                            screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
            else:
                # Dibujar card con color si no hay imagen
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
                border_color = tuple(max(0, c - 30) for c in color)
                border_thickness = 7 if seleccionado else 5
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
            
            # Dibujar nombre del escenario
            nombre_texto = pos['nombre']
            font_nombre = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nombre = 0.6
            thickness_nombre = 2
            nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            nombre_x = x_scaled + (w_scaled - nombre_size[0]) // 2
            nombre_y = y_scaled + int(180 * scale_factor)
            
            # Sombra del texto
            cv2.putText(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                       font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
            # Texto principal
            cv2.putText(screen, nombre_texto, (nombre_x, nombre_y), 
                       font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            
            # Dibujar indicador de selección si está seleccionado
            if seleccionado:
                check_size = 20
                check_x = x_scaled + w_scaled - check_size - 10
                check_y = y_scaled + 10
                cv2.circle(screen, (check_x, check_y), check_size, (0, 255, 0), -1)
                cv2.circle(screen, (check_x, check_y), check_size, (255, 255, 255), 2)
                # Dibujar checkmark
                cv2.line(screen, (check_x - 8, check_y), (check_x - 3, check_y + 8), (255, 255, 255), 3)
                cv2.line(screen, (check_x - 3, check_y + 8), (check_x + 8, check_y - 5), (255, 255, 255), 3)
    
    # Dibujar las cards inicialmente
    draw_escenario_cards(escenarios_screen, escenario_positions)
    # Dibujar card de cerrar (X roja)
    draw_close_card(escenarios_screen)
    
    # Usar el nombre de ventana existente si se proporciona, o crear uno nuevo
    window_name = existing_window_name if existing_window_name else "Selección de Escenarios"
    
    # Verificar si la ventana existe
    window_exists = False
    if existing_window_name:
        try:
            prop = cv2.getWindowProperty(existing_window_name, cv2.WND_PROP_VISIBLE)
            if prop >= 0:
                window_exists = True
        except:
            window_exists = False
    
    # Si no existe, crear nueva ventana
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar en pantalla
    escenarios_screen_scaled = scale_to_videobeam(escenarios_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT)
    cv2.imshow(window_name, escenarios_screen_scaled)
    
    # Función para detectar escenario seleccionado
    def detectar_escenario_seleccionado(x_touch, y_touch, escenario_positions):
        for escenario_id, pos in escenario_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            if x <= x_touch <= x + w and y <= y_touch <= y + h:
                return escenario_id
        return None
    
    # Iniciar streams de cámara para detección de toques
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    # Bucle de detección de toques
    escenario_seleccionado = None
    close_card_elevated = False
    frame_count = 0
    initialization_delay = 10  # Delay inicial para evitar detecciones inmediatas
    
    try:
        while True:
            frame_count += 1
            
            # Redibujar continuamente (al inicio del bucle para asegurar que siempre se muestre)
            escenarios_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
            for y in range(view_height):
                ratio = y / view_height
                r = int(255 * (0.3 + 0.4 * ratio))
                g = int(200 * (0.5 + 0.3 * ratio))
                b = int(255 * (0.8 - 0.3 * ratio))
                escenarios_screen[y, :] = [b, g, r]
            draw_logo_func(escenarios_screen, view_width, view_height)
            draw_escenario_cards(escenarios_screen, escenario_positions)
            draw_close_card(escenarios_screen, elevated=close_card_elevated)
            escenarios_screen_scaled = scale_to_videobeam(escenarios_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT)
            cv2.imshow(window_name, escenarios_screen_scaled)
            
            # No procesar toques durante el delay inicial
            if frame_count < initialization_delay:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            frame = rgb_stream.read_frame()
            depth_frame = depth_stream.read_frame()
            
            if frame is None or depth_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            bgr_data = cv2.flip(bgr_data, 1)
            bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]
            
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
            
            # Crear la máscara que considera solo los valores entre dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
            
            # Aplicar filtros
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
            kernel = np.ones((2, 2), np.uint8)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar los contornos de los toques
            contours, _ = cv2.findContours(touch_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Procesar cada contorno
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Área mínima
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas de ventana a viewport
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Verificar si se tocó la card de cerrar
                        if detectar_close_card_touch(x_touch, y_touch):
                            print("Card de cerrar tocada")
                            close_card_elevated = True
                            # Redibujar con efecto
                            escenarios_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                escenarios_screen[y, :] = [b, g, r]
                            draw_logo_func(escenarios_screen, view_width, view_height)
                            draw_escenario_cards(escenarios_screen, escenario_positions)
                            draw_close_card(escenarios_screen, elevated=True)
                            escenarios_screen_scaled = scale_to_videobeam(escenarios_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT)
                            cv2.imshow(window_name, escenarios_screen_scaled)
                            cv2.waitKey(100)
                            # Volver al menú
                            return None
                        
                        # Detectar si se seleccionó un escenario
                        escenario_id = detectar_escenario_seleccionado(x_touch, y_touch, escenario_positions)
                        if escenario_id:
                            # Toggle selección
                            escenario_positions[escenario_id]['seleccionado'] = not escenario_positions[escenario_id]['seleccionado']
                            print(f"Escenario {escenario_id} {'seleccionado' if escenario_positions[escenario_id]['seleccionado'] else 'deseleccionado'}")
                            
                            # Si se seleccionó, guardar como seleccionado y salir
                            if escenario_positions[escenario_id]['seleccionado']:
                                escenario_seleccionado = escenario_id
                            
                            # Redibujar
                            escenarios_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                escenarios_screen[y, :] = [b, g, r]
                            draw_logo_func(escenarios_screen, view_width, view_height)
                            draw_escenario_cards(escenarios_screen, escenario_positions, elevated_card=escenario_id)
                            draw_close_card(escenarios_screen, elevated=close_card_elevated)
                            escenarios_screen_scaled = scale_to_videobeam(escenarios_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT)
                            cv2.imshow(window_name, escenarios_screen_scaled)
                            cv2.waitKey(100)
                            close_card_elevated = False
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            close_card_elevated = False
    
    finally:
        # Detener los streams de la cámara
        rgb_stream.stop()
        depth_stream.stop()
    
    return escenario_seleccionado

