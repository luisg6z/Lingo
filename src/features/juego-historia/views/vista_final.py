"""
Vista final: summary cards and Hablar button.
"""
import os
import sys
import cv2
import numpy as np
import time

_views_dir = os.path.dirname(os.path.abspath(__file__))
_feature_dir = os.path.dirname(_views_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_feature_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.components import draw_circular_button, is_point_in_circular_button

from .voice_flow import run_historia_voice_flow

def mostrar_vista_final(device, coordenadas, dmax_map, dmin_map, draw_logo_func, sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados, existing_window_name=None):
    """
    Muestra la vista final con las 4 cards seleccionadas (2 sujetos, 1 acción, 1 lugar).
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        sujetos_seleccionados: Lista de sujetos seleccionados (máximo 2)
        acciones_seleccionadas: Lista de acciones seleccionadas (máximo 1)
        lugares_seleccionados: Lista de lugares seleccionados (máximo 1)
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        "MENU" o None: "MENU" si se debe volver al menú, None si se canceló
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
    final_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        final_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(final_screen)
    
    # Función para dibujar card redonda con X (estilo infantil) - en la parte superior derecha
    def draw_close_card_final(screen, elevated=False):
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
    
    # Variables para la card de cerrar (necesarias para la detección)
    close_card_radius_final = 50
    close_card_margin_x_final = 180
    close_card_margin_y_final = 80
    close_card_center_x_final = view_width - close_card_margin_x_final - close_card_radius_final
    close_card_center_y_final = close_card_margin_y_final + close_card_radius_final
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_final = close_card_radius_final * 2.4
    close_card_detection_x_final = close_card_center_x_final - close_card_radius_final * 1.2
    close_card_detection_y_final = close_card_center_y_final - close_card_radius_final * 1.2
    close_card_detection_w_final = close_card_detection_size_final
    close_card_detection_h_final = close_card_detection_size_final
    
    # Función para detectar si se tocó la card de cerrar
    def detectar_close_card_touch_final(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar"""
        return (close_card_detection_x_final <= x_touch <= close_card_detection_x_final + close_card_detection_w_final and
                close_card_detection_y_final <= y_touch <= close_card_detection_y_final + close_card_detection_h_final)
    
    # Importar componente de botón circular
    from src.components import draw_circular_button, is_point_in_circular_button
    
    # Botón circular "Hablar" (misma lógica y posición que absurdos-visuales)
    # Calcular posición igual que en absurdos-visuales
    circle_radius_hablar = 60
    circle_center_x_hablar = view_width // 2
    espacio_para_franja = 100  # Espacio necesario para la franja "Escuchando..." (50 arriba + 50 abajo)
    # Posición del botón: cerca de la parte inferior pero dejando espacio para la franja
    circle_center_y_hablar = view_height - espacio_para_franja - circle_radius_hablar
    
    hablar_button_bounds_final = None  # Se inicializará cuando se dibuje
    
    # Cargar imágenes de sujetos, acciones y lugares
    # Sujetos: Niño, Niña, Cocinera, Policia, Doctor, Maestra
    historias = [
        {"nombre": "Niño", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Niño.png"},
        {"nombre": "Niña", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Niña.png"},
        {"nombre": "Cocinera", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Cocinera.png"},
        {"nombre": "Policia", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Policia.png"},
        {"nombre": "Doctor", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Doctor.png"},
        {"nombre": "Maestra", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Maestra.png"}
    ]
    
    # Acciones: Ayudar, Trabajar, Cocinar, Correr, Llamar, Jugar
    acciones = [
        {"nombre": "Ayudar", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Ayudar.png"},
        {"nombre": "Trabajar", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Trabajar.png"},
        {"nombre": "Cocinar", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Cocinar.png"},
        {"nombre": "Correr", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Correr.png"},
        {"nombre": "Llamar", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Llamar.png"},
        {"nombre": "Jugar", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Jugar.png"}
    ]
    
    # Lugares: Casa, Clinica, Escuela, Estacion-policia, Parque, Cocina
    lugares = [
        {"nombre": "Casa", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Casa.png"},
        {"nombre": "Clinica", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Clinica.png"},
        {"nombre": "Escuela", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Escuela.png"},
        {"nombre": "Estacion-policia", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Estacion-Policia.png"},
        {"nombre": "Parque", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Parque.png"},
        {"nombre": "Cocina", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Cocina.png"}
    ]
    
    # Crear diccionarios para buscar imágenes
    historia_images = {}
    for historia in historias:
        if os.path.exists(historia["imagen"]):
            img = cv2.imread(historia["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                historia_images[historia["nombre"]] = img
    
    accion_images = {}
    for accion in acciones:
        if os.path.exists(accion["imagen"]):
            img = cv2.imread(accion["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                accion_images[accion["nombre"]] = img
    
    lugar_images = {}
    for lugar in lugares:
        if os.path.exists(lugar["imagen"]):
            img = cv2.imread(lugar["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                lugar_images[lugar["nombre"]] = img
    
    # Preparar las 4 cards: 2 sujetos, 1 acción, 1 lugar
    cards_final = []
    
    # Agregar sujetos (máximo 2)
    for sujeto in sujetos_seleccionados[:2]:
        for historia in historias:
            if historia["nombre"] == sujeto:
                cards_final.append({
                    'nombre': sujeto,
                    'tipo': 'sujeto',
                    'color': historia["color"],
                    'imagen': historia_images.get(sujeto)
                })
                break
    
    # Agregar acción (máximo 1)
    if acciones_seleccionadas and len(acciones_seleccionadas) > 0:
        accion = acciones_seleccionadas[0]
        for acc in acciones:
            if acc["nombre"] == accion:
                cards_final.append({
                    'nombre': accion,
                    'tipo': 'accion',
                    'color': acc["color"],
                    'imagen': accion_images.get(accion)
                })
                break
    
    # Agregar lugar (máximo 1)
    if lugares_seleccionados and len(lugares_seleccionados) > 0:
        lugar = lugares_seleccionados[0]
        for lug in lugares:
            if lug["nombre"] == lugar:
                cards_final.append({
                    'nombre': lugar,
                    'tipo': 'lugar',
                    'color': lug["color"],
                    'imagen': lugar_images.get(lugar)
                })
                break
    
    # Dimensiones de las cards (4 cards en una fila)
    card_width = 250
    card_height = 300
    card_spacing = 30
    
    # Calcular posiciones (centradas, 4 cards en una fila)
    total_width = len(cards_final) * card_width + (len(cards_final) - 1) * card_spacing
    start_x = (view_width - total_width) // 2
    start_y = 250  # Debajo del logo
    
    card_positions_final = {}
    for idx, card in enumerate(cards_final):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        card_positions_final[card['nombre']] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': card['color'],
            'imagen': card['imagen']
        }
    
    # Función para dibujar las 4 cards finales (no presionables)
    def draw_final_cards(screen, card_positions):
        for nombre, pos in card_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            color = pos['color']
            img = pos.get('imagen')
            
            # Dibujar sombra
            shadow_offset = 5
            shadow_color = (40, 40, 40)
            for i in range(3, 0, -1):
                shadow_alpha = i / 3.0
                shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
                offset_layer = shadow_offset + (3 - i)
                cv2.rectangle(screen, 
                            (x + offset_layer, y + offset_layer), 
                            (x + w + offset_layer, y + h + offset_layer), 
                            shadow_color_layer, -1)
            
            # Dibujar la card
            if img is not None:
                # Redimensionar imagen para que quepa en la card
                img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                
                # Si la imagen tiene canal alpha, compositar
                if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                    # Con transparencia
                    alpha = img_resized[:, :, 3] / 255.0
                    img_bgr = img_resized[:, :, :3]
                    for c in range(3):
                        screen[y:y+h, x:x+w, c] = (
                            alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y:y+h, x:x+w, c]
                        )
                else:
                    # Sin transparencia
                    screen[y:y+h, x:x+w] = img_resized[:, :, :3]
            else:
                # Si no hay imagen, dibujar card con color
                cv2.rectangle(screen, (x, y), (x + w, y + h), color, -1)
            
            # Dibujar borde
            border_color = (100, 100, 100)
            border_thickness = 2
            cv2.rectangle(screen, (x, y), (x + w, y + h), border_color, border_thickness)
    
    # Dibujar las cards y la X
    draw_final_cards(final_screen, card_positions_final)
    draw_close_card_final(final_screen, elevated=False)
    
    # Configurar ventana
    window_name = existing_window_name if existing_window_name else "Vista Final"
    window_exists = False
    try:
        window_exists = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0
    except:
        pass
    
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    final_screen_scaled = scale_to_videobeam(final_screen)
    cv2.imshow(window_name, final_screen_scaled)
    
    # Iniciar streams de cámara para detección de toques (solo para la X)
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    close_elevated_final = False
    frame_count = 0
    initialization_delay = 10
    
    # Sistema de debounce temporal
    from collections import defaultdict
    touch_history = defaultdict(list)
    min_touch_frames = 2
    min_touch_area = 100
    max_touch_area = 50000
    last_valid_touch_time = time.time()
    touch_cooldown = 0.15
    history_cleanup_interval = 30
    max_history_age = 1.0
    
    # Bucle principal de detección de toques (solo para la X)
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
            draw_logo_func(temp_screen)
            draw_final_cards(temp_screen, card_positions_final)
            # Redibujar botón Hablar usando componente
            hablar_button_bounds_final = draw_circular_button(
                temp_screen,
                circle_center_x_hablar, circle_center_y_hablar, circle_radius_hablar,
                "Hablar",
                bg_color=(0, 200, 0),  # Green
                border_color=(255, 255, 255),
                border_thickness=3,
                text_color=(255, 255, 255),
                font_scale=1.3,
                bold=True,
                shadow=True
            )
            draw_close_card_final(temp_screen, elevated=False)
            final_screen_scaled = scale_to_videobeam(temp_screen)
            cv2.imshow(window_name, final_screen_scaled)
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
        
        # Procesar toques válidos (solo para la X)
        for x_touch, y_touch, touch_key in valid_touches_this_frame:
            if touch_key in touch_history:
                del touch_history[touch_key]
            last_valid_touch_time = current_time
            
            # Verificar si se tocó la X
            if detectar_close_card_touch_final(x_touch, y_touch):
                print("Card de cerrar tocada en vista final")
                # Mostrar efecto de elevación en la X
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_func(temp_screen)
                draw_final_cards(temp_screen, card_positions_final)
                # Redibujar botón Hablar usando componente
                hablar_button_bounds_final = draw_circular_button(
                    temp_screen,
                    circle_center_x_hablar, circle_center_y_hablar, circle_radius_hablar,
                    "Hablar",
                    bg_color=(0, 200, 0),  # Green
                    border_color=(255, 255, 255),
                    border_thickness=3,
                    text_color=(255, 255, 255),
                    font_scale=1.3,
                    bold=True,
                    shadow=True
                )
                draw_close_card_final(temp_screen, elevated=True)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver al menú principal (retornar "MENU" para indicar que se debe volver al menú)
                rgb_stream.stop()
                depth_stream.stop()
                return "MENU"
            # Verificar si se tocó el botón Hablar usando componente helper
            if hablar_button_bounds_final and is_point_in_circular_button(x_touch, y_touch, hablar_button_bounds_final):
                print("Botón Hablar tocado en vista final")
                rgb_stream.stop()
                depth_stream.stop()
                ret = run_historia_voice_flow(
                    window_name, view_width, view_height, scale_to_videobeam, draw_logo_func,
                    sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados,
                    device, coordenadas, dmax_map, dmin_map,
                    draw_close_card_final, detectar_close_card_touch_final,
                    close_card_detection_x_final, close_card_detection_y_final,
                    close_card_detection_w_final, close_card_detection_h_final,
                    VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT,
                )
                # Si retorna "RESTART", reiniciar streams y continuar en el bucle para volver a la vista final
                if ret == "RESTART":
                    # Reiniciar streams para poder detectar toques en la vista final
                    # Los streams anteriores se detuvieron en _run_historia_voice_flow
                    try:
                        rgb_stream.stop()
                        depth_stream.stop()
                    except:
                        pass
                    rgb_stream = device.create_color_stream()
                    depth_stream = device.create_depth_stream()
                    rgb_stream.start()
                    depth_stream.start()
                    continue
                return ret if ret is not None else "MENU"
        
        # Redibujar la pantalla en cada frame
        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            temp_screen[y, :] = [b, g, r]
        draw_logo_func(temp_screen)
        draw_final_cards(temp_screen, card_positions_final)
        # Redibujar botón Hablar usando componente
        hablar_button_bounds_final = draw_circular_button(
            temp_screen,
            circle_center_x_hablar, circle_center_y_hablar, circle_radius_hablar,
            "Hablar",
            bg_color=(0, 200, 0),  # Green
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.3,
            bold=True,
            shadow=True
        )
        draw_close_card_final(temp_screen, elevated=close_elevated_final)
        # Redibujar botón Hablar usando componente
        hablar_button_bounds_final = draw_circular_button(
            temp_screen,
            circle_center_x_hablar, circle_center_y_hablar, circle_radius_hablar,
            "Hablar",
            bg_color=(0, 200, 0),  # Green
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.3,
            bold=True,
            shadow=True
        )
        final_screen_scaled = scale_to_videobeam(temp_screen)
        cv2.imshow(window_name, final_screen_scaled)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Solo cerrar si se presiona 'q' explícitamente
            rgb_stream.stop()
            depth_stream.stop()
            return None
    
    # Detener streams y cerrar (esto no debería ejecutarse normalmente)
    rgb_stream.stop()
    depth_stream.stop()
    return None


