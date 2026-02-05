import cv2
import numpy as np
import os
import time
import pygame
from detection_logic import ObjectDetector, draw_shine_effect

# --- CONFIGURACIÓN DE PANTALLA ---
# Si usas un segundo monitor o videobeam, ajusta SCREEN_OFFSET_X al ancho de tu pantalla principal (ej: 1920)
SCREEN_OFFSET_X = 1920 
SCREEN_OFFSET_Y = 0

# Resolución del videobeam/segunda pantalla (ajusta si no se ve a pantalla completa)
# Comúnmente 1280x800, 1920x1080, etc.
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
# --------------------------------

# Variable global para mantener el estado del audio entre vistas
_bocina_muted_global = False
_background_music_global = None


def mostrar_seleccion_niveles_clasificacion(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de niveles para el juego de Clasificación.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        str o None: Nombre del nivel seleccionado ("1", "2", "3") o None si se canceló
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
    view_width = VIEW_WIDTH
    view_height = VIEW_HEIGHT
    
    # Crear fondo
    niveles_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        niveles_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(niveles_screen)
    
    # Inicializar pygame si no está inicializado
    global _bocina_muted_global, _background_music_global
    try:
        pygame.mixer.get_init()
    except:
        pygame.mixer.init()
    
    # Cargar imágenes de bocina
    bocina_image = None
    bocina_mute_image = None
    if os.path.exists("images/Bocina.png"):
        bocina_image = cv2.imread("images/Bocina.png", cv2.IMREAD_UNCHANGED)
        if bocina_image is not None:
            print("✓ Imagen de bocina cargada: images/Bocina.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina: images/Bocina.png")
    else:
        print("⚠ No se encontró la imagen: images/Bocina.png")
    
    if os.path.exists("images/BocinaMute.png"):
        bocina_mute_image = cv2.imread("images/BocinaMute.png", cv2.IMREAD_UNCHANGED)
        if bocina_mute_image is not None:
            print("✓ Imagen de bocina mute cargada: images/BocinaMute.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina mute: images/BocinaMute.png")
    else:
        print("⚠ No se encontró la imagen: images/BocinaMute.png")
    
    # Cargar y configurar el audio de fondo (solo si no está cargado)
    if _background_music_global is None:
        if os.path.exists("relax-meditate-gentle-peaceful-291162.mp3"):
            try:
                _background_music_global = pygame.mixer.Sound("relax-meditate-gentle-peaceful-291162.mp3")
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3")
                # Iniciar el audio automáticamente si no está muteado
                if not _bocina_muted_global:
                    _background_music_global.play(-1)  # -1 significa bucle infinito
                    print("✓ Audio de fondo iniciado automáticamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el audio de fondo: {e}")
        else:
            print("⚠ No se encontró el archivo de audio: relax-meditate-gentle-peaceful-291162.mp3")
    
    # Usar el estado global del audio
    bocina_muted = _bocina_muted_global
    
    # Función para dibujar card cuadrada con icono de bocina
    def draw_bocina_card(screen, muted=False):
        """
        Dibuja una card cuadrada con icono de bocina en el centro
        
        Args:
            screen: Pantalla donde dibujar
            muted: Si True, muestra la imagen de bocina muteada (BocinaMute.png), si False muestra Bocina.png
        """
        # Posición base del lado derecho (parte inferior)
        base_card_size = 100
        card_margin_x = 180
        card_margin_y = 50  # Margen desde el borde inferior
        
        card_size = base_card_size
        # Calcular posición del cuadrado (esquina superior izquierda)
        card_x = view_width - card_margin_x - card_size
        card_y = view_height - card_margin_y - card_size
        
        # Dibujar la imagen completa como fondo de la card
        current_bocina_image = bocina_mute_image if muted and bocina_mute_image is not None else bocina_image
        if current_bocina_image is not None:
            # Redimensionar la imagen para que llene completamente la card
            bocina_resized = cv2.resize(current_bocina_image, (card_size, card_size), interpolation=cv2.INTER_AREA)
            
            # Asegurar que esté dentro de los límites
            if card_x >= 0 and card_y >= 0 and card_x + card_size <= screen.shape[1] and card_y + card_size <= screen.shape[0]:
                # Si la imagen tiene canal alfa (transparencia)
                if len(bocina_resized.shape) == 3 and bocina_resized.shape[2] == 4:
                    # Extraer canal alfa
                    alpha = bocina_resized[:, :, 3] / 255.0
                    # Convertir BGR de la imagen
                    img_bgr = bocina_resized[:, :, :3]
                    # Mezclar con el fondo
                    for c in range(3):
                        screen[card_y:card_y+card_size, card_x:card_x+card_size, c] = (
                            alpha * img_bgr[:, :, c] + 
                            (1 - alpha) * screen[card_y:card_y+card_size, card_x:card_x+card_size, c]
                        )
                else:
                    # Sin canal alfa, copiar directamente
                    screen[card_y:card_y+card_size, card_x:card_x+card_size] = bocina_resized[:, :, :3]
    
    # Variables para la card de bocina (necesarias para la detección)
    bocina_card_size = 100
    bocina_card_margin_x = 180
    bocina_card_margin_y = 50
    bocina_card_x = view_width - bocina_card_margin_x - bocina_card_size
    bocina_card_y = view_height - bocina_card_margin_y - bocina_card_size
    
    # Crear un área rectangular de detección
    bocina_card_detection_size = int(bocina_card_size * 1.2)
    bocina_card_detection_x = bocina_card_x - int(bocina_card_size * 0.1)
    bocina_card_detection_y = bocina_card_y - int(bocina_card_size * 0.1)
    bocina_card_detection_w = bocina_card_detection_size
    bocina_card_detection_h = bocina_card_detection_size
    
    # Función para detectar si se tocó la card de bocina
    def detectar_bocina_card_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de bocina"""
        return (bocina_card_detection_x <= x_touch <= bocina_card_detection_x + bocina_card_detection_w and
                bocina_card_detection_y <= y_touch <= bocina_card_detection_y + bocina_card_detection_h)
    
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
        card_margin_x = 180  # Aumentado para mover la card más a la izquierda
        card_margin_y = 80  # Reducido más para mover la card más arriba
        
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
    # Usar un área rectangular para la detección, similar a las otras cards
    close_card_radius_niveles = 50
    close_card_margin_x_niveles = 180
    close_card_margin_y_niveles = 80  # Reducido más para mover la card más arriba
    close_card_center_x_niveles = view_width - close_card_margin_x_niveles - close_card_radius_niveles
    close_card_center_y_niveles = close_card_margin_y_niveles + close_card_radius_niveles
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_niveles = close_card_radius_niveles * 2.4  # Área más grande para facilitar el toque
    close_card_detection_x_niveles = close_card_center_x_niveles - close_card_radius_niveles * 1.2
    close_card_detection_y_niveles = close_card_center_y_niveles - close_card_radius_niveles * 1.2
    close_card_detection_w_niveles = close_card_detection_size_niveles
    close_card_detection_h_niveles = close_card_detection_size_niveles
    
    # Función para detectar si se tocó la card de cerrar (usando área rectangular como las otras cards)
    def detectar_close_card_touch_niveles(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar (usando área rectangular)"""
        return (close_card_detection_x_niveles <= x_touch <= close_card_detection_x_niveles + close_card_detection_w_niveles and
                close_card_detection_y_niveles <= y_touch <= close_card_detection_y_niveles + close_card_detection_h_niveles)
    
    # Cargar imágenes de los niveles (3 escenarios)
    nivel_images = {}
    nivel_paths = {
        "1": "images/Nivel1Clasificacion.png",
        "2": "images/Nivel2Clasificacion.png",
        "3": "images/Nivel3Clasificacion.png"
    }
    
    for nivel, path in nivel_paths.items():
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                nivel_images[nivel] = img
                print(f"✓ Imagen cargada para {nivel} escenario(s): {path}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {nivel}: {path}")
        else:
            print(f"⚠ Archivo no encontrado: {path}")
    
    # Título
    titulo_texto = "Elige el numero de escenarios"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 180
    # Sombra del título
    cv2.putText(niveles_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    cv2.putText(niveles_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Dimensiones de las cards de niveles (3 cards en una fila)
    card_width = 250
    card_height = 350
    card_spacing = 40
    
    # Calcular posiciones (centradas, 3 cards en fila)
    total_width = 3 * card_width + 2 * card_spacing
    start_x = (view_width - total_width) // 2
    start_y = 250  # Debajo del título
    
    nivel_positions = {}
    for idx, (nivel, path) in enumerate(nivel_paths.items()):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        nivel_positions[nivel] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'imagen': nivel_images.get(nivel)
        }
    
    # Función para dibujar las cards de niveles
    def draw_nivel_cards(screen, nivel_positions, elevated_card=None):
        for nombre, pos in nivel_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            
            # Efecto de elevación si esta card está seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 8  # Sombra base más pequeña y consistente
            
            if elevated_card == nombre:
                elevation_offset = -20  # Menos elevación
                scale_factor = 1.05  # Menos escala
                shadow_offset_base = 12  # Sombra más grande cuando está elevada, pero consistente
            
            # Aplicar escala
            w_scaled = int(w * scale_factor)
            h_scaled = int(h * scale_factor)
            x_scaled = x - (w_scaled - w) // 2
            y_scaled = y + elevation_offset - (h_scaled - h) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, screen.shape[0] - h_scaled))
            
            # Dibujar sombra (más suave y consistente)
            shadow_offset = int(shadow_offset_base * scale_factor)
            shadow_color = (40, 40, 40)  # Color de sombra más consistente
            # Sombra más suave con múltiples capas
            for i in range(3, 0, -1):
                shadow_alpha = i / 3.0
                shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
                offset_layer = shadow_offset + (3 - i)
                cv2.rectangle(screen, 
                            (x_scaled + offset_layer, y_scaled + offset_layer), 
                            (x_scaled + w_scaled + offset_layer, y_scaled + h_scaled + offset_layer), 
                            shadow_color_layer, -1)
            
            # Dibujar la imagen como fondo completo de la card
            if pos['imagen'] is not None:
                img = pos['imagen'].copy()
                img_resized = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                
                if x_scaled >= 0 and y_scaled >= 0 and x_scaled + w_scaled <= screen.shape[1] and y_scaled + h_scaled <= screen.shape[0]:
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        # Con transparencia
                        alpha = img_resized[:, :, 3] / 255.0
                        img_bgr = img_resized[:, :, :3]
                        for c in range(3):
                            screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        # Sin transparencia
                        screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
            
            # Dibujar texto del nivel (ej: "1 escenario", "2 escenarios", etc.)
            nivel_num = nombre
            texto_nivel = f"{nivel_num} escenario" if nivel_num == "1" else f"{nivel_num} escenarios"
            font_nivel = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nivel = 0.9
            thickness_nivel = 2
            
            # Calcular el ancho disponible
            available_width = w_scaled - 40
            
            # Verificar si el texto cabe
            nivel_size, _ = cv2.getTextSize(texto_nivel, font_nivel, font_scale_nivel, thickness_nivel)
            while nivel_size[0] > available_width and font_scale_nivel > 0.5:
                font_scale_nivel -= 0.1
                nivel_size, _ = cv2.getTextSize(texto_nivel, font_nivel, font_scale_nivel, thickness_nivel)
            
            # Calcular posición del texto (centrado horizontalmente, en la parte inferior de la card)
            nivel_x = x_scaled + (w_scaled - nivel_size[0]) // 2
            nivel_y = y_scaled + int(320 * scale_factor)  # Posición en la parte inferior de la card
            
            # Dibujar sombra del texto
            cv2.putText(screen, texto_nivel, (nivel_x + 2, nivel_y + 2), 
                       font_nivel, font_scale_nivel, (0, 0, 0), thickness_nivel + 1)
            # Dibujar texto principal (blanco)
            cv2.putText(screen, texto_nivel, (nivel_x, nivel_y), 
                       font_nivel, font_scale_nivel, (255, 255, 255), thickness_nivel)
    
    # Dibujar las cards inicialmente
    draw_nivel_cards(niveles_screen, nivel_positions)
    # Dibujar card de cerrar (X roja)
    draw_close_card(niveles_screen)
    # Dibujar card de bocina
    draw_bocina_card(niveles_screen, muted=bocina_muted)
    
    # Usar el nombre de ventana existente si se proporciona, o crear uno nuevo
    window_name = existing_window_name if existing_window_name else "Selección de Niveles"
    
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
        cv2.moveWindow(window_name, SCREEN_OFFSET_X, SCREEN_OFFSET_Y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar en pantalla (transición suave sin cerrar)
    cv2.imshow(window_name, niveles_screen)
    
    # Función para detectar nivel seleccionado
    def detectar_nivel_seleccionado(x_touch, y_touch, nivel_positions):
        for nombre, pos in nivel_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            if x <= x_touch <= x + w and y <= y_touch <= y + h:
                return nombre
        return None
    
    # Iniciar streams de cámara para detección de toques
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    # Bucle de detección de toques
    nivel_seleccionado_flag = False
    nivel_seleccionado = None
    
    try:
        while True:
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
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Procesar cada contorno
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Primero verificar si se tocó la card de bocina
                        if detectar_bocina_card_touch(x_touch, y_touch):
                            print("Card de bocina tocada en selección de niveles")
                            
                            # Cambiar el estado de mute (usando variable global)
                            _bocina_muted_global = not _bocina_muted_global
                            bocina_muted = _bocina_muted_global
                            print(f"Bocina {'muteada' if bocina_muted else 'activada'}")
                            
                            # Controlar el audio según el estado
                            if _background_music_global is not None:
                                if bocina_muted:
                                    # Detener el audio cuando está muteada
                                    pygame.mixer.stop()
                                    print("Audio de fondo detenido")
                                else:
                                    # Reproducir el audio en bucle cuando está activada
                                    _background_music_global.play(-1)  # -1 significa bucle infinito
                                    print("Audio de fondo iniciado (bucle)")
                            
                            # Redibujar la pantalla con el nuevo estado
                            temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                temp_screen[y, :] = [b, g, r]
                            draw_logo_func(temp_screen)
                            cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                       font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                            cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                       font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                            draw_nivel_cards(temp_screen, nivel_positions)
                            draw_close_card(temp_screen)
                            draw_bocina_card(temp_screen, muted=bocina_muted)
                            niveles_screen = temp_screen
                            cv2.imshow(window_name, niveles_screen)
                            continue
                        
                        # Verificar si se tocó la card de cerrar
                        if detectar_close_card_touch_niveles(x_touch, y_touch):
                            print("Card de cerrar tocada - Volviendo al menú principal")
                            
                            # Efecto visual de elevación (animación)
                            for frame_num in range(10):
                                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                for y in range(view_height):
                                    ratio = y / view_height
                                    r = int(255 * (0.3 + 0.4 * ratio))
                                    g = int(200 * (0.5 + 0.3 * ratio))
                                    b = int(255 * (0.8 - 0.3 * ratio))
                                    temp_screen[y, :] = [b, g, r]
                                draw_logo_func(temp_screen)
                                # Redibujar título
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                
                                # Dibujar cards de niveles
                                draw_nivel_cards(temp_screen, nivel_positions)
                                
                                # Dibujar card de cerrar con efecto de elevación
                                if frame_num >= 5:
                                    draw_close_card(temp_screen, elevated=True)
                                else:
                                    draw_close_card(temp_screen, elevated=False)
                                
                                # Dibujar card de bocina
                                draw_bocina_card(temp_screen, muted=bocina_muted)
                                
                                cv2.imshow(window_name, temp_screen)
                                cv2.waitKey(30)
                            
                            return None  # Retornar None para volver al menú principal
                        
                        # Detectar si se seleccionó un nivel
                        if not nivel_seleccionado_flag:
                            nivel_sel = detectar_nivel_seleccionado(x_touch, y_touch, nivel_positions)
                            if nivel_sel:
                                print(f"Nivel seleccionado: {nivel_sel}")
                                
                                nivel_seleccionado = nivel_sel
                                nivel_seleccionado_flag = True
                                
                                # Animación rápida de elevación (reducida para que no tarde)
                                for frame_num in range(5):
                                    temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                    for y in range(view_height):
                                        ratio = y / view_height
                                        r = int(255 * (0.3 + 0.4 * ratio))
                                        g = int(200 * (0.5 + 0.3 * ratio))
                                        b = int(255 * (0.8 - 0.3 * ratio))
                                        temp_screen[y, :] = [b, g, r]
                                    draw_logo_func(temp_screen)
                                    # Redibujar título
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                    
                                    if frame_num >= 2:
                                        draw_nivel_cards(temp_screen, nivel_positions, elevated_card=nivel_sel)
                                    else:
                                        draw_nivel_cards(temp_screen, nivel_positions)
                                    
                                    # Dibujar card de cerrar y card de bocina (mantenerlas visibles)
                                    draw_close_card(temp_screen)
                                    draw_bocina_card(temp_screen, muted=bocina_muted)
                                    cv2.imshow(window_name, temp_screen)
                                    cv2.waitKey(20)
                                
                                # Mantener elevada brevemente antes de pasar a la siguiente vista
                                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                for y in range(view_height):
                                    ratio = y / view_height
                                    r = int(255 * (0.3 + 0.4 * ratio))
                                    g = int(200 * (0.5 + 0.3 * ratio))
                                    b = int(255 * (0.8 - 0.3 * ratio))
                                    temp_screen[y, :] = [b, g, r]
                                draw_logo_func(temp_screen)
                                # Redibujar título
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                draw_nivel_cards(temp_screen, nivel_positions, elevated_card=nivel_sel)
                                # Dibujar card de cerrar y card de bocina (mantenerlas visibles)
                                draw_close_card(temp_screen)
                                draw_bocina_card(temp_screen, muted=bocina_muted)
                                cv2.imshow(window_name, temp_screen)
                                cv2.waitKey(200)
                                
                                # Convertir el nivel seleccionado a número
                                num_escenarios = int(nivel_seleccionado)
                                
                                # Pasar el nombre de la ventana existente para reutilizarla
                                # Mostrar vista de 5 escenarios (reutilizará la misma ventana)
                                cards_seleccionadas = mostrar_vista_5_escenarios(
                                    device, coordenadas, dmax_map, dmin_map, draw_logo_func, num_escenarios, 
                                    existing_window_name=window_name
                                )
                                
                                # Si se presionó la card de retroceso (flecha), redibujar la pantalla de selección de niveles
                                if cards_seleccionadas == "BACK":
                                    # Redibujar la pantalla de selección de niveles
                                    for y in range(view_height):
                                        ratio = y / view_height
                                        r = int(255 * (0.3 + 0.4 * ratio))
                                        g = int(200 * (0.5 + 0.3 * ratio))
                                        b = int(255 * (0.8 - 0.3 * ratio))
                                        niveles_screen[y, :] = [b, g, r]
                                    draw_logo_func(niveles_screen)
                                    cv2.putText(niveles_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                    cv2.putText(niveles_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                    draw_nivel_cards(niveles_screen, nivel_positions)
                                    draw_close_card(niveles_screen)
                                    draw_bocina_card(niveles_screen, muted=bocina_muted)
                                    cv2.imshow(window_name, niveles_screen)
                                    nivel_seleccionado_flag = False
                                    nivel_seleccionado = None
                                    continue  # Continuar el bucle para permitir más selecciones
                                
                                # Si se presionó la card de cerrar, retornar None para volver al menú principal
                                if cards_seleccionadas is None:
                                    return None
                                
                                return nivel_seleccionado
            
            # Redibujar card de cerrar y card de bocina en cada frame
            draw_close_card(niveles_screen)
            draw_bocina_card(niveles_screen, muted=bocina_muted)
            cv2.imshow(window_name, niveles_screen)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return None


def mostrar_vista_5_escenarios(device, coordenadas, dmax_map, dmin_map, draw_logo_func, num_escenarios, existing_window_name=None):
    """
    Muestra la vista con 5 escenarios donde el usuario debe seleccionar N escenarios según num_escenarios.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        num_escenarios: Número de escenarios que debe seleccionar (1-3)
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        list: Lista con los nombres de los escenarios seleccionados
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
    view_width = VIEW_WIDTH
    view_height = VIEW_HEIGHT
    
    # Crear fondo
    cards_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        cards_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(cards_screen)
    
    # Inicializar pygame si no está inicializado
    global _bocina_muted_global, _background_music_global
    try:
        pygame.mixer.get_init()
    except:
        pygame.mixer.init()
    
    # Cargar imágenes de bocina
    bocina_image = None
    bocina_mute_image = None
    if os.path.exists("images/Bocina.png"):
        bocina_image = cv2.imread("images/Bocina.png", cv2.IMREAD_UNCHANGED)
        if bocina_image is not None:
            print("✓ Imagen de bocina cargada: images/Bocina.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina: images/Bocina.png")
    else:
        print("⚠ No se encontró la imagen: images/Bocina.png")
    
    if os.path.exists("images/BocinaMute.png"):
        bocina_mute_image = cv2.imread("images/BocinaMute.png", cv2.IMREAD_UNCHANGED)
        if bocina_mute_image is not None:
            print("✓ Imagen de bocina mute cargada: images/BocinaMute.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina mute: images/BocinaMute.png")
    else:
        print("⚠ No se encontró la imagen: images/BocinaMute.png")
    
    # Cargar y configurar el audio de fondo (solo si no está cargado)
    if _background_music_global is None:
        if os.path.exists("relax-meditate-gentle-peaceful-291162.mp3"):
            try:
                _background_music_global = pygame.mixer.Sound("relax-meditate-gentle-peaceful-291162.mp3")
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3")
                # Iniciar el audio automáticamente si no está muteado
                if not _bocina_muted_global:
                    _background_music_global.play(-1)  # -1 significa bucle infinito
                    print("✓ Audio de fondo iniciado automáticamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el audio de fondo: {e}")
        else:
            print("⚠ No se encontró el archivo de audio: relax-meditate-gentle-peaceful-291162.mp3")
    
    # Usar el estado global del audio
    bocina_muted = _bocina_muted_global
    
    # Función para dibujar card cuadrada con icono de bocina
    def draw_bocina_card(screen, muted=False):
        """
        Dibuja una card cuadrada con icono de bocina en el centro
        
        Args:
            screen: Pantalla donde dibujar
            muted: Si True, muestra la imagen de bocina muteada (BocinaMute.png), si False muestra Bocina.png
        """
        # Posición base del lado derecho (parte inferior)
        base_card_size = 100
        card_margin_x = 180
        card_margin_y = 50  # Margen desde el borde inferior
        
        card_size = base_card_size
        # Calcular posición del cuadrado (esquina superior izquierda)
        card_x = view_width - card_margin_x - card_size
        card_y = view_height - card_margin_y - card_size
        
        # Dibujar la imagen completa como fondo de la card
        current_bocina_image = bocina_mute_image if muted and bocina_mute_image is not None else bocina_image
        if current_bocina_image is not None:
            # Redimensionar la imagen para que llene completamente la card
            bocina_resized = cv2.resize(current_bocina_image, (card_size, card_size), interpolation=cv2.INTER_AREA)
            
            # Asegurar que esté dentro de los límites
            if card_x >= 0 and card_y >= 0 and card_x + card_size <= screen.shape[1] and card_y + card_size <= screen.shape[0]:
                # Si la imagen tiene canal alfa (transparencia)
                if len(bocina_resized.shape) == 3 and bocina_resized.shape[2] == 4:
                    # Extraer canal alfa
                    alpha = bocina_resized[:, :, 3] / 255.0
                    # Convertir BGR de la imagen
                    img_bgr = bocina_resized[:, :, :3]
                    # Mezclar con el fondo
                    for c in range(3):
                        screen[card_y:card_y+card_size, card_x:card_x+card_size, c] = (
                            alpha * img_bgr[:, :, c] + 
                            (1 - alpha) * screen[card_y:card_y+card_size, card_x:card_x+card_size, c]
                        )
                else:
                    # Sin canal alfa, copiar directamente
                    screen[card_y:card_y+card_size, card_x:card_x+card_size] = bocina_resized[:, :, :3]
    
    # Variables para la card de bocina (necesarias para la detección)
    bocina_card_size = 100
    bocina_card_margin_x = 180
    bocina_card_margin_y = 50
    bocina_card_x = view_width - bocina_card_margin_x - bocina_card_size
    bocina_card_y = view_height - bocina_card_margin_y - bocina_card_size
    
    # Crear un área rectangular de detección
    bocina_card_detection_size = int(bocina_card_size * 1.2)
    bocina_card_detection_x = bocina_card_x - int(bocina_card_size * 0.1)
    bocina_card_detection_y = bocina_card_y - int(bocina_card_size * 0.1)
    bocina_card_detection_w = bocina_card_detection_size
    bocina_card_detection_h = bocina_card_detection_size
    
    # Función para detectar si se tocó la card de bocina
    def detectar_bocina_card_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de bocina"""
        return (bocina_card_detection_x <= x_touch <= bocina_card_detection_x + bocina_card_detection_w and
                bocina_card_detection_y <= y_touch <= bocina_card_detection_y + bocina_card_detection_h)
    
    # Función para dibujar card redonda con X (estilo infantil)
    def draw_close_card(screen, elevated=False):
        """
        Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
        """
        # Posición base del lado derecho (misma posición que en mostrar_seleccion_niveles_clasificacion)
        base_card_radius = 50
        card_margin_x = 180  # Mismo margen que en mostrar_seleccion_niveles_clasificacion
        card_margin_y = 80  # Mismo margen que en mostrar_seleccion_niveles_clasificacion
        
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
    # Usar un área rectangular para la detección, similar a las otras cards
    # Misma posición que en mostrar_seleccion_niveles_clasificacion
    close_card_radius = 50
    close_card_margin_x = 180  # Mismo margen que en mostrar_seleccion_niveles_clasificacion
    close_card_margin_y = 80  # Mismo margen que en mostrar_seleccion_niveles_clasificacion
    close_card_center_x = view_width - close_card_margin_x - close_card_radius
    close_card_center_y = close_card_margin_y + close_card_radius
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size = close_card_radius * 2.4  # Área más grande para facilitar el toque
    close_card_detection_x = close_card_center_x - close_card_radius * 1.2
    close_card_detection_y = close_card_center_y - close_card_radius * 1.2
    close_card_detection_w = close_card_detection_size
    close_card_detection_h = close_card_detection_size
    
    # Función para detectar si se tocó la card de cerrar (usando área rectangular como las otras cards)
    def detectar_close_card_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar (usando área rectangular)"""
        return (close_card_detection_x <= x_touch <= close_card_detection_x + close_card_detection_w and
                close_card_detection_y <= y_touch <= close_card_detection_y + close_card_detection_h)
    
    # Título
    titulo_texto = f"Selecciona {num_escenarios} escenario(s)"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 200  # Bajado más para evitar choque con el logo
    # Sombra del título
    cv2.putText(cards_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    cv2.putText(cards_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Definir 5 cards con las imágenes reales
    card_images = {}
    card_paths = {
        "Escenario 1": "images/EscenarioGranja.png",
        "Escenario 2": "images/Escenario 6.png",
        "Escenario 3": "images/Escenario 3.png",
        "Escenario 4": "images/Escenario 4.png",
        "Escenario 5": "images/Escenario 8.png"
    }
    
    # Cargar imágenes de las cards
    for card_name, path in card_paths.items():
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                card_images[card_name] = img
        else:
            # Si no existe la imagen, crear una card de color sólido
            card_images[card_name] = None
    
    # Dimensiones de las cards (5 cards: 3 arriba, 2 abajo)
    card_width = 260  # Reducido para evitar que se corten
    card_height = 230  # Reducido para evitar que se corten
    card_spacing_x = 35  # Espacio entre cards horizontal
    card_spacing_y = 25  # Espacio entre cards vertical
    
    # Calcular posiciones (grid 3x2: 3 cards arriba, 2 cards abajo)
    # Fila superior: 3 cards
    total_width_top = 3 * card_width + 2 * card_spacing_x
    # Fila inferior: 2 cards (centradas)
    total_width_bottom = 2 * card_width + 1 * card_spacing_x
    total_height = 2 * card_height + card_spacing_y
    start_x_top = (view_width - total_width_top) // 2
    start_x_bottom = (view_width - total_width_bottom) // 2
    start_y = 250  # Posición inicial ajustada para que quepan todas las cards
    
    card_positions = {}
    card_names = list(card_paths.keys())
    
    for idx, card_name in enumerate(card_names):
        if idx < 3:
            # Primera fila: 3 cards
            row = 0
            col = idx
            x = start_x_top + col * (card_width + card_spacing_x)
        else:
            # Segunda fila: 2 cards (centradas)
            row = 1
            col = idx - 3
            x = start_x_bottom + col * (card_width + card_spacing_x)
        y = start_y + row * (card_height + card_spacing_y)
        card_positions[card_name] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'imagen': card_images.get(card_name),
            'seleccionada': False  # Estado de selección
        }
    
    # Función para dibujar las cards
    def draw_cards(screen, card_positions, selected_cards=None):
        if selected_cards is None:
            selected_cards = []
        
        for card_name, pos in card_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            is_selected = card_name in selected_cards
            
            # Efecto visual si está seleccionada: hacer la card más ancha
            width_scale = 1.15 if is_selected else 1.0  # 15% más ancha cuando está seleccionada
            w_scaled = int(w * width_scale)
            # Centrar la card expandida
            x_scaled = x - (w_scaled - w) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            
            # Efecto visual si está seleccionada
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
            
            # Dibujar fondo de la card (color si no hay imagen)
            if pos['imagen'] is not None:
                img = pos['imagen'].copy()
                img_resized = cv2.resize(img, (w_scaled, h), interpolation=cv2.INTER_AREA)
                
                if x_scaled >= 0 and y >= 0 and x_scaled + w_scaled <= screen.shape[1] and y + h <= screen.shape[0]:
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        alpha = img_resized[:, :, 3] / 255.0
                        img_bgr = img_resized[:, :, :3]
                        for c in range(3):
                            screen[y:y+h, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y:y+h, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        screen[y:y+h, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
            else:
                # Card de color sólido si no hay imagen
                card_color = (150, 150, 200) if not is_selected else (100, 200, 100)
                cv2.rectangle(screen, (x_scaled, y), (x_scaled + w_scaled, y + h), card_color, -1)
                # Texto del nombre de la card
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size, _ = cv2.getTextSize(card_name, font, font_scale, thickness)
                text_x = x_scaled + (w_scaled - text_size[0]) // 2
                text_y = y + (h + text_size[1]) // 2
                cv2.putText(screen, card_name, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            # Dibujar borde
            cv2.rectangle(screen, (x_scaled, y), (x_scaled + w_scaled, y + h), border_color, border_thickness)
    
    # Dibujar las cards inicialmente
    selected_cards = []  # Inicializar lista de cards seleccionadas
    draw_cards(cards_screen, card_positions, selected_cards)
    # Dibujar card de cerrar (X roja)
    draw_close_card(cards_screen)
    # Dibujar card de bocina
    draw_bocina_card(cards_screen, muted=bocina_muted)
    
    # Función para dibujar card redonda con flecha hacia la izquierda (estilo infantil)
    def draw_back_card(screen, elevated=False):
        """
        Dibuja una card redonda con flecha hacia la izquierda en el centro, estilo infantil, azul con flecha blanca
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
        """
        # Posición base del lado izquierdo
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
        # Hacer el color más brillante si está elevada
        color_intensity = 1.15 if elevated else 1.0
        base_blue_light = int(100 * color_intensity)
        base_blue_medium = int(50 * color_intensity)
        base_blue_dark = int(30 * color_intensity)
        # Limitar valores a 255
        base_blue_light = min(255, base_blue_light)
        base_blue_medium = min(255, base_blue_medium)
        base_blue_dark = min(255, base_blue_dark)
        
        # Círculo exterior más claro
        cv2.circle(screen, card_center, card_radius, (255, base_blue_light, base_blue_light), -1)  # Azul pastel claro
        # Círculo interior más intenso
        cv2.circle(screen, card_center, int(card_radius * 0.85), (255, base_blue_medium, base_blue_medium), -1)  # Azul pastel medio
        # Círculo más interno
        cv2.circle(screen, card_center, int(card_radius * 0.7), (255, base_blue_dark, base_blue_dark), -1)  # Azul más intenso
        
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
    
    # Variables para la card de retroceso (necesarias para la detección)
    back_card_radius_rect = 50
    back_card_margin_x_rect = 180
    back_card_margin_y_rect = 80
    back_card_center_x_rect = back_card_margin_x_rect + back_card_radius_rect
    back_card_center_y_rect = back_card_margin_y_rect + back_card_radius_rect
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    back_card_detection_size_rect = back_card_radius_rect * 2.4  # Área más grande para facilitar el toque
    back_card_detection_x_rect = back_card_center_x_rect - back_card_radius_rect * 1.2
    back_card_detection_y_rect = back_card_center_y_rect - back_card_radius_rect * 1.2
    back_card_detection_w_rect = back_card_detection_size_rect
    back_card_detection_h_rect = back_card_detection_size_rect
    
    # Función para detectar si se tocó la card de retroceso (usando área rectangular como las otras cards)
    def detectar_back_card_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de retroceso (usando área rectangular)"""
        return (back_card_detection_x_rect <= x_touch <= back_card_detection_x_rect + back_card_detection_w_rect and
                back_card_detection_y_rect <= y_touch <= back_card_detection_y_rect + back_card_detection_h_rect)
    
    # Dibujar card de retroceso (flecha azul)
    draw_back_card(cards_screen)
    
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
        cv2.moveWindow(window_name, SCREEN_OFFSET_X, SCREEN_OFFSET_Y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar la nueva vista (transición suave sin cerrar)
    cv2.imshow(window_name, cards_screen)
    
    # Función para detectar card seleccionada
    def detectar_card_seleccionada(x_touch, y_touch, card_positions, selected_cards=None):
        if selected_cards is None:
            selected_cards = []
        for card_name, pos in card_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            is_selected = card_name in selected_cards
            
            # Si está seleccionada, usar el ancho expandido
            if is_selected:
                width_scale = 1.15  # Mismo factor que en draw_cards
                w_scaled = int(w * width_scale)
                x_scaled = x - (w_scaled - w) // 2
                # Verificar toque con el área expandida
                if x_scaled <= x_touch <= x_scaled + w_scaled and y <= y_touch <= y + h:
                    return card_name
            else:
                # Verificar toque con el área normal
                if x <= x_touch <= x + w and y <= y_touch <= y + h:
                    return card_name
        return None
    
    # Iniciar streams de cámara
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    # Variables de selección (selected_cards ya inicializada arriba)
    last_touch_time = {}  # Para evitar selecciones múltiples rápidas
    debounce_time = 0.3  # 300ms de debounce
    
    try:
        while True:
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
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Procesar el contorno más grande (evitar múltiples detecciones)
            if contours:
                # Ordenar por área y tomar el más grande
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 50:
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Primero verificar si se tocó la card de bocina
                        if detectar_bocina_card_touch(x_touch, y_touch):
                            print("Card de bocina tocada en vista 5 escenarios")
                            
                            # Cambiar el estado de mute (usando variable global)
                            _bocina_muted_global = not _bocina_muted_global
                            bocina_muted = _bocina_muted_global
                            print(f"Bocina {'muteada' if bocina_muted else 'activada'}")
                            
                            # Controlar el audio según el estado
                            if _background_music_global is not None:
                                if bocina_muted:
                                    # Detener el audio cuando está muteada
                                    pygame.mixer.stop()
                                    print("Audio de fondo detenido")
                                else:
                                    # Reproducir el audio en bucle cuando está activada
                                    _background_music_global.play(-1)  # -1 significa bucle infinito
                                    print("Audio de fondo iniciado (bucle)")
                            
                            # Redibujar la pantalla con el nuevo estado
                            temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                temp_screen[y, :] = [b, g, r]
                            draw_logo_func(temp_screen)
                            cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                       font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                            cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                       font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                            draw_cards(temp_screen, card_positions, selected_cards)
                            draw_close_card(temp_screen)
                            draw_back_card(temp_screen)
                            draw_bocina_card(temp_screen, muted=bocina_muted)
                            cards_screen = temp_screen
                            cv2.imshow(window_name, cards_screen)
                            continue
                        
                        # Verificar si se tocó la card de retroceso (flecha)
                        if detectar_back_card_touch(x_touch, y_touch):
                                print("Card de retroceso (flecha) tocada - Volviendo a la vista anterior")
                                
                                # Efecto visual de elevación (animación)
                                for frame_num in range(10):
                                    temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                    # Redibujar fondo
                                    for y in range(view_height):
                                        ratio = y / view_height
                                        r = int(255 * (0.3 + 0.4 * ratio))
                                        g = int(200 * (0.5 + 0.3 * ratio))
                                        b = int(255 * (0.8 - 0.3 * ratio))
                                        temp_screen[y, :] = [b, g, r]
                                    
                                    # Redibujar logo
                                    draw_logo_func(temp_screen)
                                    
                                    # Redibujar título
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                    
                                    # Redibujar cards
                                    draw_cards(temp_screen, card_positions, selected_cards)
                                    
                                    # Dibujar cards con efecto de elevación
                                    draw_close_card(temp_screen, elevated=False)
                                    if frame_num >= 5:
                                        draw_back_card(temp_screen, elevated=True)
                                    else:
                                        draw_back_card(temp_screen, elevated=False)
                                    
                                    # Dibujar card de bocina
                                    draw_bocina_card(temp_screen, muted=bocina_muted)
                                    
                                    cv2.imshow(window_name, temp_screen)
                                    cv2.waitKey(30)
                                
                                return "BACK"  # Retornar "BACK" para volver a la vista anterior
                        
                        # Verificar si se tocó la card de cerrar (X)
                        if detectar_close_card_touch(x_touch, y_touch):
                            print("Card de cerrar (X) tocada - Volviendo al menú principal")
                            
                            # Efecto visual de elevación (animación)
                            for frame_num in range(10):
                                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                # Redibujar fondo
                                for y in range(view_height):
                                    ratio = y / view_height
                                    r = int(255 * (0.3 + 0.4 * ratio))
                                    g = int(200 * (0.5 + 0.3 * ratio))
                                    b = int(255 * (0.8 - 0.3 * ratio))
                                    temp_screen[y, :] = [b, g, r]
                                
                                # Redibujar logo
                                draw_logo_func(temp_screen)
                                
                                # Redibujar título
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                
                                # Redibujar cards
                                draw_cards(temp_screen, card_positions, selected_cards)
                                
                                # Dibujar cards con efecto de elevación
                                draw_back_card(temp_screen, elevated=False)
                                if frame_num >= 5:
                                    draw_close_card(temp_screen, elevated=True)
                                else:
                                    draw_close_card(temp_screen, elevated=False)
                                
                                # Dibujar card de bocina
                                draw_bocina_card(temp_screen, muted=bocina_muted)
                                
                                cv2.imshow(window_name, temp_screen)
                                cv2.waitKey(30)
                            
                            return None  # Retornar None para volver al menú principal
                        
                        # Detectar card tocada
                        card_tocada = detectar_card_seleccionada(x_touch, y_touch, card_positions, selected_cards)
                        if card_tocada:
                            current_time = time.time()
                            
                            # Verificar debounce (evitar múltiples selecciones muy rápidas)
                            if card_tocada not in last_touch_time or (current_time - last_touch_time[card_tocada]) > debounce_time:
                                last_touch_time[card_tocada] = current_time
                                
                                if card_tocada in selected_cards:
                                    # Deseleccionar si ya está seleccionada
                                    selected_cards.remove(card_tocada)
                                    print(f"Escenario deseleccionado: {card_tocada}. Total: {len(selected_cards)}/{num_escenarios}")
                                else:
                                    # Seleccionar si aún no se ha alcanzado el límite
                                    if len(selected_cards) < num_escenarios:
                                        selected_cards.append(card_tocada)
                                        print(f"Escenario seleccionado: {card_tocada}. Total: {len(selected_cards)}/{num_escenarios}")
                                        
                                        # Si ya se seleccionaron todos los necesarios
                                        if len(selected_cards) == num_escenarios:
                                            print(f"¡Se han seleccionado {num_escenarios} escenario(s)!")
                                            
                                            # Redibujar la pantalla con las cards seleccionadas (borde verde) antes de cambiar de vista
                                            for y in range(view_height):
                                                ratio = y / view_height
                                                r = int(255 * (0.3 + 0.4 * ratio))
                                                g = int(200 * (0.5 + 0.3 * ratio))
                                                b = int(255 * (0.8 - 0.3 * ratio))
                                                cards_screen[y, :] = [b, g, r]
                                            draw_logo_func(cards_screen)
                                            cv2.putText(cards_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                                       font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                            cv2.putText(cards_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                                       font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                            draw_cards(cards_screen, card_positions, selected_cards)
                                            draw_close_card(cards_screen)
                                            draw_back_card(cards_screen)
                                            draw_bocina_card(cards_screen, muted=bocina_muted)
                                            cv2.imshow(window_name, cards_screen)
                                            
                                            # Esperar un momento para mostrar las cards seleccionadas con borde verde
                                            cv2.waitKey(800)  # 800ms de pausa antes de cambiar de vista
                                            
                                            # Mostrar vista con rectángulos para los escenarios seleccionados
                                            resultado = mostrar_vista_rectangulos_escenarios(
                                                device, coordenadas, dmax_map, dmin_map, draw_logo_func, 
                                                selected_cards, existing_window_name=window_name
                                            )
                                            # Si se presionó la card de retroceso, retornar None para volver al menú principal
                                            if resultado == "MAIN_MENU":
                                                return None  # Retornar None para que mostrar_seleccion_niveles_clasificacion también retorne None
                                            # Si se presionó la card de cerrar (flecha), retornar None para volver a la vista anterior
                                            if resultado is None:
                                                # Desmarcar todos los escenarios seleccionados
                                                selected_cards = []
                                                print("Escenarios desmarcados - Volviendo a la selección de escenarios")
                                                
                                                # Redibujar la pantalla de selección de escenarios
                                                for y in range(view_height):
                                                    ratio = y / view_height
                                                    r = int(255 * (0.3 + 0.4 * ratio))
                                                    g = int(200 * (0.5 + 0.3 * ratio))
                                                    b = int(255 * (0.8 - 0.3 * ratio))
                                                    cards_screen[y, :] = [b, g, r]
                                                draw_logo_func(cards_screen)
                                                cv2.putText(cards_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                                cv2.putText(cards_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                                draw_cards(cards_screen, card_positions, selected_cards)
                                                draw_close_card(cards_screen)
                                                draw_back_card(cards_screen)
                                                cv2.imshow(window_name, cards_screen)
                                                continue  # Continuar el bucle para permitir más selecciones
                                            return selected_cards
                                    
                                    else:
                                        print(f"Ya has seleccionado {num_escenarios} escenario(s). Deselecciona uno primero.")
                                
                                # Redibujar las cards con el estado actualizado
                                # Redibujar fondo
                                for y in range(view_height):
                                    ratio = y / view_height
                                    r = int(255 * (0.3 + 0.4 * ratio))
                                    g = int(200 * (0.5 + 0.3 * ratio))
                                    b = int(255 * (0.8 - 0.3 * ratio))
                                    cards_screen[y, :] = [b, g, r]
                                
                                # Redibujar logo
                                draw_logo_func(cards_screen)
                                
                                # Redibujar título
                                cv2.putText(cards_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                cv2.putText(cards_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                
                                # Redibujar cards con estado actualizado
                                draw_cards(cards_screen, card_positions, selected_cards)
                                draw_close_card(cards_screen)
                                draw_back_card(cards_screen)
            
            # Redibujar cards en cada frame
            draw_close_card(cards_screen)
            draw_back_card(cards_screen)
            draw_bocina_card(cards_screen, muted=bocina_muted)
            cv2.imshow(window_name, cards_screen)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return selected_cards if len(selected_cards) == num_escenarios else None
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return selected_cards if len(selected_cards) == num_escenarios else None


def mostrar_vista_rectangulos_escenarios(device, coordenadas, dmax_map, dmin_map, draw_logo_func, escenarios_seleccionados, existing_window_name=None):
    """
    Muestra una vista con rectángulos vacíos donde irán las imágenes de los escenarios seleccionados.
    Los rectángulos están centrados y son del mismo tamaño.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        escenarios_seleccionados: Lista con los nombres de los escenarios seleccionados
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        None
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
    view_width = VIEW_WIDTH
    view_height = VIEW_HEIGHT
    
    # Número de escenarios seleccionados
    num_escenarios = len(escenarios_seleccionados)
    
    # Mapeo de escenarios a imágenes
    escenario_images = {
        "Escenario 1": "images/EscenarioGranja.png",
        "Escenario 2": "images/EscenarioCalle.png",
        "Escenario 3": "images/EscenarioRestaurante.png",
        "Escenario 4": "images/EscenarioRopa.png",
        "Escenario 5": "images/EscenarioColegio.png"
    }
    
    # Inicializar detector de objetos YOLO
    detector = ObjectDetector()
    
    # Cargar imágenes de los escenarios seleccionados
    loaded_escenario_images = {}
    for escenario in escenarios_seleccionados:
        if escenario in escenario_images:
            path = escenario_images[escenario]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    loaded_escenario_images[escenario] = img
                else:
                    print(f"Error: No se pudo cargar la imagen para {escenario} desde {path}")
            else:
                print(f"Advertencia: No se encontró la imagen para {escenario} en {path}")
    
    # Crear fondo
    rectangulos_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        rectangulos_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(rectangulos_screen)
    
    # Función para dibujar card redonda con X (estilo infantil)
    def draw_close_card(screen, elevated=False):
        """
        Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
        """
        # Posición base del lado derecho (misma posición que en otras vistas)
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
    # Usar un área rectangular para la detección, similar a las otras cards
    close_card_radius_rect = 50
    close_card_margin_x_rect = 180
    close_card_margin_y_rect = 80
    close_card_center_x_rect = view_width - close_card_margin_x_rect - close_card_radius_rect
    close_card_center_y_rect = close_card_margin_y_rect + close_card_radius_rect
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_rect = close_card_radius_rect * 2.4  # Área más grande para facilitar el toque
    close_card_detection_x_rect = close_card_center_x_rect - close_card_radius_rect * 1.2
    close_card_detection_y_rect = close_card_center_y_rect - close_card_radius_rect * 1.2
    close_card_detection_w_rect = close_card_detection_size_rect
    close_card_detection_h_rect = close_card_detection_size_rect
    
    # Función para detectar si se tocó la card de cerrar (usando área rectangular como las otras cards)
    def detectar_close_card_touch_rect(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar (usando área rectangular)"""
        return (close_card_detection_x_rect <= x_touch <= close_card_detection_x_rect + close_card_detection_w_rect and
                close_card_detection_y_rect <= y_touch <= close_card_detection_y_rect + close_card_detection_h_rect)
    
    # Función para dibujar card redonda con flecha hacia la izquierda (estilo infantil)
    def draw_back_card(screen, elevated=False):
        """
        Dibuja una card redonda con flecha hacia la izquierda en el centro, estilo infantil, azul con flecha blanca
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
        """
        # Posición base del lado izquierdo
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
        # Hacer el color más brillante si está elevada
        color_intensity = 1.15 if elevated else 1.0
        base_blue_light = int(100 * color_intensity)
        base_blue_medium = int(50 * color_intensity)
        base_blue_dark = int(30 * color_intensity)
        # Limitar valores a 255
        base_blue_light = min(255, base_blue_light)
        base_blue_medium = min(255, base_blue_medium)
        base_blue_dark = min(255, base_blue_dark)
        
        # Círculo exterior más claro
        cv2.circle(screen, card_center, card_radius, (255, base_blue_light, base_blue_light), -1)  # Azul pastel claro
        # Círculo interior más intenso
        cv2.circle(screen, card_center, int(card_radius * 0.85), (255, base_blue_medium, base_blue_medium), -1)  # Azul pastel medio
        # Círculo más interno
        cv2.circle(screen, card_center, int(card_radius * 0.7), (255, base_blue_dark, base_blue_dark), -1)  # Azul más intenso
        
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
    
    # Variables para la card de retroceso (necesarias para la detección)
    back_card_radius_rect = 50
    back_card_margin_x_rect = 180
    back_card_margin_y_rect = 80
    back_card_center_x_rect = back_card_margin_x_rect + back_card_radius_rect
    back_card_center_y_rect = back_card_margin_y_rect + back_card_radius_rect
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    back_card_detection_size_rect = back_card_radius_rect * 2.4  # Área más grande para facilitar el toque
    back_card_detection_x_rect = back_card_center_x_rect - back_card_radius_rect * 1.2
    back_card_detection_y_rect = back_card_center_y_rect - back_card_radius_rect * 1.2
    back_card_detection_w_rect = back_card_detection_size_rect
    back_card_detection_h_rect = back_card_detection_size_rect
    
    # Función para detectar si se tocó la card de retroceso (usando área rectangular como las otras cards)
    def detectar_back_card_touch_rect(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de retroceso (usando área rectangular)"""
        return (back_card_detection_x_rect <= x_touch <= back_card_detection_x_rect + back_card_detection_w_rect and
                back_card_detection_y_rect <= y_touch <= back_card_detection_y_rect + back_card_detection_h_rect)
    
    # Título
    titulo_texto = f"Escenarios seleccionados: {num_escenarios}"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 180
    # Sombra del título
    cv2.putText(rectangulos_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    cv2.putText(rectangulos_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Calcular el área visible de la cámara (donde realmente puede detectar toques)
    visible_area_width = xv_max - xv_min
    visible_area_height = yv_max - yv_min
    
    # Dimensiones de los rectángulos (aumentadas pero ajustadas al área visible)
    # Tamaño base más grande
    base_rect_width = 450  # Aumentado para hacer las cards más anchas
    base_rect_height = 480
    rect_spacing = 70  # Espacio entre rectángulos inicial (aumentado para más separación)
    
    # Margen de seguridad desde los bordes del área visible (para evitar problemas en los bordes)
    margin_x = 30  # Margen horizontal mínimo (aumentado para evitar el problema del borde derecho)
    margin_y = 30  # Margen vertical mínimo
    
    # Calcular ancho disponible dentro del área visible (con márgenes en ambos lados)
    available_width = visible_area_width - 2 * margin_x
    available_height = visible_area_height - 2 * margin_y
    
    # Ajustar el tamaño de las cards para que quepan en el área visible
    # Empezar con el tamaño base y ajustar si es necesario
    rect_width = base_rect_width
    rect_height = base_rect_height
    
    # Ajustar altura si es necesario para que quepa en el área visible
    if rect_height > available_height:
        rect_height = available_height
    
    # Calcular el ancho total necesario con el spacing inicial
    total_width_needed = num_escenarios * rect_width + (num_escenarios - 1) * rect_spacing
    
    # Si no caben, ajustar el spacing (y si es necesario, reducir el ancho de los rectángulos)
    if total_width_needed > available_width:
        if num_escenarios > 1:
            # Calcular el spacing máximo permitido
            max_spacing = (available_width - num_escenarios * rect_width) / (num_escenarios - 1)
            # Aceptar spacing mínimo de 30 píxeles (aumentado para más separación)
            if max_spacing >= 30:
                rect_spacing = int(max_spacing)
            else:
                # Si ni siquiera con spacing mínimo caben, reducir el ancho de los rectángulos
                # Calcular el ancho máximo permitido para cada rectángulo
                max_rect_width = (available_width - (num_escenarios - 1) * 30) / num_escenarios
                rect_width = int(max_rect_width)
                rect_spacing = 30
        else:
            # Solo un rectángulo, ajustar su ancho si es necesario
            if rect_width > available_width:
                rect_width = available_width
    
    # Recalcular el ancho total con los valores ajustados
    total_width = num_escenarios * rect_width + (num_escenarios - 1) * rect_spacing
    
    # Calcular start_x centrado dentro del área visible, asegurando que quepa todo con márgenes
    # Usar xv_min como referencia en lugar de 0
    start_x = xv_min + margin_x + (available_width - total_width) // 2
    
    # Asegurar que el primer rectángulo tenga al menos el margen mínimo desde xv_min
    if start_x < xv_min + margin_x:
        start_x = xv_min + margin_x
    
    # Verificar que el último rectángulo no choque con el borde derecho del área visible
    # El último rectángulo termina en: start_x + (num_escenarios - 1) * (rect_width + rect_spacing) + rect_width
    last_rect_end = start_x + (num_escenarios - 1) * (rect_width + rect_spacing) + rect_width
    
    # Si el último rectángulo se sale del área visible, ajustar start_x hacia la izquierda
    if last_rect_end > xv_max - margin_x:
        start_x = xv_max - margin_x - (num_escenarios - 1) * (rect_width + rect_spacing) - rect_width
        # Asegurar que no se salga por la izquierda
        start_x = max(xv_min + margin_x, start_x)
    
    # Calcular start_y más abajo para evitar choque con el título
    # El título está en y=180, así que empezamos desde y=250 para dejar espacio
    start_y_base = 250  # Posición base más abajo para evitar el título
    # Calcular start_y considerando el área visible pero empezando más abajo
    start_y = max(start_y_base, yv_min + margin_y + (available_height - rect_height) // 2)
    # Asegurar que no se salga del área visible
    start_y = max(start_y_base, min(start_y, yv_max - margin_y - rect_height))
    
    # Crear diccionario con las posiciones de los rectángulos
    rectangulos_positions = {}
    for idx, escenario in enumerate(escenarios_seleccionados):
        x = start_x + idx * (rect_width + rect_spacing)
        y = start_y
        rectangulos_positions[escenario] = {
            'x': x,
            'y': y,
            'width': rect_width,
            'height': rect_height,
            'nombre': escenario
        }
    
    # Función para dibujar los rectángulos
    def draw_rectangulos(screen, rectangulos_positions):
        for escenario, pos in rectangulos_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            
            # Dibujar sombra
            shadow_offset = 10
            shadow_color = (50, 50, 50)
            cv2.rectangle(screen, 
                        (x + shadow_offset, y + shadow_offset), 
                        (x + w + shadow_offset, y + h + shadow_offset), 
                        shadow_color, -1)
            
            # Dibujar rectángulo principal (fondo blanco/gris claro)
            rect_color = (240, 240, 240)  # Gris muy claro
            cv2.rectangle(screen, (x, y), (x + w, y + h), rect_color, -1)
            
            # Dibujar la imagen del escenario si está disponible
            if escenario in loaded_escenario_images:
                img = loaded_escenario_images[escenario]
                img_h, img_w = img.shape[:2]
                
                # Todas las imágenes ocupan todo el tamaño de la card
                # Usar todo el espacio del rectángulo
                img_area_height = h
                img_area_width = w
                
                # Calcular el factor de escala para que la imagen llene todo el espacio
                scale_w = img_area_width / img_w
                scale_h = img_area_height / img_h
                scale = max(scale_w, scale_h)  # Usar el mayor para llenar todo el espacio
                
                # Redimensionar la imagen
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Calcular posición para centrar la imagen en el rectángulo
                img_x = x + (w - new_w) // 2
                img_y = y + (h - new_h) // 2
                
                # Recortar si es necesario para que quepa exactamente en el rectángulo
                if new_w > w or new_h > h:
                    # Calcular el área de recorte
                    crop_x = max(0, (new_w - w) // 2)
                    crop_y = max(0, (new_h - h) // 2)
                    crop_w = min(w, new_w)
                    crop_h = min(h, new_h)
                    
                    resized_img = resized_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    img_x = x
                    img_y = y
                    new_w = crop_w
                    new_h = crop_h
                
                # Aplicar transparencia del 50% a todas las imágenes
                opacity = 0.5  # 50% de transparencia
                
                if resized_img.shape[2] == 4:
                    # Si la imagen tiene canal alfa, combinar con la transparencia del 50%
                    b, g, r, a_original = cv2.split(resized_img)
                    # Normalizar el canal alfa original
                    a_original = a_original.astype(np.float32) / 255.0
                    # Combinar la transparencia original con el 50%
                    a_combined = a_original * opacity
                    
                    # Aplicar la imagen con transparencia combinada
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - a_combined) +
                            resized_img[:, :, c] * a_combined
                        ).astype(np.uint8)
                else:
                    # Si no tiene canal alfa, aplicar transparencia del 50% directamente
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - opacity) +
                            resized_img[:, :, c] * opacity
                        ).astype(np.uint8)
            
            # Dibujar borde del rectángulo
            border_color = (150, 150, 150)  # Gris medio
            border_thickness = 4
            cv2.rectangle(screen, (x, y), (x + w, y + h), border_color, border_thickness)
    
    # Dibujar los rectángulos inicialmente
    draw_rectangulos(rectangulos_screen, rectangulos_positions)
    # Dibujar card de cerrar (X roja)
    draw_close_card(rectangulos_screen)
    # Dibujar card de retroceso (flecha azul)
    draw_back_card(rectangulos_screen)
    
    # Usar el nombre de ventana existente si se proporciona, o crear uno nuevo
    window_name = existing_window_name if existing_window_name else "Vista de Escenarios"
    
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
        cv2.moveWindow(window_name, SCREEN_OFFSET_X, SCREEN_OFFSET_Y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar en pantalla (transición suave sin cerrar)
    cv2.imshow(window_name, rectangulos_screen)
    
    # Iniciar streams de cámara (aunque no se usen para interacción, mantener consistencia)
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    # Inicializar detecciones vacías
    detections = []
    
    try:
        # Mostrar la vista indefinidamente hasta que se presione 'q' o se toque la card de cerrar
        while True:
            frame = rgb_stream.read_frame()
            depth_frame = depth_stream.read_frame()
            
            if frame is None or depth_frame is None:
                continue
            
            rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            bgr_data = cv2.flip(bgr_data, 1)
            
            # Realizar detección de objetos
            detections = detector.detect(bgr_data)
            bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]
            
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
            
            # Crear la máscara de toques
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Procesar el contorno más grande (evitar múltiples detecciones)
            if contours:
                # Ordenar por área y tomar el más grande
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 50:
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Primero verificar si se tocó la card de cerrar (X) - lleva al menú principal
                        if detectar_close_card_touch_rect(x_touch, y_touch):
                                print("Card de cerrar (X) tocada - Volviendo al menú principal")
                                
                                # Efecto visual de elevación (animación)
                                for frame_num in range(10):
                                    temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                    # Redibujar fondo
                                    for y in range(view_height):
                                        ratio = y / view_height
                                        r = int(255 * (0.3 + 0.4 * ratio))
                                        g = int(200 * (0.5 + 0.3 * ratio))
                                        b = int(255 * (0.8 - 0.3 * ratio))
                                        temp_screen[y, :] = [b, g, r]
                                    
                                    # Redibujar logo
                                    draw_logo_func(temp_screen)
                                    
                                    # Redibujar título
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                    cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                    
                                    # Redibujar rectángulos
                                    draw_rectangulos(temp_screen, rectangulos_positions)
                                    
                                    # Dibujar cards con efecto de elevación
                                    draw_back_card(temp_screen, elevated=False)
                                    if frame_num >= 5:
                                        draw_close_card(temp_screen, elevated=True)
                                    else:
                                        draw_close_card(temp_screen, elevated=False)
                                    
                                    cv2.imshow(window_name, temp_screen)
                                    cv2.waitKey(30)
                                
                                return "MAIN_MENU"  # Retornar "MAIN_MENU" para volver al menú principal
                        
                        # Verificar si se tocó la card de retroceso (flecha) - retrocede a la vista anterior
                        if detectar_back_card_touch_rect(x_touch, y_touch):
                            print("Card de retroceso (flecha) tocada - Volviendo a la vista anterior")
                            
                            # Efecto visual de elevación (animación)
                            for frame_num in range(10):
                                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                                # Redibujar fondo
                                for y in range(view_height):
                                    ratio = y / view_height
                                    r = int(255 * (0.3 + 0.4 * ratio))
                                    g = int(200 * (0.5 + 0.3 * ratio))
                                    b = int(255 * (0.8 - 0.3 * ratio))
                                    temp_screen[y, :] = [b, g, r]
                                
                                # Redibujar logo
                                draw_logo_func(temp_screen)
                                
                                # Redibujar título
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                                           font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
                                cv2.putText(temp_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                                           font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
                                
                                # Redibujar rectángulos
                                draw_rectangulos(temp_screen, rectangulos_positions)
                                
                                # Dibujar cards con efecto de elevación
                                draw_close_card(temp_screen, elevated=False)
                                if frame_num >= 5:
                                    draw_back_card(temp_screen, elevated=True)
                                else:
                                    draw_back_card(temp_screen, elevated=False)
                                
                                cv2.imshow(window_name, temp_screen)
                                cv2.waitKey(30)
                            
                            return None  # Retornar None para volver a la vista anterior
            
            # Redibujar todo en cada frame (incluyendo la card de cerrar)
            # Redibujar fondo
            for y in range(view_height):
                ratio = y / view_height
                r = int(255 * (0.3 + 0.4 * ratio))
                g = int(200 * (0.5 + 0.3 * ratio))
                b = int(255 * (0.8 - 0.3 * ratio))
                rectangulos_screen[y, :] = [b, g, r]
            
            # Redibujar logo
            draw_logo_func(rectangulos_screen)
            
            # Redibujar título
            cv2.putText(rectangulos_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                       font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
            cv2.putText(rectangulos_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                       font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
            
            # Redibujar rectángulos
            draw_rectangulos(rectangulos_screen, rectangulos_positions)
            
            # Redibujar cards
            draw_close_card(rectangulos_screen)
            draw_back_card(rectangulos_screen)
            
            # Dibujar efectos de brillo para objetos detectados en el escenario correcto
            for det in detections:
                categoria_objetivo = det["category"]
                if categoria_objetivo in rectangulos_positions:
                    pos = rectangulos_positions[categoria_objetivo]
                    
                    # Calcular centro del objeto detectado en la cámara
                    x1, y1, x2, y2 = det["bbox"]
                    cx_cam = int((x1 + x2) / 2)
                    cy_cam = int((y1 + y2) / 2)
                    
                    # Mapear a coordenadas de proyección
                    x_proj, y_proj = detector.map_coordinates(cx_cam, cy_cam, coordenadas)
                    
                    # Verificar si el objeto está dentro de su escenario objetivo
                    rx, ry, rw, rh = pos['x'], pos['y'], pos['width'], pos['height']
                    if rx <= x_proj <= rx + rw and ry <= y_proj <= ry + rh:
                        # Dibujar efecto de brillo (un poco más grande que el punto)
                        draw_shine_effect(rectangulos_screen, x_proj - 40, y_proj - 40, 80, 80)
                        
                        # Opcional: Mostrar el nombre del objeto detectado
                        label = det["label"]
                        cv2.putText(rectangulos_screen, label, (x_proj - 30, y_proj - 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, rectangulos_screen)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return None
