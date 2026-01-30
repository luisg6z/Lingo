import cv2
import numpy as np
import os
import time


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
    view_width = 1280
    view_height = 800
    
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
        cv2.moveWindow(window_name, 1920, 0)
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
            nivel_seleccionado_flag = False
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
                        
                        # Detectar si se seleccionó un nivel
                        if not nivel_seleccionado_flag:
                            nivel_sel = detectar_nivel_seleccionado(x_touch, y_touch, nivel_positions)
                            if nivel_sel:
                                print(f"Nivel seleccionado: {nivel_sel}")
                                
                                # Animación de elevación
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
                                    
                                    if frame_num >= 5:
                                        draw_nivel_cards(temp_screen, nivel_positions, elevated_card=nivel_sel)
                                    else:
                                        draw_nivel_cards(temp_screen, nivel_positions)
                                    
                                    cv2.imshow(window_name, temp_screen)
                                    cv2.waitKey(30)
                                
                                nivel_seleccionado = nivel_sel
                                nivel_seleccionado_flag = True
                                
                                # Mantener elevada
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
                                cv2.imshow(window_name, temp_screen)
                                cv2.waitKey(500)
                                
                                # Convertir el nivel seleccionado a número
                                num_escenarios = int(nivel_seleccionado)
                                
                                # Pasar el nombre de la ventana existente para reutilizarla
                                # Mostrar vista de 8 cards (reutilizará la misma ventana)
                                cards_seleccionadas = mostrar_vista_8_cards(
                                    device, coordenadas, dmax_map, dmin_map, draw_logo_func, num_escenarios, 
                                    existing_window_name=window_name
                                )
                                
                                return nivel_seleccionado
            
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


def mostrar_vista_8_cards(device, coordenadas, dmax_map, dmin_map, draw_logo_func, num_escenarios, existing_window_name=None):
    """
    Muestra la vista con 5 cards donde el usuario debe seleccionar N cards según num_escenarios.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        num_escenarios: Número de cards que debe seleccionar (1-3)
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        list: Lista con los nombres de las cards seleccionadas
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
        "Escenario 1": "images/Escenario1.png",      # Nota: sin espacio en el nombre del archivo
        "Escenario 2": "images/Escenario 2.png",
        "Escenario 3": "images/Escenario 3.png",
        "Escenario 4": "images/Escenario 4.png",
        "Escenario 5": "images/Escenario 5.png"
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
    card_width = 300  # Reducido de 320 a 300
    card_height = 270  # Reducido de 290 a 270
    card_spacing_x = 40  # Mantener espacio entre cards
    card_spacing_y = 30  # Reducido de 35 a 30 para que quepan mejor
    
    # Calcular posiciones (grid 3x2: 3 cards arriba, 2 cards abajo)
    # Fila superior: 3 cards
    total_width_top = 3 * card_width + 2 * card_spacing_x
    # Fila inferior: 2 cards (centradas)
    total_width_bottom = 2 * card_width + 1 * card_spacing_x
    total_height = 2 * card_height + card_spacing_y
    start_x_top = (view_width - total_width_top) // 2
    start_x_bottom = (view_width - total_width_bottom) // 2
    start_y = 240  # Subido de 260 a 240 para evitar que se corten las cards de abajo
    
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
    draw_cards(cards_screen, card_positions)
    
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
    
    # Variables de selección
    selected_cards = []
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
                                            # Mostrar vista con rectángulos para los escenarios seleccionados
                                            mostrar_vista_rectangulos_escenarios(
                                                device, coordenadas, dmax_map, dmin_map, draw_logo_func, 
                                                selected_cards, existing_window_name=window_name
                                            )
                                            return selected_cards
                                    
                                    else:
                                        print(f"Ya has seleccionado {num_escenarios} escenario(s). Deselecciona uno primero.")
                                
                                # Redibujar con las selecciones actualizadas (inmediatamente, sin pausa)
                                cards_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
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
                            cv2.imshow(window_name, cards_screen)
            
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
    view_width = 1280
    view_height = 800
    
    # Número de escenarios seleccionados
    num_escenarios = len(escenarios_seleccionados)
    
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
            
            # Dibujar borde del rectángulo
            border_color = (150, 150, 150)  # Gris medio
            border_thickness = 4
            cv2.rectangle(screen, (x, y), (x + w, y + h), border_color, border_thickness)
            
            # Dibujar nombre del escenario en la parte inferior del rectángulo
            nombre_texto = pos['nombre']
            font_nombre = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nombre = 0.7
            thickness_nombre = 2
            
            # Calcular el ancho disponible
            available_width = w - 20
            
            # Verificar si el texto cabe
            nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            while nombre_size[0] > available_width and font_scale_nombre > 0.4:
                font_scale_nombre -= 0.1
                nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            
            # Calcular posición del texto (centrado horizontalmente, en la parte inferior)
            nombre_x = x + (w - nombre_size[0]) // 2
            nombre_y = y + h - 15  # Un poco arriba del borde inferior
            
            # Dibujar sombra del texto
            cv2.putText(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                       font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
            # Dibujar texto principal (negro)
            cv2.putText(screen, nombre_texto, (nombre_x, nombre_y), 
                       font_nombre, font_scale_nombre, (50, 50, 50), thickness_nombre)
    
    # Dibujar los rectángulos inicialmente
    draw_rectangulos(rectangulos_screen, rectangulos_positions)
    
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
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar en pantalla (transición suave sin cerrar)
    cv2.imshow(window_name, rectangulos_screen)
    
    # Iniciar streams de cámara (aunque no se usen para interacción, mantener consistencia)
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    try:
        # Mostrar la vista indefinidamente hasta que se presione 'q'
        while True:
            cv2.imshow(window_name, rectangulos_screen)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return None
