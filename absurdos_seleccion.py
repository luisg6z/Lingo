import cv2
import numpy as np
import os
import time

# --- CONFIGURACIÓN DE PANTALLA ---
# Si usas un segundo monitor o videobeam, ajusta SCREEN_OFFSET_X al ancho de tu pantalla principal (ej: 1920)
SCREEN_OFFSET_X = 1920 
SCREEN_OFFSET_Y = 0

# Resolución del videobeam/segunda pantalla (ajusta si no se ve a pantalla completa)
# Comúnmente 1280x800, 1920x1080, etc.
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
# --------------------------------


def mostrar_seleccion_absurdos(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de tipo de absurdos (Visuales o Auditivos).
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        str o None: Tipo de absurdo seleccionado ("Absurdos Visuales" o "Absurdos Auditivos") o None si se canceló
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
    
    # Resolución del videobeam (segunda pantalla)
    VIDEOBEAM_WIDTH = 1920
    VIDEOBEAM_HEIGHT = 1080
    
    def scale_to_videobeam(image, source_width=1280, source_height=800):
        """
        Escala una imagen de la resolución fuente a la resolución del videobeam.
        
        Args:
            image: Imagen a escalar (numpy array)
            source_width: Ancho de la imagen fuente (default: 1280)
            source_height: Alto de la imagen fuente (default: 800)
        
        Returns:
            Imagen escalada a la resolución del videobeam
        """
        if image is None or image.size == 0:
            return image
        return cv2.resize(image, (VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Crear fondo
    absurdos_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        absurdos_screen[y, :] = [b, g, r]
    
    # Dibujar el logo
    draw_logo_func(absurdos_screen)
    
    # Cargar imágenes de los tipos de absurdos
    tipo_images = {}
    tipo_paths = {
        "Absurdos Visuales": "images/Nivel1absurdovisuales.png",
        "Absurdos Auditivos": "images/Nivel2absurdovsuales.png"
    }
    
    for tipo, path in tipo_paths.items():
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                tipo_images[tipo] = img
                print(f"✓ Imagen cargada para {tipo}: {path}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {tipo}: {path}")
                tipo_images[tipo] = None
        else:
            tipo_images[tipo] = None
            if path:
                print(f"⚠ Archivo no encontrado: {path}")
    
    # Título
    titulo_texto = "Elige el tipo de absurdos"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 180
    # Sombra del título
    cv2.putText(absurdos_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_titulo, font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    cv2.putText(absurdos_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_titulo, font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Dimensiones de las cards (2 cards centradas)
    card_width = 400
    card_height = 450
    card_spacing = 80
    
    # Calcular posiciones (centradas, 2 cards en fila)
    total_width = 2 * card_width + card_spacing
    start_x = (view_width - total_width) // 2
    start_y = 250  # Debajo del título
    
    tipo_positions = {}
    for idx, (tipo, path) in enumerate(tipo_paths.items()):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        tipo_positions[tipo] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'imagen': tipo_images.get(tipo),
            'color': (100, 255, 150) if tipo == "Absurdos Visuales" else (150, 200, 255),  # Verde claro o azul claro
            'icono': "👁️" if tipo == "Absurdos Visuales" else "🔊"
        }
    
    # Función para dibujar las cards de tipos de absurdos
    def draw_tipo_cards(screen, tipo_positions, elevated_card=None):
        for nombre, pos in tipo_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            color = pos['color']
            
            # Efecto de elevación si esta card está seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 8
            
            if elevated_card == nombre:
                elevation_offset = -20
                scale_factor = 1.05
                shadow_offset_base = 15
                # Hacer el color más brillante cuando está elevada
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
            shadow_color = (50, 50, 50)
            cv2.rectangle(screen, (x_scaled + shadow_offset, y_scaled + shadow_offset), 
                        (x_scaled + w_scaled + shadow_offset, y_scaled + h_scaled + shadow_offset), 
                        shadow_color, -1)
            
            # Dibujar la card
            if pos['imagen'] is not None:
                # Usar la imagen como fondo completo de la card
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
            else:
                # Card de color sólido si no hay imagen
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
                
                # Borde de la card
                border_color = tuple(max(0, c - 30) for c in color)
                border_thickness = 7 if elevated_card == nombre else 5
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
                
                # Dibujar icono emoji
                icono = pos['icono']
                font_icon = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_icon = 4.0 * scale_factor
                thickness_icon = int(3 * scale_factor)
                icon_size, _ = cv2.getTextSize(icono, font_icon, font_scale_icon, thickness_icon)
                icon_x = x_scaled + (w_scaled - icon_size[0]) // 2
                icon_y = y_scaled + int(150 * scale_factor)
                cv2.putText(screen, icono, (icon_x, icon_y), font_icon, font_scale_icon, (255, 255, 255), thickness_icon)
            
            # Dibujar nombre del tipo
            nombre_texto = nombre
            font_nombre = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nombre = 1.0
            thickness_nombre = 2
            
            # Calcular el ancho disponible
            available_width = w_scaled - 40
            
            # Verificar si el texto cabe
            nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            while nombre_size[0] > available_width and font_scale_nombre > 0.5:
                font_scale_nombre -= 0.1
                nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            
            # Si no cabe, dividir en múltiples líneas
            if nombre_size[0] > available_width:
                palabras = nombre_texto.split()
                lineas = []
                linea_actual = ""
                for palabra in palabras:
                    test_linea = linea_actual + (" " if linea_actual else "") + palabra
                    test_size, _ = cv2.getTextSize(test_linea, font_nombre, font_scale_nombre, thickness_nombre)
                    if test_size[0] <= available_width:
                        linea_actual = test_linea
                    else:
                        if linea_actual:
                            lineas.append(linea_actual)
                        linea_actual = palabra
                if linea_actual:
                    lineas.append(linea_actual)
                
                # Dibujar cada línea centrada
                line_height = nombre_size[1] + 5
                start_y_text = y_scaled + int(350 * scale_factor) - (len(lineas) - 1) * line_height // 2
                for i, linea in enumerate(lineas):
                    linea_size, _ = cv2.getTextSize(linea, font_nombre, font_scale_nombre, thickness_nombre)
                    linea_x = x_scaled + (w_scaled - linea_size[0]) // 2
                    linea_y = start_y_text + i * line_height
                    # Sombra del texto
                    cv2.putText(screen, linea, (linea_x + 2, linea_y + 2), 
                               font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                    # Texto principal
                    cv2.putText(screen, linea, (linea_x, linea_y), 
                               font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            else:
                # El texto cabe en una línea
                nombre_x = x_scaled + (w_scaled - nombre_size[0]) // 2
                nombre_y = y_scaled + int(350 * scale_factor)
                # Sombra del texto
                cv2.putText(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                           font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                # Texto principal
                cv2.putText(screen, nombre_texto, (nombre_x, nombre_y), 
                           font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
    
    # Dibujar las cards inicialmente
    draw_tipo_cards(absurdos_screen, tipo_positions)
    
    # Usar el nombre de ventana existente si se proporciona, o crear uno nuevo
    window_name = existing_window_name if existing_window_name else "Selección de Absurdos"
    
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
        cv2.waitKey(50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
    
    # Escalar a la resolución del videobeam antes de mostrar
    absurdos_screen_scaled = scale_to_videobeam(absurdos_screen)
    # Mostrar en pantalla (transición suave sin cerrar)
    cv2.imshow(window_name, absurdos_screen_scaled)
    cv2.waitKey(50)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
    
    # Función para detectar tipo seleccionado
    def detectar_tipo_seleccionado(x_touch, y_touch, tipo_positions):
        for nombre, pos in tipo_positions.items():
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
    tipo_seleccionado_flag = False
    tipo_seleccionado = None
    
    try:
        while True:
            tipo_seleccionado_flag = False
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
                        
                        # Detectar si se seleccionó un tipo
                        if not tipo_seleccionado_flag:
                            tipo_sel = detectar_tipo_seleccionado(x_touch, y_touch, tipo_positions)
                            if tipo_sel:
                                print(f"Tipo de absurdo seleccionado: {tipo_sel}")
                                
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
                                        draw_tipo_cards(temp_screen, tipo_positions, elevated_card=tipo_sel)
                                    else:
                                        draw_tipo_cards(temp_screen, tipo_positions)
                                    
                                    temp_screen_scaled = scale_to_videobeam(temp_screen)
                                    cv2.imshow(window_name, temp_screen_scaled)
                                    cv2.waitKey(30)
                                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                
                                tipo_seleccionado = tipo_sel
                                tipo_seleccionado_flag = True
                                
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
                                draw_tipo_cards(temp_screen, tipo_positions, elevated_card=tipo_sel)
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
                                cv2.waitKey(500)
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                
                                return tipo_seleccionado
            
            # Escalar a la resolución del videobeam antes de mostrar
            absurdos_screen_scaled = scale_to_videobeam(absurdos_screen)
            cv2.imshow(window_name, absurdos_screen_scaled)
            # Forzar pantalla completa en cada frame
            cv2.waitKey(10)
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
            except:
                pass
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return None

