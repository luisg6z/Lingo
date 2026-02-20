"""
Level selection for classification game
"""
import cv2
import numpy as np
import os
import time
import pygame
import json
from openni import openni2
from detection_logic import ObjectDetector, draw_shine_effect
import pyttsx3

# Get project root and add to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sys
sys.path.insert(0, project_root)


# Variable global para mantener el estado del audio entre vistas
_bocina_muted_global = False
_background_music_loaded = False  # Flag para saber si la música ya está cargada

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
    
    # Escalar la imagen para que llene toda la pantalla del videobeam
    scaled_image = cv2.resize(image, (VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return scaled_image

def configurar_camara_kinect(color_stream):
    """
    Configura los parámetros de la cámara del Kinect:
    - Reduce la exposición EV (baja el valor de exposición)
    - Aumenta la velocidad de obturación (menor exposición = mayor velocidad)
    - Ajusta el balance de blanco
    
    Args:
        color_stream: Stream de color de OpenNI2
    """
    try:
        # Obtener los ajustes de la cámara
        camera_settings = openni2.CameraSettings(color_stream)
        
        # Desactivar auto exposición para control manual
        camera_settings.auto_exposure = False
        
        # Reducir la exposición (valores más bajos = menos exposición = mayor velocidad de obturación)
        # Rango típico: 0-100, valores más bajos = menos exposición
        # Reducir a aproximadamente 30-40% del valor máximo para bajar exposición
        try:
            # Intentar obtener el valor actual
            current_exposure = camera_settings.exposure
            # Reducir la exposición (bajar el valor)
            # Si el valor actual es alto, reducirlo significativamente
            new_exposure = max(10, int(current_exposure * 0.3))  # Reducir a 30% del valor actual, mínimo 10
            camera_settings.exposure = new_exposure
            print(f"✓ Exposición configurada: {new_exposure} (reducida desde {current_exposure})")
        except Exception as e:
            # Si no se puede obtener el valor actual, usar un valor fijo bajo
            camera_settings.exposure = 20  # Valor bajo para menos exposición
            print(f"✓ Exposición configurada: 20 (valor fijo)")
        
        # Configurar balance de blanco
        # Desactivar auto balance de blanco para control manual
        camera_settings.auto_white_balance = False
        
        # Ajustar ganancia si es necesario (puede ayudar con el balance de blanco)
        try:
            current_gain = camera_settings.gain
            # Mantener la ganancia en un valor moderado (no muy alto para evitar ruido)
            # Valores típicos: 0-100, usar un valor medio-bajo
            camera_settings.gain = min(60, max(30, current_gain))  # Entre 30 y 60
            print(f"✓ Ganancia configurada: {camera_settings.gain}")
        except Exception as e:
            print(f"⚠ No se pudo configurar la ganancia: {e}")
        
        print("✓ Configuración de cámara aplicada correctamente")
        
    except Exception as e:
        print(f"⚠ Error al configurar la cámara: {e}")
        print("  Continuando con configuración por defecto...")

def configurar_voz_tts():
    """
    Configura el motor de texto a voz (TTS) con una voz de mujer en español, lenta y dulce.
    
    Returns:
        pyttsx3.Engine: Motor TTS configurado
    """
    try:
        engine = pyttsx3.init()
        
        # Obtener todas las voces disponibles
        voices = engine.getProperty('voices')
        
        # Buscar una voz de mujer en español
        female_spanish_voice = None
        for voice in voices:
            voice_name = voice.name.lower()
            voice_id = voice.id.lower()
            
            # Buscar voces en español (comúnmente tienen "spanish", "español", "es-es", "es-mx", etc.)
            is_spanish = any(keyword in voice_name or keyword in voice_id 
                           for keyword in ['spanish', 'español', 'es-es', 'es-mx', 'es-ar', 'es-co', 
                                         'sabina', 'helena', 'pablo', 'diego'])
            
            # Buscar voces femeninas en español
            is_female = any(keyword in voice_name or keyword in voice_id 
                          for keyword in ['female', 'woman', 'mujer', 'sabina', 'helena', 'zira'])
            
            if is_spanish and (is_female or 'sabina' in voice_name or 'helena' in voice_name):
                female_spanish_voice = voice
                break
        
        # Si no se encuentra una voz femenina en español, buscar cualquier voz en español
        if female_spanish_voice is None:
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                if any(keyword in voice_name or keyword in voice_id 
                       for keyword in ['spanish', 'español', 'es-es', 'es-mx', 'es-ar', 'es-co', 
                                     'sabina', 'helena', 'pablo', 'diego']):
                    female_spanish_voice = voice
                    break
        
        # Si aún no se encuentra, usar la primera voz disponible
        if female_spanish_voice is None and len(voices) > 0:
            female_spanish_voice = voices[0]
            print("⚠ No se encontró una voz en español, usando la primera disponible")
        
        if female_spanish_voice:
            engine.setProperty('voice', female_spanish_voice.id)
            print(f"✓ Voz configurada: {female_spanish_voice.name}")
        
        # Configurar velocidad (fluida y natural, como una maestra dulce)
        # Rango típico: 50-200, valores más altos = más rápido
        # 140-150 es una velocidad natural y fluida, no muy rápida ni muy lenta
        engine.setProperty('rate', 145)  # Velocidad fluida y natural
        
        # Configurar volumen (dulce y amable) - rango: 0.0 a 1.0
        engine.setProperty('volume', 0.85)  # Volumen suave y dulce, como una maestra amable
        
        return engine
    except Exception as e:
        print(f"⚠ Error al configurar TTS: {e}")
        return None

def reproducir_texto_tts(texto):
    """
    Reproduce un texto usando TTS con la configuración de voz de mujer, lenta y dulce.
    
    Args:
        texto: Texto a reproducir
    """
    try:
        engine = configurar_voz_tts()
        if engine:
            engine.say(texto)
            engine.runAndWait()
    except Exception as e:
        print(f"⚠ Error al reproducir TTS: {e}")

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
    
    # Inicializar pygame si no está inicializado
    global _bocina_muted_global, _background_music_loaded
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
    
    # Cargar y configurar el audio de fondo usando mixer.music (solo si no está cargado)
    if not _background_music_loaded:
        archivo = "relax-meditate-gentle-peaceful-291162.mp3"
        if os.path.exists(archivo):
            try:
                pygame.mixer.music.load(archivo)
                # El módulo music suele responder mejor a volúmenes bajos
                pygame.mixer.music.set_volume(0.05)  # 5% de volumen
                _background_music_loaded = True
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3 (volumen al 5%)")
                # Iniciar el audio automáticamente si no está muteado
                if not _bocina_muted_global:
                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
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
    
    # Cargar imágenes de los niveles (2 escenarios: nivel 1 y nivel 2)
    nivel_images = {}
    nivel_paths = {
        "1": "images/Nivel1Clasificacion.png",
        "2": "images/Nivel2Clasificacion.png"
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
    titulo_texto = "Selecciona el nivel"
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
    
    # Dimensiones de las cards de niveles (mismo tamaño que las cards del menú principal)
    card_width = 320  # Mismo tamaño que las cards del menú principal
    card_height = 400  # Mismo tamaño que las cards del menú principal
    card_spacing = 50  # Mismo espaciado que las cards del menú principal
    
    # Calcular posiciones (centradas, 2 cards en fila)
    total_width = 2 * card_width + card_spacing
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
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar en pantalla (transición suave sin cerrar)
    # Escalar a la resolución del videobeam antes de mostrar
    niveles_screen_scaled = scale_to_videobeam(niveles_screen)
    cv2.imshow(window_name, niveles_screen_scaled)
    
    # Reproducir instrucción con TTS después de mostrar la vista
    reproducir_texto_tts("Elige cuántos escenarios va a ver en el juego.")
    
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
    
    # Configurar parámetros de la cámara antes de iniciar
    configurar_camara_kinect(rgb_stream)
    
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
                            if _background_music_loaded:
                                if bocina_muted:
                                    # Detener el audio cuando está muteada
                                    pygame.mixer.music.stop()
                                    print("Audio de fondo detenido")
                                else:
                                    # Asegurar que el volumen esté al 1% antes de reproducir
                                    pygame.mixer.music.set_volume(0.05)
                                    # Reproducir el audio en bucle cuando está activada
                                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
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
                                
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
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
                                    temp_screen_scaled = scale_to_videobeam(temp_screen)
                                    cv2.imshow(window_name, temp_screen_scaled)
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
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
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
                                    niveles_screen_scaled = scale_to_videobeam(niveles_screen)
                                    cv2.imshow(window_name, niveles_screen_scaled)
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
            niveles_screen_scaled = scale_to_videobeam(niveles_screen)
            cv2.imshow(window_name, niveles_screen_scaled)
            
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
    
    # Inicializar pygame si no está inicializado
    global _bocina_muted_global, _background_music_loaded
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
    
    # Cargar y configurar el audio de fondo usando mixer.music (solo si no está cargado)
    if not _background_music_loaded:
        archivo = "relax-meditate-gentle-peaceful-291162.mp3"
        if os.path.exists(archivo):
            try:
                pygame.mixer.music.load(archivo)
                # El módulo music suele responder mejor a volúmenes bajos
                pygame.mixer.music.set_volume(0.05)  # 5% de volumen
                _background_music_loaded = True
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3 (volumen al 5%)")
                # Iniciar el audio automáticamente si no está muteado
                if not _bocina_muted_global:
                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
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
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mostrar la nueva vista (transición suave sin cerrar)
    # Escalar a la resolución del videobeam antes de mostrar
    cards_screen_scaled = scale_to_videobeam(cards_screen)
    cv2.imshow(window_name, cards_screen_scaled)
    
    # Reproducir instrucción con TTS después de mostrar la vista
    reproducir_texto_tts("Elige los escenarios para jugar.")
    
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
    
    # Configurar parámetros de la cámara antes de iniciar
    configurar_camara_kinect(rgb_stream)
    
    rgb_stream.start()
    depth_stream.start()
    
    # Variables de selección (selected_cards ya inicializada arriba)
    last_touch_time = {}  # Para evitar selecciones múltiples rápidas
    debounce_time = 0.3  # 300ms de debounce
    
    try:
        while True:
            try:
                frame = rgb_stream.read_frame()
                depth_frame = depth_stream.read_frame()
                
                if frame is None or depth_frame is None:
                    # Mostrar pantalla incluso si no hay frame
                    draw_close_card(cards_screen)
                    draw_back_card(cards_screen)
                    draw_bocina_card(cards_screen, muted=bocina_muted)
                    cards_screen_scaled = scale_to_videobeam(cards_screen)
                    cv2.imshow(window_name, cards_screen_scaled)
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
                
                # Crear la máscara de toques
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except Exception as e:
                print(f"Error al procesar frame: {e}")
                import traceback
                traceback.print_exc()
                # Mostrar pantalla incluso si hay error
                draw_close_card(cards_screen)
                draw_back_card(cards_screen)
                draw_bocina_card(cards_screen, muted=bocina_muted)
                cards_screen_scaled = scale_to_videobeam(cards_screen)
                cv2.imshow(window_name, cards_screen_scaled)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
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
                            if _background_music_loaded:
                                if bocina_muted:
                                    # Detener el audio cuando está muteada
                                    pygame.mixer.music.stop()
                                    print("Audio de fondo detenido")
                                else:
                                    # Asegurar que el volumen esté al 1% antes de reproducir
                                    pygame.mixer.music.set_volume(0.05)
                                    # Reproducir el audio en bucle cuando está activada
                                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
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
                            cards_screen_scaled = scale_to_videobeam(cards_screen)
                            cv2.imshow(window_name, cards_screen_scaled)
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
                                
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
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
                                
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
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
                                            cards_screen_scaled = scale_to_videobeam(cards_screen)
                                            cv2.imshow(window_name, cards_screen_scaled)
                                            
                                            # Esperar un momento para mostrar las cards seleccionadas con borde verde
                                            cv2.waitKey(800)  # 800ms de pausa antes de cambiar de vista
                                            
                                            # Mostrar vista con rectángulos para los escenarios seleccionados
                                            resultado = mostrar_vista_rectangulos_escenarios(
                                                device, coordenadas, dmax_map, dmin_map, draw_logo_func, 
                                                selected_cards, existing_window_name=window_name
                                            )
                                            # Si se presionó el botón "Salir", retornar None para volver al menú principal
                                            if resultado == "MAIN_MENU":
                                                return None  # Retornar None para que mostrar_seleccion_niveles_clasificacion también retorne None
                                            # Si se presionó el botón "Volver a jugar", volver a la vista de niveles
                                            if resultado == "RESTART":
                                                # Desmarcar todos los escenarios seleccionados y volver a la vista de niveles
                                                selected_cards = []
                                                nivel_seleccionado_flag = False
                                                print("Volviendo a la vista de niveles")
                                                # Continuar el bucle para mostrar la vista de niveles nuevamente
                                                continue
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
                                                cards_screen_scaled = scale_to_videobeam(cards_screen)
                                                cv2.imshow(window_name, cards_screen_scaled)
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
            cards_screen_scaled = scale_to_videobeam(cards_screen)
            cv2.imshow(window_name, cards_screen_scaled)
            
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
    
    # Mapeo de escenarios a imágenes
    escenario_images = {
        "Escenario 1": "images/EscenarioGranja.png",
        "Escenario 2": "images/EscenarioCalle.png",
        "Escenario 3": "images/Escenariorestaurante.png",
        "Escenario 4": "images/EscenarioRopa.png",
        "Escenario 5": "images/EscenarioColegio.png"
    }
    
    # Inicializar detector de objetos YOLO
    detector = ObjectDetector()
    
    # Cargar traducciones desde JSON
    translations = {}
    translation_file = os.path.join(project_root, "src", "config", "label_translations.json")
    if os.path.exists(translation_file):
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                print(f"✓ Traducciones cargadas desde {translation_file}")
        except Exception as e:
            print(f"⚠ Error al cargar traducciones: {e}")
            translations = {"labels": {}, "escenarios": {}}
    else:
        print(f"⚠ No se encontró el archivo de traducciones: {translation_file}")
        translations = {"labels": {}, "escenarios": {}}
    
    # Diccionario de artículos para cada pieza (basado en el label en inglés)
    articulos_piezas = {
        # Animales - Escenario 1
        "Cat": "el",      # el gato
        "Dog": "el",      # el perro
        "Pig": "el",      # el cerdo
        "bird": "el",     # el pájaro
        "cow": "la",      # la vaca
        "duck": "el",     # el pato
        "hen": "la",      # la gallina
        "horse": "el",    # el caballo
        "sheep": "la",    # la oveja
        # Vehículos - Escenario 2
        "Skateboard": "la",    # la patineta
        "Train": "el",         # el tren
        "Van": "la",           # la camioneta
        "bike": "la",          # la bicicleta
        "bus": "el",           # el autobús
        "car": "el",           # el carro
        "motorbike": "la",     # la motocicleta
        # Comida - Escenario 3
        "apple": "la",         # la manzana
        "banana": "el",        # el plátano
        "cake": "el",          # el pastel
        "hamburger": "la",     # la hamburguesa
        "hot dog": "el",       # el perro caliente
        "milkshakes": "el",    # el batido
        "orange": "la",        # la naranja
        "pizza": "la",         # la pizza
        "potatoes": "las",     # las papas (plural)
        "soda": "el",          # el refresco
        "spaghetti": "el",     # el espagueti
        "strawberry": "la",    # la fresa
        # Ropa - Escenario 4
        "cap": "la",           # la gorra
        "glasses": "los",      # los lentes (plural)
        "pants": "los",        # los pantalones (plural)
        "sandals": "las",      # las sandalias (plural)
        "shirt": "la",         # la camisa
        "shoes": "los",        # los zapatos (plural)
        "socks": "los",        # los calcetines (plural)
        "sweater": "el",       # el suéter
        # Material escolar - Escenario 5
        "backpack": "la",      # la mochila
        "book": "el",          # el libro
        "colors": "los",       # los colores (plural)
        "eraser": "el",        # el borrador
        "notebook": "el",      # el cuaderno
        "pencil": "el",        # el lápiz
        "ruler": "la"          # la regla
    }
    
    # Diccionario de artículos para cada escenario (basado en el nombre del escenario en español)
    articulos_escenarios = {
        "granja": "la",        # la granja
        "calle": "la",         # la calle
        "restaurante": "el",   # el restaurante
        "ropa": "la",          # la ropa
        "colegio": "el"        # el colegio
    }
    
    # Función para obtener el artículo correcto para una pieza
    def obtener_articulo(label_ingles):
        """Devuelve el artículo correcto (el/la/los/las) para un label en inglés"""
        return articulos_piezas.get(label_ingles, "el")  # Por defecto "el" si no se encuentra
    
    # Función para obtener el artículo correcto para un escenario
    def obtener_articulo_escenario(escenario_es):
        """Devuelve el artículo correcto (el/la) para un escenario en español"""
        return articulos_escenarios.get(escenario_es, "el")  # Por defecto "el" si no se encuentra
    
    # Inicializar pygame si no está inicializado
    try:
        pygame.mixer.get_init()
    except:
        pygame.mixer.init()
    
    # Cargar sonido de incorrecto
    incorrecto_sound = None
    if os.path.exists("sounds/incorrecto.mp3"):
        try:
            incorrecto_sound = pygame.mixer.Sound("sounds/incorrecto.mp3")
            # Ajustar el volumen al 100% (volumen completo) para que se escuche bien
            incorrecto_sound.set_volume(1.0)
            print("✓ Sonido de incorrecto cargado: sounds/incorrecto.mp3 (volumen al 100%)")
        except Exception as e:
            print(f"⚠ No se pudo cargar el sonido de incorrecto: {e}")
    else:
        print("⚠ No se encontró el archivo de audio: sounds/incorrecto.mp3")
    
    # Cargar imagen de éxito
    imagen_exito = None
    imagen_exito_path = "images/LingoBien.png"
    if os.path.exists(imagen_exito_path):
        try:
            imagen_exito = cv2.imread(imagen_exito_path, cv2.IMREAD_UNCHANGED)
            if imagen_exito is not None:
                print(f"✓ Imagen de éxito cargada: {imagen_exito_path}")
            else:
                print(f"⚠ No se pudo cargar la imagen de éxito desde {imagen_exito_path}")
        except Exception as e:
            print(f"⚠ Error al cargar la imagen de éxito: {e}")
    else:
        print(f"⚠ No se encontró el archivo de imagen: {imagen_exito_path}")
    
    # Cargar sonido de correcto
    correcto_sound = None
    if os.path.exists("sounds/correcto.wav"):
        try:
            correcto_sound = pygame.mixer.Sound("sounds/correcto.wav")
            correcto_sound.set_volume(1.0)
            print("✓ Sonido de correcto cargado: sounds/correcto.wav (volumen al 100%)")
        except Exception as e:
            print(f"⚠ No se pudo cargar el sonido de correcto: {e}")
    else:
        print("⚠ No se encontró el archivo de audio: sounds/correcto.wav")
    
    # Variable para rastrear si el juego está completo y si ya se reprodujo el sonido
    juego_completo = False
    sonido_reproducido = False
    
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
            
            # Dibujar rectángulo principal (fondo gris claro)
            rect_color = (200, 200, 200)  # Gris claro (menos claro que antes)
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
                
                # Aplicar transparencia del 40% a todas las imágenes
                opacity = 0.4  # 40% de opacidad
                
                if resized_img.shape[2] == 4:
                    # Si la imagen tiene canal alfa, combinar con la transparencia
                    b, g, r, a_original = cv2.split(resized_img)
                    # Normalizar el canal alfa original
                    a_original = a_original.astype(np.float32) / 255.0
                    # Combinar la transparencia original con la opacidad configurada
                    a_combined = a_original * opacity
                    
                    # Aplicar la imagen con transparencia combinada
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - a_combined) +
                            resized_img[:, :, c] * a_combined
                        ).astype(np.uint8)
                else:
                    # Si no tiene canal alfa, aplicar transparencia directamente
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - opacity) +
                            resized_img[:, :, c] * opacity
                        ).astype(np.uint8)
                
                # Filtro blanco removido - las imágenes se muestran sin opacidad blanca
            
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
        # Crear ventana en modo normal primero
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Mover a la segunda pantalla (1920, 0)
        cv2.moveWindow(window_name, 1920, 0)
        # Esperar un momento para que la ventana se mueva
        cv2.waitKey(50)
        # Establecer pantalla completa
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Esperar un momento más para que se aplique
        cv2.waitKey(50)
    else:
        # Si la ventana ya existe, moverla y asegurar pantalla completa
        try:
            cv2.moveWindow(window_name, 1920, 0)
            cv2.waitKey(10)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except:
            pass
    
    # Mostrar en pantalla (transición suave sin cerrar)
    # Escalar a la resolución del videobeam antes de mostrar
    rectangulos_screen_scaled = scale_to_videobeam(rectangulos_screen)
    cv2.imshow(window_name, rectangulos_screen_scaled)
    
    # Reproducir instrucción con TTS después de mostrar la vista
    reproducir_texto_tts("Coloca cada figura en el escenario que le corresponde.")
    
    # Forzar pantalla completa después de mostrar (asegurar que se aplique)
    cv2.waitKey(50)
    try:
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Forzar también el resize por si acaso
        cv2.resizeWindow(window_name, view_width, view_height)
    except:
        pass
    
    # Iniciar streams de cámara (aunque no se usen para interacción, mantener consistencia)
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    
    # Configurar parámetros de la cámara antes de iniciar
    configurar_camara_kinect(rgb_stream)
    
    rgb_stream.start()
    depth_stream.start()
    
    # Inicializar detecciones vacías
    detections = []
    
    # Sistema de estabilización de nombres: historial por posición del objeto
    # Usa una tupla (cx_proj, cy_proj) como clave para identificar objetos
    label_history = {}  # {(cx, cy): [lista de nombres recientes]}
    max_history_size = 5  # Número de detecciones a considerar para estabilización (reducido para respuesta más rápida)
    confidence_threshold = 0.6  # Umbral mínimo de confianza para mostrar el nombre
    
    # Sistema de seguimiento de objetos para reproducir sonido de incorrecto
    # Rastrea qué objetos estaban presentes en el frame anterior
    # Clave: label - Valor: (cx_proj, cy_proj, escenario_anterior, presente)
    objeto_posicion_anterior = {}  # {label: (cx_proj, cy_proj, escenario_anterior, presente)}
    
    # Sistema de rastreo de piezas correctas por escenario
    # Rastrea qué objetos están correctamente colocados en cada escenario
    # Clave: escenario - Valor: set de labels de objetos correctos
    piezas_correctas_por_escenario = {escenario: set() for escenario in escenarios_seleccionados}
    
    # Sistema de seguimiento de tiempo para sonido incorrecto (3 segundos de umbral)
    # Clave: label - Valor: timestamp cuando el objeto entró en posición incorrecta
    objeto_incorrecto_tiempo = {}  # {label: timestamp}
    
    # Sistema de seguimiento de objetos que ya han recibido anuncio TTS de correcto
    # Clave: label - Valor: True si ya se anunció
    objetos_anunciados = set()  # Set de labels que ya recibieron anuncio TTS
    
    # Sistema de seguimiento de escenarios que ya fueron anunciados cuando completaron las 5 piezas
    escenarios_anunciados_completos = set()  # Set de escenarios que ya recibieron anuncio cuando completaron 5 piezas
    
    # Sistema de seguimiento de objetos incorrectos que ya fueron anunciados
    # Clave: (label, escenario) - Valor: True si ya se anunció que este objeto no pertenece a este escenario
    objetos_incorrectos_anunciados = set()  # Set de tuplas (label, escenario) que ya recibieron anuncio TTS de incorrecto
    
    # Sistema de seguimiento de objetos temporalmente perdidos (ocultos por mano u otro objeto)
    # Clave: label - Valor: (timestamp_desaparicion, escenario_anterior, cx_prev, cy_prev)
    # Este sistema permite tolerar pérdidas temporales de detección (ej: cuando una mano pasa frente a la figura)
    objetos_temporalmente_perdidos = {}  # {label: (timestamp, escenario, cx, cy)}
    umbral_perdida_temporal = 3.0  # Segundos que un objeto puede estar perdido antes de considerarlo realmente removido
    
    def get_stable_label(cx_proj, cy_proj, current_label, label_history, max_history_size):
        """
        Obtiene el nombre estabilizado basado en el historial de detecciones.
        Usa una ventana deslizante de posiciones cercanas.
        """
        # Buscar en el historial objetos cercanos (dentro de 50 píxeles)
        tolerance = 50
        best_match = None
        best_distance = float('inf')
        
        for (hist_cx, hist_cy), hist_labels in label_history.items():
            distance = ((cx_proj - hist_cx) ** 2 + (cy_proj - hist_cy) ** 2) ** 0.5
            if distance < tolerance and distance < best_distance:
                best_match = (hist_cx, hist_cy)
                best_distance = distance
        
        if best_match is not None:
            # Usar el historial existente
            hist_labels = label_history[best_match]
            hist_labels.append(current_label)
            if len(hist_labels) > max_history_size:
                hist_labels.pop(0)
            # Retornar el nombre más frecuente en el historial
            from collections import Counter
            most_common = Counter(hist_labels).most_common(1)
            if most_common:
                # Actualizar la posición en el historial
                del label_history[best_match]
                label_history[(int(cx_proj), int(cy_proj))] = hist_labels
                return most_common[0][0]
        else:
            # Nuevo objeto, crear entrada en el historial
            label_history[(int(cx_proj), int(cy_proj))] = [current_label]
        
        return current_label
    
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
            
            # Si el juego está completo, solo procesar toques para los botones
            if juego_completo:
                # Solo procesar toques, no detecciones de objetos
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                depth_data = cv2.flip(depth_data, 1)
                depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
                
                # Crear la máscara de toques
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Procesar toques para los botones
                if contours:
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
                            
                            # Verificar toques en los botones
                            if imagen_exito is not None:
                                # Usar las mismas dimensiones escaladas que en el dibujado
                                scale_factor = 1.8
                                img_h_original, img_w_original = imagen_exito.shape[:2]
                                img_w = int(img_w_original * scale_factor)
                                img_h = int(img_h_original * scale_factor)
                                center_x = view_width // 2
                                center_y = view_height // 2
                                y1_img = center_y - img_h // 2
                                y2_img = y1_img + img_h
                                
                                button_height = 80
                                button_width = 200
                                button_spacing = 50
                                button_y = y1_img - button_height - 40  # 40 píxeles arriba de la imagen
                                
                                button_volver_x1 = center_x - button_width - button_spacing // 2
                                button_volver_y1 = button_y
                                button_volver_x2 = button_volver_x1 + button_width
                                button_volver_y2 = button_volver_y1 + button_height
                                
                                button_salir_x1 = center_x + button_spacing // 2
                                button_salir_y1 = button_y
                                button_salir_x2 = button_salir_x1 + button_width
                                button_salir_y2 = button_salir_y1 + button_height
                                
                                # Verificar si se tocó el botón "Volver a jugar"
                                if (button_volver_x1 <= x_touch <= button_volver_x2 and 
                                    button_volver_y1 <= y_touch <= button_volver_y2):
                                    print("Botón 'Volver a jugar' presionado - Volviendo a vista de niveles")
                                    # Retornar "RESTART" para volver a la vista de niveles
                                    return "RESTART"
                                
                                # Verificar si se tocó el botón "Salir"
                                elif (button_salir_x1 <= x_touch <= button_salir_x2 and 
                                      button_salir_y1 <= y_touch <= button_salir_y2):
                                    print("Botón 'Salir' presionado - Volviendo al menú principal")
                                    # Retornar "MAIN_MENU" para volver al menú principal
                                    return "MAIN_MENU"
                
                # Continuar con el bucle para mostrar la pantalla de éxito
                # (el código de dibujado se ejecutará más abajo)
            else:
                # Realizar detección de objetos solo si el juego no está completo
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
                                    
                                    temp_screen_scaled = scale_to_videobeam(temp_screen)
                                    cv2.imshow(window_name, temp_screen_scaled)
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
                                
                                temp_screen_scaled = scale_to_videobeam(temp_screen)
                                cv2.imshow(window_name, temp_screen_scaled)
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
            
            # Función para calcular IoU (Intersection over Union) entre dos bounding boxes
            def calculate_iou(bbox1, bbox2):
                """Calcula el IoU entre dos bounding boxes [x1, y1, x2, y2]"""
                x1_1, y1_1, x2_1, y2_1 = bbox1
                x1_2, y1_2, x2_2, y2_2 = bbox2
                
                # Calcular intersección
                x1_inter = max(x1_1, x1_2)
                y1_inter = max(y1_1, y1_2)
                x2_inter = min(x2_1, x2_2)
                y2_inter = min(y2_1, y2_2)
                
                if x2_inter <= x1_inter or y2_inter <= y1_inter:
                    return 0.0
                
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                
                # Calcular áreas
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union_area = area1 + area2 - inter_area
                
                if union_area == 0:
                    return 0.0
                
                return inter_area / union_area
            
            # Función para combinar detecciones solapadas de la misma clase
            def merge_overlapping_detections(detections, iou_threshold=0.3):
                """
                Agrupa y combina detecciones solapadas de la misma clase.
                Retorna una lista de detecciones combinadas.
                """
                if not detections:
                    return []
                
                # Agrupar por clase (label)
                detections_by_label = {}
                for det in detections:
                    label = det.get("label", "")
                    if label not in detections_by_label:
                        detections_by_label[label] = []
                    detections_by_label[label].append(det)
                
                merged_detections = []
                
                # Procesar cada grupo de detecciones de la misma clase
                for label, label_detections in detections_by_label.items():
                    if len(label_detections) == 1:
                        # Solo una detección, agregarla directamente
                        merged_detections.append(label_detections[0])
                        continue
                    
                    # Múltiples detecciones de la misma clase
                    # Agrupar las que están solapadas
                    groups = []
                    used = [False] * len(label_detections)
                    
                    for i, det1 in enumerate(label_detections):
                        if used[i]:
                            continue
                        
                        # Crear un nuevo grupo con esta detección
                        group = [det1]
                        used[i] = True
                        
                        # Buscar otras detecciones solapadas
                        for j, det2 in enumerate(label_detections):
                            if used[j] or i == j:
                                continue
                            
                            # Calcular IoU
                            iou = calculate_iou(det1["bbox"], det2["bbox"])
                            
                            if iou > iou_threshold:
                                group.append(det2)
                                used[j] = True
                        
                        groups.append(group)
                    
                    # Combinar cada grupo en una sola detección
                    for group in groups:
                        if len(group) == 1:
                            merged_detections.append(group[0])
                        else:
                            # Combinar múltiples detecciones en una
                            # Tomar el bounding box que engloba todas las detecciones
                            x1_min = min(d["bbox"][0] for d in group)
                            y1_min = min(d["bbox"][1] for d in group)
                            x2_max = max(d["bbox"][2] for d in group)
                            y2_max = max(d["bbox"][3] for d in group)
                            
                            # Usar la confianza promedio del grupo
                            avg_confidence = sum(d.get("confidence", 0.0) for d in group) / len(group)
                            
                            # Crear detección combinada
                            merged_det = {
                                "label": label,
                                "confidence": avg_confidence,
                                "bbox": [x1_min, y1_min, x2_max, y2_max],
                                "category": group[0].get("category")  # Usar la categoría de la primera
                            }
                            merged_detections.append(merged_det)
                
                return merged_detections
            
            # Solo procesar detecciones si el juego no está completo
            if not juego_completo:
                # Agrupar detecciones solapadas antes de dibujarlas
                merged_detections = merge_overlapping_detections(detections, iou_threshold=0.3)
                
                # Conjunto para rastrear qué objetos están presentes en este frame
                labels_presentes = set()
            else:
                # Si el juego está completo, no hay detecciones
                merged_detections = []
                labels_presentes = set()
            
            # Dibujar efectos de brillo para objetos detectados
            # Calcular relación píxeles/cm (basado en view_width=1280 y ancho físico=192cm)
            ancho_fisico_cm = 192
            pixels_per_cm = view_width / ancho_fisico_cm  # ≈ 6.67 píxeles/cm
            aumento_cm = 2  # Agregar 2 cm más al tamaño del cuadro
            aumento_px = int(aumento_cm * pixels_per_cm)  # ≈ 13 píxeles
            green_color = (0, 255, 0)  # Verde en BGR para escenario correcto
            red_color = (0, 0, 255)  # Rojo en BGR para escenario incorrecto
            
            # Solo procesar detecciones si el juego no está completo
            if not juego_completo:
                for det in merged_detections:
                    categoria_objetivo = det.get("category")
                    confidence = det.get("confidence", 0.0)
                    
                    # Solo procesar si la confianza es suficiente
                    if confidence < confidence_threshold:
                        continue
                    
                    # Obtener el bounding box del objeto detectado
                    x1, y1, x2, y2 = det["bbox"]
                    
                    # Mapear las esquinas del bounding box a coordenadas de proyección
                    x1_proj, y1_proj = detector.map_coordinates(int(x1), int(y1), coordenadas)
                    x2_proj, y2_proj = detector.map_coordinates(int(x2), int(y2), coordenadas)
                    
                    # Calcular el centro directamente de las esquinas mapeadas (más preciso)
                    cx_proj = (x1_proj + x2_proj) / 2
                    cy_proj = (y1_proj + y2_proj) / 2
                    
                    # Calcular el tamaño del bounding box en proyección
                    bbox_width_proj = abs(x2_proj - x1_proj)
                    bbox_height_proj = abs(y2_proj - y1_proj)
                    
                    # Usar el tamaño real del bounding box y agregar 2 cm más
                    # Asegurar un tamaño mínimo razonable
                    min_size_px = int(5 * pixels_per_cm)  # Mínimo 5 cm
                    box_width = max(bbox_width_proj + aumento_px, min_size_px)
                    box_height = max(bbox_height_proj + aumento_px, min_size_px)
                    
                    # Determinar en qué escenario está el objeto (si está en alguno)
                    escenario_actual = None
                    for escenario, pos in rectangulos_positions.items():
                        rx, ry, rw, rh = pos['x'], pos['y'], pos['width'], pos['height']
                        if rx <= cx_proj <= rx + rw and ry <= cy_proj <= ry + rh:
                            escenario_actual = escenario
                            break
                    
                    # Si no está en ningún escenario, no dibujar cuadro
                    if escenario_actual is None:
                        continue
                    
                    # Obtener el nombre estabilizado (necesario para el sonido y el texto)
                    current_label = det["label"]
                    stable_label = get_stable_label(cx_proj, cy_proj, current_label, label_history, max_history_size)
                    
                    # Agregar el label al conjunto de objetos presentes
                    labels_presentes.add(stable_label)
                    
                    # Verificar si este objeto estaba temporalmente perdido y restaurar su estado
                    objeto_restaurado_silenciosamente = False
                    if stable_label in objetos_temporalmente_perdidos:
                        # El objeto reapareció después de estar temporalmente perdido
                        timestamp_perdida, escenario_perdido, cx_perdido, cy_perdido = objetos_temporalmente_perdidos[stable_label]
                        tiempo_perdido = time.time() - timestamp_perdida
                        
                        # Verificar si está en el mismo escenario o cerca de donde estaba
                        distancia = ((cx_proj - cx_perdido) ** 2 + (cy_proj - cy_perdido) ** 2) ** 0.5
                        
                        # Si reapareció en el mismo escenario donde estaba correctamente colocado, restaurar silenciosamente
                        if escenario_actual == escenario_perdido and distancia < 150:  # Dentro de 150 píxeles
                            # Verificar si estaba correctamente colocado antes de perderse
                            if escenario_perdido in piezas_correctas_por_escenario:
                                if stable_label in piezas_correctas_por_escenario[escenario_perdido]:
                                    # El objeto estaba correctamente colocado antes, restaurar su estado silenciosamente
                                    # No anunciar de nuevo, solo restaurar
                                    objeto_posicion_anterior[stable_label] = (cx_proj, cy_proj, escenario_perdido, True)
                                    # Asegurar que sigue en las piezas correctas (por si acaso se removió)
                                    piezas_correctas_por_escenario[escenario_perdido].add(stable_label)
                                    # Marcar como anunciado para evitar re-anunciar
                                    objetos_anunciados.add(stable_label)
                                    objeto_restaurado_silenciosamente = True
                                    print(f"Objeto {stable_label} restaurado silenciosamente en {escenario_perdido} después de pérdida temporal ({tiempo_perdido:.2f}s)")
                        
                        # Remover de objetos temporalmente perdidos ya que reapareció
                        del objetos_temporalmente_perdidos[stable_label]
                    
                    # Determinar el color del cuadro
                    box_color = green_color  # Por defecto verde
                    
                    # Verificar si el objeto está en el escenario correcto
                    es_correcto = False
                    if categoria_objetivo is not None and categoria_objetivo == escenario_actual:
                        # Está en el escenario correcto: verde
                        box_color = green_color
                        es_correcto = True
                    else:
                        # Está en un escenario incorrecto (o no tiene categoría válida): rojo
                        box_color = red_color
                        es_correcto = False
                    
                    # Reproducir sonido de incorrecto si el objeto está en escenario incorrecto
                    if not es_correcto and incorrecto_sound is not None:
                        
                        # Verificar si este objeto estaba presente en el frame anterior
                        objeto_anterior = objeto_posicion_anterior.get(stable_label)
                        current_time = time.time()
                        
                        # Verificar si el objeto cambió de posición o es nuevo
                        if objeto_anterior is None:
                            # El objeto no estaba presente en el frame anterior (nuevo objeto o recién colocado)
                            # Iniciar el temporizador de 3 segundos
                            objeto_incorrecto_tiempo[stable_label] = current_time
                        else:
                            _, _, escenario_anterior, estaba_presente = objeto_anterior
                            
                            # Si el objeto cambió de escenario o no estaba presente, reiniciar el temporizador
                            if not estaba_presente or escenario_anterior != escenario_actual:
                                objeto_incorrecto_tiempo[stable_label] = current_time
                            # Si el objeto está en el mismo escenario incorrecto, verificar si han pasado 1 segundo
                            elif escenario_anterior == escenario_actual and stable_label in objeto_incorrecto_tiempo:
                                tiempo_incorrecto = current_time - objeto_incorrecto_tiempo[stable_label]
                                if tiempo_incorrecto >= 1.0:  # Reducido de 3.0 a 1.0 segundos para respuesta más rápida
                                    # Han pasado 3 segundos, reproducir el sonido y anunciar con TTS
                                    try:
                                        # Asegurar que el volumen esté al 100% antes de reproducir
                                        incorrecto_sound.set_volume(1.0)
                                        incorrecto_sound.play()
                                        print(f"Sonido de incorrecto reproducido para {stable_label} en {escenario_actual} (después de 3 segundos)")
                                    except Exception as e:
                                        print(f"Error al reproducir sonido de incorrecto: {e}")
                                    
                                    # Anunciar con TTS si no se ha anunciado antes para este objeto en este escenario
                                    objeto_escenario_key = (stable_label, escenario_actual)
                                    if objeto_escenario_key not in objetos_incorrectos_anunciados:
                                        # Obtener traducciones
                                        label_es = translations.get("labels", {}).get(stable_label, stable_label)
                                        escenario_es = translations.get("escenarios", {}).get(escenario_actual, escenario_actual)
                                        
                                        # Obtener los artículos correctos
                                        articulo = obtener_articulo(stable_label)
                                        articulo_escenario = obtener_articulo_escenario(escenario_es)
                                        
                                        # Construir mensaje TTS: "[artículo] [nombre_figura] no pertenece a [artículo] [escenario]"
                                        mensaje = f"{articulo} {label_es} no pertenece a {articulo_escenario} {escenario_es}"
                                        reproducir_texto_tts(mensaje)
                                        objetos_incorrectos_anunciados.add(objeto_escenario_key)
                                        print(f"TTS anunciado (incorrecto): {mensaje}")
                                    
                                    # Reiniciar el temporizador para evitar repetir el sonido continuamente
                                    objeto_incorrecto_tiempo[stable_label] = current_time
                        
                        # Actualizar la posición, escenario y estado de presencia del objeto
                        objeto_posicion_anterior[stable_label] = (cx_proj, cy_proj, escenario_actual, True)
                        
                        # Remover de las piezas correctas si estaba en otro escenario
                        for escenario in piezas_correctas_por_escenario:
                            if stable_label in piezas_correctas_por_escenario[escenario]:
                                piezas_correctas_por_escenario[escenario].discard(stable_label)
                                # Solo remover de escenarios_anunciados_completos si el escenario realmente tiene menos de 5 piezas
                                # Esto evita que se repita el anuncio cuando una pieza se mueve temporalmente
                                if len(piezas_correctas_por_escenario[escenario]) < 5:
                                    escenarios_anunciados_completos.discard(escenario)
                        
                        # Remover del conjunto de objetos anunciados si estaba en posición correcta antes
                        if stable_label in objetos_anunciados:
                            objetos_anunciados.discard(stable_label)
                        
                        # Limpiar anuncios de incorrecto para este objeto en otros escenarios (para que pueda anunciarse de nuevo si cambia)
                        objetos_incorrectos_anunciados = {(label, esc) for (label, esc) in objetos_incorrectos_anunciados 
                                                          if not (label == stable_label and esc != escenario_actual)}
                    elif es_correcto:
                        # Si el objeto fue restaurado silenciosamente, no procesar anuncios
                        if objeto_restaurado_silenciosamente:
                            # Solo actualizar posición y dibujar, sin anunciar
                            objeto_posicion_anterior[stable_label] = (cx_proj, cy_proj, escenario_actual, True)
                        # Si el escenario ya está completo (tiene 5 piezas y fue anunciado), 
                        # mantener el conteo pero no volver a anunciar
                        elif escenario_actual in escenarios_anunciados_completos:
                            # El escenario ya está completo, actualizar posición y mantener el conteo
                            objeto_posicion_anterior[stable_label] = (cx_proj, cy_proj, escenario_actual, True)
                            # Agregar la pieza de vuelta al conteo si no estaba (para mantener el conteo correcto)
                            if escenario_actual in piezas_correctas_por_escenario:
                                if stable_label not in piezas_correctas_por_escenario[escenario_actual]:
                                    piezas_correctas_por_escenario[escenario_actual].add(stable_label)
                        else:
                            # Si está en el escenario correcto, marcar como presente pero no reproducir sonido
                            objeto_anterior = objeto_posicion_anterior.get(stable_label)
                            objeto_posicion_anterior[stable_label] = (cx_proj, cy_proj, escenario_actual, True)
                            
                            # Remover de las piezas correctas de otros escenarios primero
                            for escenario in piezas_correctas_por_escenario:
                                if escenario != escenario_actual and stable_label in piezas_correctas_por_escenario[escenario]:
                                    piezas_correctas_por_escenario[escenario].discard(stable_label)
                                    # Solo remover de escenarios_anunciados_completos si el escenario realmente tiene menos de 5 piezas
                                    # Esto evita que se repita el anuncio cuando una pieza se mueve temporalmente
                                    if len(piezas_correctas_por_escenario[escenario]) < 5:
                                        escenarios_anunciados_completos.discard(escenario)
                            
                            # Verificar si es la primera vez que este objeto está en la posición correcta
                            # Solo anunciar si no estaba en posición correcta antes o es un objeto nuevo
                            debe_anunciar = False
                            if objeto_anterior is None:
                                # Objeto nuevo, anunciar si no se ha anunciado antes
                                if stable_label not in objetos_anunciados:
                                    debe_anunciar = True
                            else:
                                _, _, escenario_anterior, estaba_presente = objeto_anterior
                                # Anunciar si:
                                # 1. No estaba presente antes (objeto nuevo)
                                # 2. Estaba en un escenario diferente (cambió de posición)
                                # 3. No se ha anunciado antes para este escenario
                                if not estaba_presente or escenario_anterior != escenario_actual:
                                    if stable_label not in objetos_anunciados:
                                        debe_anunciar = True
                            
                            # Agregar a las piezas correctas del escenario actual
                            if escenario_actual in piezas_correctas_por_escenario:
                                # Solo agregar si no estaba ya en este escenario
                                if stable_label not in piezas_correctas_por_escenario[escenario_actual]:
                                    piezas_correctas_por_escenario[escenario_actual].add(stable_label)
                                    
                                    # Verificar si el escenario alcanzó exactamente 5 piezas correctas
                                    piezas_ahora = len(piezas_correctas_por_escenario[escenario_actual])
                                    
                                    # Si alcanzó 5 piezas y no se ha anunciado antes para este escenario
                                    if piezas_ahora == 5 and escenario_actual not in escenarios_anunciados_completos:
                                        # Obtener todas las piezas del escenario
                                        piezas_escenario = piezas_correctas_por_escenario[escenario_actual]
                                        
                                        # Obtener traducciones
                                        escenario_es = translations.get("escenarios", {}).get(escenario_actual, escenario_actual)
                                        
                                        # Obtener el artículo del escenario
                                        articulo_escenario = obtener_articulo_escenario(escenario_es)
                                        
                                        # Construir lista de nombres de piezas en español con artículos
                                        nombres_piezas = []
                                        for pieza_label in piezas_escenario:
                                            pieza_es = translations.get("labels", {}).get(pieza_label, pieza_label)
                                            articulo = obtener_articulo(pieza_label)
                                            # Agregar artículo + nombre: "el gato", "la manzana", etc.
                                            nombres_piezas.append(f"{articulo} {pieza_es}")
                                        
                                        # Construir mensaje TTS: "Está correcto. [artículo pieza1], [artículo pieza2], [artículo pieza3], [artículo pieza4] y [artículo pieza5] pertenecen a [artículo] [escenario]"
                                        mensaje = "Está correcto. "
                                        if len(nombres_piezas) > 0:
                                            # Unir todas las piezas con comas, excepto la última que lleva "y"
                                            if len(nombres_piezas) == 1:
                                                mensaje += nombres_piezas[0]
                                            elif len(nombres_piezas) == 2:
                                                mensaje += f"{nombres_piezas[0]} y {nombres_piezas[1]}"
                                            else:
                                                # Para 3 o más: "el gato, la vaca, el perro, la gallina y el caballo"
                                                mensaje += ", ".join(nombres_piezas[:-1])
                                                mensaje += f" y {nombres_piezas[-1]}"
                                        
                                        mensaje += f" pertenecen a {articulo_escenario} {escenario_es}"
                                        
                                        reproducir_texto_tts(mensaje)
                                        escenarios_anunciados_completos.add(escenario_actual)
                                        print(f"TTS anunciado (escenario completo): {mensaje}")
                            
                            # Marcar el objeto como anunciado (aunque no anunciemos individualmente)
                            if debe_anunciar:
                                objetos_anunciados.add(stable_label)
                        
                        # Limpiar el temporizador de incorrecto si estaba en posición incorrecta antes
                        if stable_label in objeto_incorrecto_tiempo:
                            del objeto_incorrecto_tiempo[stable_label]
                        
                        # Limpiar anuncios de incorrecto para este objeto (ya que ahora está correcto)
                        objetos_incorrectos_anunciados = {(label, esc) for (label, esc) in objetos_incorrectos_anunciados 
                                                          if label != stable_label}
                    
                    # Dibujar efecto de brillo centrado en el centro calculado de las esquinas mapeadas
                    box_x = int(cx_proj - box_width / 2)
                    box_y = int(cy_proj - box_height / 2)
                    box_w = int(box_width)
                    box_h = int(box_height)
                    draw_shine_effect(rectangulos_screen, box_x, box_y, box_w, box_h, color=box_color)
                    
                    # Mostrar el nombre del objeto detectado (solo si la confianza es alta)
                    if confidence >= confidence_threshold:
                        text_x = int(cx_proj - box_width / 4)
                        text_y = int(cy_proj - box_height / 2 - 10)
                        # Usar color negro para el texto (mejor visibilidad)
                        cv2.putText(rectangulos_screen, stable_label, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Marcar como no presentes los objetos que estaban en el frame anterior pero no en el actual
            current_time = time.time()
            for label in list(objeto_posicion_anterior.keys()):
                if label not in labels_presentes:
                    # El objeto ya no está presente
                    cx_prev, cy_prev, escenario_prev, estaba_presente = objeto_posicion_anterior[label]
                    
                    # Si el objeto estaba presente antes, marcarlo como temporalmente perdido
                    if estaba_presente:
                        # Verificar si ya está en la lista de objetos temporalmente perdidos
                        if label not in objetos_temporalmente_perdidos:
                            # Primera vez que se pierde, registrar timestamp
                            objetos_temporalmente_perdidos[label] = (current_time, escenario_prev, cx_prev, cy_prev)
                            objeto_posicion_anterior[label] = (cx_prev, cy_prev, escenario_prev, False)
                            print(f"Objeto {label} temporalmente perdido en {escenario_prev}")
                        else:
                            # Ya estaba perdido, verificar si ha pasado el umbral
                            timestamp_perdida, _, _, _ = objetos_temporalmente_perdidos[label]
                            tiempo_perdido = current_time - timestamp_perdida
                            
                            if tiempo_perdido >= umbral_perdida_temporal:
                                # Ha pasado el umbral, remover definitivamente
                                print(f"Objeto {label} removido definitivamente después de {tiempo_perdido:.2f}s")
                                
                                # Remover de las piezas correctas si estaba registrada
                                if escenario_prev in piezas_correctas_por_escenario:
                                    piezas_correctas_por_escenario[escenario_prev].discard(label)
                                    # Solo remover de escenarios_anunciados_completos si el escenario realmente tiene menos de 5 piezas
                                    if len(piezas_correctas_por_escenario[escenario_prev]) < 5:
                                        escenarios_anunciados_completos.discard(escenario_prev)
                                
                                # Limpiar temporizador de incorrecto y anuncios
                                if label in objeto_incorrecto_tiempo:
                                    del objeto_incorrecto_tiempo[label]
                                if label in objetos_anunciados:
                                    objetos_anunciados.discard(label)
                                # Limpiar anuncios de incorrecto para este objeto
                                objetos_incorrectos_anunciados = {(l, esc) for (l, esc) in objetos_incorrectos_anunciados 
                                                                  if l != label}
                                
                                # Remover de objetos temporalmente perdidos
                                del objetos_temporalmente_perdidos[label]
                                
                                # Limpiar el historial de labels para este objeto
                                cx_prev_int, cy_prev_int = int(cx_prev), int(cy_prev)
                                tolerance = 50  # Misma tolerancia que en get_stable_label
                                keys_to_remove = []
                                for (hist_cx, hist_cy) in label_history.keys():
                                    distance = ((cx_prev_int - hist_cx) ** 2 + (cy_prev_int - hist_cy) ** 2) ** 0.5
                                    if distance < tolerance:
                                        keys_to_remove.append((hist_cx, hist_cy))
                                for key in keys_to_remove:
                                    del label_history[key]
                                
                                # Remover de objeto_posicion_anterior
                                del objeto_posicion_anterior[label]
                            else:
                                # Aún dentro del umbral, mantener como temporalmente perdido
                                objeto_posicion_anterior[label] = (cx_prev, cy_prev, escenario_prev, False)
                    else:
                        # El objeto ya no estaba presente, actualizar estado
                        objeto_posicion_anterior[label] = (cx_prev, cy_prev, escenario_prev, False)
            
            # Verificar objetos temporalmente perdidos que aún no han sido procesados en el bucle anterior
            # (para objetos que ya estaban perdidos antes de este frame)
            for label in list(objetos_temporalmente_perdidos.keys()):
                if label not in labels_presentes:
                    timestamp_perdida, escenario_perdido, cx_perdido, cy_perdido = objetos_temporalmente_perdidos[label]
                    tiempo_perdido = current_time - timestamp_perdida
                    
                    if tiempo_perdido >= umbral_perdida_temporal:
                        # Ha pasado el umbral, remover definitivamente
                        print(f"Objeto {label} removido definitivamente después de {tiempo_perdido:.2f}s (verificación periódica)")
                        
                        # Remover de las piezas correctas si estaba registrada
                        if escenario_perdido in piezas_correctas_por_escenario:
                            piezas_correctas_por_escenario[escenario_perdido].discard(label)
                            if len(piezas_correctas_por_escenario[escenario_perdido]) < 5:
                                escenarios_anunciados_completos.discard(escenario_perdido)
                        
                        # Limpiar temporizador de incorrecto y anuncios
                        if label in objeto_incorrecto_tiempo:
                            del objeto_incorrecto_tiempo[label]
                        if label in objetos_anunciados:
                            objetos_anunciados.discard(label)
                        objetos_incorrectos_anunciados = {(l, esc) for (l, esc) in objetos_incorrectos_anunciados 
                                                          if l != label}
                        
                        # Limpiar historial
                        cx_perdido_int, cy_perdido_int = int(cx_perdido), int(cy_perdido)
                        tolerance = 50
                        keys_to_remove = []
                        for (hist_cx, hist_cy) in label_history.keys():
                            distance = ((cx_perdido_int - hist_cx) ** 2 + (cy_perdido_int - hist_cy) ** 2) ** 0.5
                            if distance < tolerance:
                                keys_to_remove.append((hist_cx, hist_cy))
                        for key in keys_to_remove:
                            del label_history[key]
                        
                        # Remover de tracking
                        del objetos_temporalmente_perdidos[label]
                        if label in objeto_posicion_anterior:
                            del objeto_posicion_anterior[label]
            
            # Verificar si todos los escenarios tienen 5 piezas correctas y están completos (solo si el juego no está completo)
            if not juego_completo:
                todos_completos = True
                for escenario in escenarios_seleccionados:
                    # Verificar que el escenario tenga 5 piezas Y esté en escenarios_anunciados_completos
                    if (len(piezas_correctas_por_escenario.get(escenario, set())) < 5 or 
                        escenario not in escenarios_anunciados_completos):
                        todos_completos = False
                        break
            else:
                todos_completos = True  # Si ya está completo, mantener el estado
            
            # Si todos los escenarios tienen 5 piezas correctas, activar juego completo
            if todos_completos and not juego_completo:
                juego_completo = True
                # Reproducir sonido de correcto solo una vez
                if correcto_sound is not None and not sonido_reproducido:
                    try:
                        correcto_sound.set_volume(1.0)
                        correcto_sound.play()
                        sonido_reproducido = True
                        print("¡Juego completado! Sonido de correcto reproducido.")
                    except Exception as e:
                        print(f"Error al reproducir sonido de correcto: {e}")
            
            # Si el juego está completo, mostrar pantalla de éxito con botones
            if juego_completo and imagen_exito is not None:
                # Crear overlay negro con opacidad 50%
                overlay_negro = np.zeros_like(rectangulos_screen)
                overlay = cv2.addWeighted(rectangulos_screen, 0.5, overlay_negro, 0.5, 0)
                
                # Escalar la imagen para hacerla más grande (1.8x el tamaño original)
                scale_factor = 1.8
                img_h_original, img_w_original = imagen_exito.shape[:2]
                img_w = int(img_w_original * scale_factor)
                img_h = int(img_h_original * scale_factor)
                
                # Escalar la imagen
                if len(imagen_exito.shape) == 3 and imagen_exito.shape[2] == 4:
                    # Imagen con canal alpha
                    imagen_escalada = cv2.resize(imagen_exito, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Imagen sin alpha
                    imagen_escalada = cv2.resize(imagen_exito, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Calcular posición centrada
                center_x = view_width // 2
                center_y = view_height // 2
                x1 = center_x - img_w // 2
                y1 = center_y - img_h // 2
                x2 = x1 + img_w
                y2 = y1 + img_h
                
                # Asegurar que la imagen quepa en la pantalla (ajustar si es necesario)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > view_width:
                    x2 = view_width
                if y2 > view_height:
                    y2 = view_height
                
                # Calcular el área de la imagen a usar
                img_x1 = max(0, -x1)
                img_y1 = max(0, -y1)
                img_x2 = img_w - max(0, x2 - view_width)
                img_y2 = img_h - max(0, y2 - view_height)
                
                # Si la imagen tiene canal alpha, usar composición con alpha
                if len(imagen_escalada.shape) == 3 and imagen_escalada.shape[2] == 4:
                    # Extraer RGB y alpha
                    img_rgb = imagen_escalada[img_y1:img_y2, img_x1:img_x2, :3]
                    img_alpha = imagen_escalada[img_y1:img_y2, img_x1:img_x2, 3:4] / 255.0
                    alpha_3ch = np.repeat(img_alpha, 3, axis=2)
                    
                    # Componer la imagen con alpha sobre el overlay
                    overlay[y1:y2, x1:x2] = (overlay[y1:y2, x1:x2] * (1 - alpha_3ch) + 
                                              img_rgb * alpha_3ch).astype(np.uint8)
                else:
                    # Si no tiene alpha, copiar directamente
                    overlay[y1:y2, x1:x2] = imagen_escalada[img_y1:img_y2, img_x1:img_x2, :3]
                
                # Dibujar botones arriba de la imagen
                button_height = 80
                button_width = 200
                button_spacing = 50
                button_y = y1 - button_height - 40  # 40 píxeles arriba de la imagen
                
                # Botón "Volver a jugar" (izquierda)
                button_volver_x1 = center_x - button_width - button_spacing // 2
                button_volver_y1 = button_y
                button_volver_x2 = button_volver_x1 + button_width
                button_volver_y2 = button_volver_y1 + button_height
                
                # Botón "Salir" (derecha)
                button_salir_x1 = center_x + button_spacing // 2
                button_salir_y1 = button_y
                button_salir_x2 = button_salir_x1 + button_width
                button_salir_y2 = button_salir_y1 + button_height
                
                # Dibujar botón "Volver a jugar" (verde)
                cv2.rectangle(overlay, (button_volver_x1, button_volver_y1), 
                             (button_volver_x2, button_volver_y2), (0, 200, 0), -1)
                cv2.rectangle(overlay, (button_volver_x1, button_volver_y1), 
                             (button_volver_x2, button_volver_y2), (255, 255, 255), 3)
                
                # Texto "Volver a jugar"
                texto_volver = "Volver a jugar"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(texto_volver, font, font_scale, thickness)
                text_x_volver = button_volver_x1 + (button_width - text_width) // 2
                text_y_volver = button_volver_y1 + (button_height + text_height) // 2
                cv2.putText(overlay, texto_volver, (text_x_volver, text_y_volver), 
                           font, font_scale, (255, 255, 255), thickness)
                
                # Dibujar botón "Salir" (rojo)
                cv2.rectangle(overlay, (button_salir_x1, button_salir_y1), 
                             (button_salir_x2, button_salir_y2), (0, 0, 200), -1)
                cv2.rectangle(overlay, (button_salir_x1, button_salir_y1), 
                             (button_salir_x2, button_salir_y2), (255, 255, 255), 3)
                
                # Texto "Salir"
                texto_salir = "Salir"
                (text_width, text_height), baseline = cv2.getTextSize(texto_salir, font, font_scale, thickness)
                text_x_salir = button_salir_x1 + (button_width - text_width) // 2
                text_y_salir = button_salir_y1 + (button_height + text_height) // 2
                cv2.putText(overlay, texto_salir, (text_x_salir, text_y_salir), 
                           font, font_scale, (255, 255, 255), thickness)
                
                rectangulos_screen = overlay
            
            # Escalar a la resolución del videobeam antes de mostrar
            rectangulos_screen_scaled = scale_to_videobeam(rectangulos_screen)
            cv2.imshow(window_name, rectangulos_screen_scaled)
            
            # Asegurar que la ventana esté en pantalla completa y en la segunda pantalla en cada frame
            try:
                cv2.moveWindow(window_name, 1920, 0)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except:
                pass
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        # Detener streams al salir
        rgb_stream.stop()
        depth_stream.stop()
    
    return None
