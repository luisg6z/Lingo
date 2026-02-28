import cv2
import numpy as np
from openni import openni2
from tqdm import tqdm
from collections import deque
import pygame
import time
import os
import customtkinter as ctk
from skimage.measure import label, regionprops
import random
import mediapipe as mp
import json
import math
import re
import threading
import niveles_clasificacion
from niveles_clasificacion import mostrar_seleccion_niveles_clasificacion
from absurdos_seleccion import mostrar_seleccion_absurdos
import importlib.util
import sys
# Import from absurdos-visuales folder (hyphens not allowed in Python module names)
spec = importlib.util.spec_from_file_location("absurdos_visuales", "absurdos-visuales/absurdos_visuales.py")
absurdos_visuales_module = importlib.util.module_from_spec(spec)
sys.modules["absurdos_visuales"] = absurdos_visuales_module
spec.loader.exec_module(absurdos_visuales_module)
juego_absurdos_reconocimiento_voz = absurdos_visuales_module.juego_absurdos_reconocimiento_voz
import speech_recognition as sr
# Import confetti system
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from src.features.confetti import ConfettiSystem
from src.components.rectangular_button import draw_rectangular_button, is_point_in_rectangular_button
from src.components.circular_button import draw_circular_button, is_point_in_circular_button
# Juego historia: voice listen and Ollama validation
_historia_story_voice = None
def _get_historia_story_voice():
    global _historia_story_voice
    if _historia_story_voice is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        story_voice_path = os.path.join(project_root, "src", "features", "juego-historia", "story_voice.py")
        spec_sv = importlib.util.spec_from_file_location("historia_story_voice", story_voice_path)
        mod_sv = importlib.util.module_from_spec(spec_sv)
        spec_sv.loader.exec_module(mod_sv)
        _historia_story_voice = mod_sv
    return _historia_story_voice
import spacy
import pyttsx3

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Import font utilities
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
try:
    from src.core.font_utils import get_ubuntu_font
except ImportError:
    get_ubuntu_font = None


def put_text_ubuntu(img, text, position, font_scale, color, thickness, line_type=cv2.LINE_AA, font_face=None, bold=False):
    """
    Render text using Ubuntu font from resources/fonts.
    This replaces cv2.putText to ensure consistent Ubuntu font usage.
    font_face parameter is accepted for compatibility but ignored (always uses Ubuntu).
    bold: If True, use bold Ubuntu font variant
    """
    if _PIL_AVAILABLE:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                # Always use Ubuntu font from resources/fonts
                if get_ubuntu_font:
                    font = get_ubuntu_font(font_scale=font_scale, bold=bold)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            x, y = position
            # Get text bounding box for accurate positioning
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
            except AttributeError:
                bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, 0, 0)
            color_rgb = (color[2], color[1], color[0])
            draw.text((x, y), text, fill=color_rgb, font=font)
            img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img[:] = img_result[:]
        except Exception as e:
            # Fallback to OpenCV if PIL fails
            fallback_face = font_face if font_face else cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, position, fallback_face, font_scale, color, thickness, line_type)
    else:
        # Fallback to OpenCV if PIL not available
        fallback_face = font_face if font_face else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, fallback_face, font_scale, color, thickness, line_type)


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
    from src.core.font_utils import get_ubuntu_font
    
    words = text.split()
    word_positions = []
    x_current = start_x
    y_current = start_y
    line_index = 0
    
    try:
        from PIL import Image, ImageDraw
        font = get_ubuntu_font(font_scale=font_scale, bold=bold)
        img_pil = Image.new('RGB', (100, 100), (0, 0, 0))  # Temporary image for measurement
        draw = ImageDraw.Draw(img_pil)
        
        for word in words:
            # Measure word width (with space after it)
            word_with_space = word + " "
            try:
                bbox = draw.textbbox((0, 0), word_with_space, font=font)
                word_width = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font.getbbox(word_with_space) if hasattr(font, "getbbox") else (0, 0, 0, 0)
                word_width = bbox[2] - bbox[0]
            
            # Check if word fits on current line
            if x_current + word_width > max_width and x_current > start_x:
                # Move to next line
                y_current += int(font_scale * 35)  # Approximate line height
                x_current = start_x
                line_index += 1
            
            # Store word position
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
        
    except Exception as e:
        # Fallback to OpenCV measurement
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


def _put_text_safe_historia(img, text, position, font_face, font_scale, color, thickness, line_type=cv2.LINE_AA, bold=False):
    """
    Render text using Ubuntu font. Wrapper for compatibility.
    """
    put_text_ubuntu(img, text, position, font_scale, color, thickness, line_type, font_face, bold=bold)


# Variable global para el modelo de sentence-transformers (se carga al inicio)
sentence_transformer_model = None

# TTS: speak card name when selected in juego-historia (non-blocking, daemon thread)
# Articles and pronunciation for subjects (characters) and places
_HISTORIA_TTS_SUJETO = {
    "Niño": "El niño",
    "Niña": "La niña",
    "Doctor": "El doctor",
    "Maestra": "La maestra",
    "Policia": "El policía",
    "Perro": "El perro",
}
_HISTORIA_TTS_LUGAR = {
    "Calle": "La calle",
    "Clinica": "La clínica",
    "Estacion-Policia": "La estación de policía",
    "Escuela": "La escuela",
    "Casa": "La casa",
    "Parque": "El parque",
}
# Pronunciation only for actions (no article)
_HISTORIA_TTS_ACCION = {
    "Dar": "Dar",
    "Ayudar": "Ayudar",
    "Correr": "Correr",
    "Jugar": "Jugar",
    "Llamar": "Llamar",
    "Trabajar": "Trabajar",
}

def _historia_tts_speak(name, tipo=None):
    """Speak the given name using pyttsx3 in a daemon thread (for sujeto/accion/lugar selection).
    tipo: 'sujeto' | 'accion' | 'lugar' to use the correct article; None = use name as-is with pronunciation fix.
    """
    if tipo == "sujeto":
        text = _HISTORIA_TTS_SUJETO.get(name, name)
    elif tipo == "lugar":
        text = _HISTORIA_TTS_LUGAR.get(name, name)
    elif tipo == "accion":
        text = _HISTORIA_TTS_ACCION.get(name, name)
    else:
        text = name
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            for v in engine.getProperty("voices"):
                if "spanish" in v.name.lower() or "español" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()

def load_and_validate_dmax_map(coordenadas):
    """
    Carga y valida el archivo dmax_map.txt contra las coordenadas proporcionadas.
    
    Args:
        coordenadas: Diccionario con las coordenadas de calibración
    
    Returns:
        tuple: (dmax_map_reshaped, w, h) si es exitoso, (None, None, None) si hay error
    """
    try:
        # Cargar el archivo dmax_map.txt
        dmax_map = np.loadtxt("config/dmax_map.txt", dtype=np.uint16)
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'config/dmax_map.txt'.")
        print("Por favor, ejecuta la calibración primero.")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: El archivo 'dmax_map.txt' tiene un formato inválido: {e}")
        return None, None, None
    
    # Calcular dimensiones esperadas
    xw_min = coordenadas.get("xw_min", 0)
    xw_max = coordenadas.get("xw_max", 0)
    yw_min = coordenadas.get("yw_min", 0)
    yw_max = coordenadas.get("yw_max", 0)
    
    w = xw_max - xw_min
    h = yw_max - yw_min
    
    expected_size = w * h
    actual_size = dmax_map.size
    
    # Validar dimensiones
    if actual_size != expected_size:
        print("=" * 70)
        print("ERROR: Incompatibilidad de dimensiones entre dmax_map.txt y coordenadas")
        print("=" * 70)
        print(f"Dimensiones esperadas: {w} x {h} = {expected_size} elementos")
        print(f"Dimensiones del archivo: {actual_size} elementos")
        
        # Intentar encontrar dimensiones posibles
        posibles_dimensiones = []
        for i in range(300, 500):
            if actual_size % i == 0:
                j = actual_size // i
                posibles_dimensiones.append((i, j))
        
        if posibles_dimensiones:
            print(f"\nDimensiones posibles del archivo:")
            for dim in posibles_dimensiones[:3]:
                print(f"  - {dim[0]} x {dim[1]} o {dim[1]} x {dim[0]}")
        
        print("\nSOLUCIÓN:")
        print("1. Ejecuta la calibración para regenerar dmax_map.txt con las dimensiones correctas:")
        print("   python calibrate_area.py")
        print("\n2. O ajusta las coordenadas en 'ultima_configuracion_coordenadas.json'")
        print("   para que coincidan con las dimensiones del archivo actual.")
        print("=" * 70)
        return None, None, None
    
    try:
        # Hacer reshape
        dmax_map_reshaped = dmax_map.reshape((h, w))
        return dmax_map_reshaped, w, h
    except ValueError as e:
        print(f"ERROR al hacer reshape del dmax_map: {e}")
        return None, None, None

def piano(device, videobeam_resolution=(1280, 800), min_contour_area=500, max_contour_area=20000):
    # Cargar las coordenadas desde el archivo JSON
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        config = json.load(file)

    # Cargar y validar dmax_map
    dmax_map, w, h = load_and_validate_dmax_map(config)
    if dmax_map is None:
        return

    # Asignar las coordenadas del viewport y la ventana
    xv_min = config['xv_min']
    xv_max = config['xv_max']
    yv_min = config['yv_min']
    yv_max = config['yv_max']
    xw_min = config['xw_min']
    xw_max = config['xw_max']
    yw_min = config['yw_min']
    yw_max = config['yw_max']

    # Cálculo de los factores de escala para el mapeo de ventana a viewport
    sx = float(xv_max - xv_min) / (xw_max - xw_min)
    sy = float(yv_max - yv_min) / (yw_max - yw_min)

    def window_to_viewport(x_touch, y_touch):
        """Convierte coordenadas de ventana a viewport utilizando los factores de escala"""
        x_viewport = int(xv_min + (x_touch) * sx)
        y_viewport = int(yv_min + (y_touch) * sy)

        # x_viewport = np.clip(x_viewport, 0, view_width - 1)
        # y_viewport = np.clip(y_viewport, 0, view_height - 1)
        return x_viewport, y_viewport

    rgb_stream = device.create_color_stream()
    rgb_stream.start()

    pygame.mixer.init()

    # Configuración del videobeam (según resolución)
    view_width, view_height = videobeam_resolution

    # Lista para almacenar las formas detectadas
    captured_shapes = []
    capture_done = False  # Estado para controlar si ya se realizó la captura

    # Diccionario de sonidos por color
    sounds = {
        "Rojo": pygame.mixer.Sound('cardinal.mp3'),
        "Verde": pygame.mixer.Sound('serpiente.mp3'),
        "Azul": pygame.mixer.Sound('delfin.mp3'),
        "Amarillo": pygame.mixer.Sound('pollito.mp3'),
        "Negro": pygame.mixer.Sound('oso.mp3'),
        "Morado": pygame.mixer.Sound('gato.mp3'),
        "Naranja": pygame.mixer.Sound('perro.mp3')
    }

    # Diccionario para el estado de las figuras
    figure_status = {color: {'active': False, 'timer': 0} for color in sounds.keys()}

    # Diccionario para mapear nombres de colores a valores BGR
    color_bgr = {
        "Rojo": (0, 0, 255),
        "Verde": (0, 255, 0),
        "Azul": (255, 0, 0),
        "Amarillo": (0, 255, 255),
        "Naranja": (0, 165, 255),
        "Morado": (128, 0, 128),
        "Negro": (0, 0, 0)
    }

    fps = 30  # Valor predeterminado
    prev_time = time.time()

    while True:
        frame = rgb_stream.read_frame()
        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)

        # Convertir los canales de color de RGB a BGR y voltear horizontalmente
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        rgb_data = cv2.flip(rgb_data, 1)

        # Definir la región de interés (ROI) dentro del área calibrada
        roi = rgb_data[yw_min:yw_max, xw_min:xw_max]

        # Convertir ROI de BGR a HSV para procesar colores
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Definir rangos de colores
        color_ranges = {
            "Rojo": ([170, 50, 50], [180, 255, 255]),
            "Verde": ([40, 50, 50], [90, 255, 255]),
            "Azul": ([90, 50, 50], [130, 255, 255]),
            "Amarillo": ([20, 100, 100], [30, 255, 255]),
            "Naranja": ([0, 100, 100], [10, 255, 255]),
            "Morado": ([130, 50, 50], [160, 255, 255]),
            "Negro": ([0, 0, 0], [180, 255, 30])
        }

        current_shapes = []  # Almacenar las formas detectadas en este frame

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

            # Aplicar operaciones morfológicas para limpiar la máscara
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Detectar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_contour_area < area < max_contour_area:
                    # Aproximación del contorno para identificar la forma
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) == 3:
                        shape = "Triangulo"
                    elif len(approx) == 4:
                        aspect_ratio = cv2.boundingRect(approx)[2] / float(cv2.boundingRect(approx)[3])
                        shape = "Cuadrado" if 0.85 <= aspect_ratio <= 1.15 else "Rectangulo"
                    elif len(approx) > 4:
                        shape = "Circulo"
                    else:
                        shape = "Forma no identificada"

                    current_shapes.append((shape, color_name, cnt))

                    # Dibujar el contorno y mostrar el nombre de la figura en la imagen
                    cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(cnt)
                    label_text = f"{shape} {color_name}"
                    put_text_ubuntu(roi, label_text, (x, y - 10), 0.6, (255, 0, 0), 2)

        cv2.imshow("Formas Detectadas", roi)

        # Capturar las formas cuando se presiona 'c'
        if not capture_done and cv2.waitKey(1) & 0xFF == ord('c'):
            print(f"{len(current_shapes)} formas capturadas.")
            print("Por favor, retire las piezas de la mesa y presione 'r' para continuar.")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    print("Piezas retiradas. Procediendo con la detección de toques.")
                    break
            captured_shapes.extend(current_shapes)
            capture_done = True  # Marcar la captura como realizada

            depth_stream = device.create_depth_stream()
            depth_stream.start()

            dmax_map = dmax_map - 5
            dmin_map = dmax_map - 10  # dmin está más cerca de la cámara que dmax

            # Parámetros del filtrado temporal
            touch_history = deque(maxlen=5)  # Guardar máscaras de los últimos 5 cuadros
            touch_duration_threshold = 3  # Requiere que el toque persista en al menos 3 cuadros

            # Crear ventana para el videobeam
            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

        if capture_done:
            # Leer la imagen de profundidad
            depth_frame = depth_stream.read_frame()
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)

            # Recortar la región de interés en la imagen de profundidad
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

            # Crear la máscara de toque utilizando dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

            # Aplicar filtros para eliminar ruido
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
            touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
            touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))
            
            # Aplicar un umbral más estricto para reducir toques fantasma (180 en lugar de 150)
            _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
            touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

            # Historial de toques para filtrado temporal
            touch_history.append(touch_mask_final)
            accumulated_mask = np.sum(touch_history, axis=0)
            accumulated_mask = np.clip(accumulated_mask, 0, 255).astype(np.uint8)

            # Iniciar pantalla del videobeam
            videobeam_screen.fill(0)

            # Para depurar y visualizar los toques
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_final, connectivity=8)
            # for i in range(1, num_labels):
            #     if stats[i, cv2.CC_STAT_AREA] >= 100:
            #         centroid = centroids[i]
            #         x_touch, y_touch = int(centroid[0]), int(centroid[1])
            #         x_viewport, y_viewport = window_to_viewport(x_touch, y_touch)
            #         cv2.circle(videobeam_screen, (x_viewport, y_viewport), 10, (255, 255, 255), 2)

            # Actualizar el estado de las figuras
            for color in figure_status.keys():
                if figure_status[color]['active']:
                    figure_status[color]['timer'] -= 1 / fps  # Restar el tiempo transcurrido (1/fps)
                    if figure_status[color]['timer'] <= 0:
                        figure_status[color]['active'] = False
                        figure_status[color]['timer'] = 0

            for shape, color_name, cnt in captured_shapes:
                # Crear máscara de la figura
                mask_shape = np.zeros_like(touch_mask_final)
                cv2.drawContours(mask_shape, [cnt], -1, 255, thickness=cv2.FILLED)

                # Verificar toque usando máscara acumulada (requiere persistencia en varios cuadros para evitar fantasmas)
                touch_in_shape = cv2.bitwise_and(accumulated_mask, mask_shape)
                min_overlap_pixels = 60  # Mínimo de píxeles presentes en al menos 2 cuadros
                if np.sum(touch_in_shape >= 2 * 255) >= min_overlap_pixels:
                    if not figure_status[color_name]['active']:
                        print(f'Toque detectado en el área del color {color_name}')

                        # Reproducir sonido correspondiente
                        if sounds.get(color_name):
                            sounds[color_name].play()

                        # Obtener la duración del sonido
                        sound_duration = sounds[color_name].get_length()
                        figure_status[color_name]['active'] = True
                        figure_status[color_name]['timer'] = sound_duration  # Tiempo en segundos
                else:
                    pass  # No desactivamos aquí para permitir que el temporizador controle la visibilidad

                # Si la figura está activa, dibujarla
                if figure_status[color_name]['active']:
                    # Transformar el contorno de la figura al espacio del viewport
                    cnt_vp = []
                    for point in cnt:
                        x_win, y_win = point[0]
                        x_viewport, y_viewport = window_to_viewport(x_win, y_win)
                        cnt_vp.append([[x_viewport, y_viewport]])
                    cnt_vp = np.array(cnt_vp, dtype=np.int32)

                    # Dibujar la figura en el videobeam_screen con su color correspondiente
                    cv2.drawContours(videobeam_screen, [cnt_vp], -1, color_bgr.get(color_name, (255, 255, 255)), thickness=cv2.FILLED)

            # Mostrar la pantalla del videobeam
            cv2.namedWindow("Videobeam", cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow("Videobeam", 1920, 0)
            cv2.setWindowProperty("Videobeam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Videobeam", videobeam_screen)

            # Mostrar la máscara de toque para depuración (opcional)
            cv2.imshow("Máscara de Toque", touch_mask_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calcular fps
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 0:
            fps = 1 / elapsed_time
        prev_time = current_time

    rgb_stream.stop()
    if capture_done:
        depth_stream.stop()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    
def show_captured_shape(contours, color_names, sounds, view_width, view_height, x_cal, y_cal, w_cal, h_cal, M):
    # Crear una pantalla blanca para mostrar las formas capturadas
    white_screen = np.ones((view_height, view_width, 3), dtype=np.uint8) * 0

    for contour, color_name in zip(contours, color_names):
        # **CHANGE 4: Apply perspective transform to contours**
        # Ajustar las coordenadas del contorno a las coordenadas completas de la imagen
        contour = contour + np.array([[[x_cal, y_cal]]], dtype=np.int32)

        # Transformar el contorno usando la transformación de perspectiva
        transformed_contour = cv2.perspectiveTransform(contour.astype(np.float32), M).astype(np.int32)
        # **End of CHANGE 4**

        # Asignación de colores basada en el nombre del color
        if color_name == "Rojo":
            color = (0, 0, 255)  # Rojo (en BGR)
        elif color_name == "Verde":
            color = (0, 255, 0)  # Verde
        elif color_name == "Azul":
            color = (255, 0, 0)  # Azul
        elif color_name == "Amarillo":
            color = (0, 255, 255)  # Amarillo
        elif color_name == "Naranja":
            color = (0, 165, 255)  # Naranja (tono estándar de BGR para naranja)
        elif color_name == "Morado":
            color = (255, 0, 255)  # Morado
        elif color_name == "Negro":
            color = (0, 0, 0)  # Negro
        else:
            color = (255, 255, 255)  # Color por defecto (blanco, si no se encuentra)

        if sounds.get(color_name):
            sounds[color_name].play()

        # Dibuja el contorno con el color asignado
        cv2.drawContours(white_screen, [transformed_contour], -1, color, -1)

    # Mostrar las formas capturadas en la pantalla del videobeam
    cv2.namedWindow("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow("Pantalla de Videobeam", 1920, 0)
    cv2.setWindowProperty("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Pantalla de Videobeam", cv2.flip(white_screen, 1))

    cv2.waitKey(1000)

    white_screen = np.ones((view_height, view_width, 3), dtype=np.uint8) * 0
    cv2.imshow("Pantalla de Videobeam", cv2.flip(white_screen, 1))
    cv2.waitKey(1)

def juego_memoria(device):
    # Cargar las coordenadas desde el archivo JSON
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        config = json.load(file)

    # Asignar las coordenadas del viewport y la ventana
    xv_min = config['xv_min']
    xv_max = config['xv_max']
    yv_min = config['yv_min']
    yv_max = config['yv_max']
    xw_min = config['xw_min']
    xw_max = config['xw_max']
    yw_min = config['yw_min']
    yw_max = config['yw_max']

    animal_images = {
        "Leon": "./images/leon.png",
        "Elefante": "./images/elefante.png",
        "Gato": "./images/gato.png",
        "Perro": "./images/perro.png",
        "Pajarito": "./images/pajarito.png"
    }

    # Inicialización de sonidos
    pygame.mixer.init()
    correct_sound = pygame.mixer.Sound('./sounds/correct.mp3')
    wrong_sound = pygame.mixer.Sound('./sounds/incorrect.mp3')
    victory_sound = pygame.mixer.Sound('./sounds/victory.mp3')  # Sonido de victoria
    confetti_video = cv2.VideoCapture('./videos/confetti.mp4')

    view_width = 1280
    view_height = 800

    TIME_TO_DISPLAY = 1.5  # Tiempo en segundos antes de mostrar el animal cuando se quita una figura
    TIME_BEFORE_REMOVE = 3  # Tiempo para que las imágenes se mantengan antes de eliminarse tras una coincidencia

    ancho_fisico_cm = 192  # Ancho físico de la proyección en cm
    alto_fisico_cm = 120   # Alto físico de la proyección en cm

    # Calcular la relación píxeles/cm
    pixels_per_cm_width = view_width / ancho_fisico_cm
    pixels_per_cm_height = view_height / alto_fisico_cm

    tamano_ficha_cm = 6  # Lado más largo en cm
    tamano_ficha_px = tamano_ficha_cm * pixels_per_cm_width  # Convertir a píxeles
    
    # Tamaño deseado en cm
    tamano_fichas_cm = 12

    # Tamaño en píxeles
    w_ficha_px = int(tamano_fichas_cm * pixels_per_cm_width)
    h_ficha_px = int(tamano_fichas_cm * pixels_per_cm_height)

    def load_animal_images(animal_images):
        loaded_images = {}
        for name, path in animal_images.items():
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Leer con canal alfa si está disponible
            if image is not None:
                loaded_images[name] = image
            else:
                print(f"Error al cargar la imagen de {name}")
        return loaded_images

    def detect_color_and_shape(image, min_contour_area=500):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            "Rojo": ([170, 50, 50], [180, 255, 255]),
            "Verde": ([40, 50, 50], [90, 255, 255]),
            "Azul": ([90, 50, 50], [130, 255, 255]),
            "Amarillo": ([20, 100, 100], [30, 255, 255]),
            "Naranja": ([0, 100, 100], [10, 255, 255]),
            "Morado": ([130, 50, 50], [160, 255, 255]),
        }

        detected_shapes = []

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) > 4:
                        shape = "Circulo"
                    elif len(approx) == 4:
                        shape = "Cuadrado"
                    elif len(approx) == 3:
                        shape = "Triangulo"

                    detected_shapes.append((shape, color_name, cnt))

        return detected_shapes

    def assign_animal_pairs(detected_shapes, animal_images):
        num_figures = len(detected_shapes)
        if num_figures % 2 != 0:
            print("El número de figuras debe ser par")
            return {}

        selected_animals = random.sample(list(animal_images.keys()), num_figures // 2)
        animal_pairs = selected_animals * 2
        random.shuffle(animal_pairs)

        figure_animal_map = {}
        for i, (shape, color, figure) in enumerate(detected_shapes):
            figure_animal_map[(shape, color)] = animal_pairs[i]
            print(figure_animal_map)

        return figure_animal_map

    def draw_animal_on_figure(image, figure, xv_min, yv_min, xv_max, yv_max, xw_min, xw_max, yw_min, yw_max, w_ficha_px, h_ficha_px, animal_image):
        # Obtener el centro de la figura
        M_moment = cv2.moments(figure)
        if M_moment["m00"] != 0:
            cX_window = int(M_moment["m10"] / M_moment["m00"])
            cY_window = int(M_moment["m01"] / M_moment["m00"])
        else:
            x_window, y_window, w_window, h_window = cv2.boundingRect(figure)
            cX_window = x_window + w_window // 2
            cY_window = y_window + h_window // 2

        # Calcular los factores de escala para el mapeo de coordenadas de ventana a viewport
        sx = float(xv_max - xv_min) / (xw_max - xw_min)
        sy = float(yv_max - yv_min) / (yw_max - yw_min)

        # Mapeo de coordenadas de la ventana al viewport
        x_viewport = int(xv_min + (cX_window * sx))
        y_viewport = int(yv_min + (cY_window * sy))

        # Calcular la posicion superior izquierda para centrar la imagen
        x_animal = int(x_viewport - w_ficha_px / 2)
        y_animal = int(y_viewport - h_ficha_px / 2)

        # Asegurarse de que las coordenadas estén dentro de los límites
        x_animal = max(0, min(image.shape[1] - w_ficha_px, x_animal))
        y_animal = max(0, min(image.shape[0] - h_ficha_px, y_animal))

        # Redimensionar la imagen del animal al tamaño deseado
        resized_animal = cv2.resize(animal_image, (w_ficha_px, h_ficha_px), interpolation=cv2.INTER_AREA)

        # Insertar la imagen del animal en la pantalla del videobeam
        if resized_animal.shape[2] == 4:
            alpha_s = resized_animal[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                image[y_animal:y_animal+h_ficha_px, x_animal:x_animal+w_ficha_px, c] = (
                    alpha_s * resized_animal[:, :, c] +
                    alpha_l * image[y_animal:y_animal+h_ficha_px, x_animal:x_animal+w_ficha_px, c]
                )
        else:
            image[y_animal:y_animal+h_ficha_px, x_animal:x_animal+w_ficha_px] = resized_animal

    def is_figure_removed(shape, color, detected_shapes):
        for detected_shape, detected_color, detected_figure in detected_shapes:
            if detected_shape == shape and detected_color == color:
                updated_positions[(shape, color)] = detected_figure
                return False
        print(f"Figura {shape}, Color {color} eliminada del tablero.") 
        return True

    loaded_images = load_animal_images(animal_images)
    # Inicializar streams
    rgb_stream = device.create_color_stream()
    rgb_stream.start()

    # Variables para detectar movimiento
    previous_frame = None
    movement_threshold = 1000  # Umbral para detectar movimiento (ajusta según lo que consideres movimiento)

    assigned_animals = False
    captured_figures = None
    figure_animal_map = {}

    removal_times = {}
    updated_positions = {}

    selected_figures = []
    selected_animals = []
    correct_pairs = []
    last_evaluated_time = None
    evaluation_done = False  
    correct_time = None  

    videobeam_screen = np.ones((view_height, view_width, 3), dtype=np.uint8) * 0

    while True:
        # Leer el frame actual de la cámara RGB
        frame = rgb_stream.read_frame()
        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        current_frame = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        current_frame = cv2.flip(current_frame, 1)

        # Convertir a escala de grises para comparar la diferencia
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray_current
            continue

        # Calcular la diferencia entre el frame actual y el anterior
        frame_diff = cv2.absdiff(previous_frame, gray_current)
        _, thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Contar los píxeles que han cambiado significativamente
        movement_area = np.sum(thresh_diff > 0)

        if movement_area > movement_threshold:
            print("Movimiento detectado, pausa en el stream")
            time.sleep(0.2)  # Pausar por 1 segundo si hay movimiento
        else:
            # Continuar con la lógica del juego
            bgr_data = current_frame[yw_min:yw_max, xw_min:xw_max]
            detected_shapes = detect_color_and_shape(bgr_data)

            for _, color, figure in detected_shapes:
                figure_shifted = figure + np.array([xw_min, yw_min])  
                cv2.drawContours(bgr_data, [figure_shifted], -1, (0, 255, 0), 2)
            cv2.imshow("Detección de Formas y Colores", bgr_data)

            if not assigned_animals:
                put_text_ubuntu(bgr_data, "Presiona 'c' para asignar animales", (20, 30), 0.7, (0, 255, 0), 2)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    captured_figures = detected_shapes[:]
                    if captured_figures and len(captured_figures) % 2 == 0:
                        figure_animal_map = assign_animal_pairs(captured_figures, animal_images)
                        assigned_animals = True
                    else:
                        print("Asegúrate de que haya un número par de figuras.")
            else:
                videobeam_screen[:] = 0
                current_time = time.time()

                if len(selected_figures) == 2 and not evaluation_done:
                    if selected_animals[0] == selected_animals[1]:
                        print("¡Coincidencia!")
                        correct_sound.play()
                        correct_pairs.append(selected_figures)
                        correct_time = current_time
                    else:
                        print("No coinciden")
                        wrong_sound.play()

                    evaluation_done = True  
                    last_evaluated_time = current_time  

                if correct_time and current_time - correct_time > TIME_BEFORE_REMOVE:
                    for pair in selected_figures:
                        if pair in figure_animal_map:
                            del figure_animal_map[pair]
                    selected_figures.clear()
                    selected_animals.clear()
                    correct_time = None

                for shape, color, captured_figure in captured_figures:
                    if is_figure_removed(shape, color, detected_shapes):
                        if (shape, color) not in removal_times:
                            removal_times[(shape, color)] = current_time

                        if current_time - removal_times[(shape, color)] > TIME_TO_DISPLAY:
                            print("valeeee")
                            animal_name = figure_animal_map.get((shape, color), None)
                            if animal_name:
                                print(animal_name)  
                                animal_image = loaded_images.get(animal_name)
                                if animal_image is not None and animal_image.size > 0:
                                    figure_to_draw = updated_positions.get((shape, color), captured_figure)
                                    adjusted_figure = figure_to_draw
                                    draw_animal_on_figure(
                                        videobeam_screen, 
                                        adjusted_figure, 
                                        xv_min, yv_min, xv_max, yv_max, 
                                        xw_min, xw_max, yw_min, yw_max, 
                                        w_ficha_px, h_ficha_px,
                                        animal_image
                                    )
                                    if (shape, color) not in selected_figures:
                                        selected_figures.append((shape, color))
                                        selected_animals.append(animal_name)
                                        evaluation_done = False
                    else:
                        if (shape, color) in selected_figures:
                            index = selected_figures.index((shape, color))
                            del selected_figures[index]
                            del selected_animals[index]
                        
                        if (shape, color) in removal_times:
                            del removal_times[(shape, color)] 

                cv2.namedWindow("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow("Pantalla de Videobeam", 1920, 0)
                cv2.setWindowProperty("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Pantalla de Videobeam", videobeam_screen)

                if correct_pairs is not None and captured_figures is not None:
                    if len(correct_pairs) == len(captured_figures) // 2:
                        victory_sound.play()
                        while True:
                            ret, frame = confetti_video.read()
                            if not ret:
                                break
                            cv2.imshow("Pantalla de Videobeam", frame)
                            if cv2.waitKey(40) & 0xFF == ord('q'):
                                break
                        confetti_video.release()
                        cv2.destroyWindow("Pantalla de Videobeam")
                        break

        previous_frame = gray_current

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rgb_stream.stop()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

def juego_clasificacion(device, modo_clasificacion, piezas_fisicas, num_piezas):
    # Cargar las coordenadas desde el archivo JSON
    import json
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Cargar y validar dmax_map
    dmax_map, w, h = load_and_validate_dmax_map(coordenadas)
    if dmax_map is None:
        return

    # Ajustar el dmax_map (restar offset)
    dmax_map = dmax_map - 7
    dmin_map = dmax_map - 50

    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800

    # Inicializar pygame
    pygame.init()
    pygame.mixer.init()

    # Crear una pantalla negra para el videobeam
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    mostrar_ovalos = True

    # Ejes de las elipses (óvalos)
    eje_horizontal = int((xv_max - xv_min) * 0.25)  # Eje horizontal más pequeño
    eje_vertical = int((yv_max - yv_min) * 0.50)    # Aumentar eje vertical para hacerlo más alargado

    # Crear los dos óvalos en el área acotada, asegurando que no choquen
    area1_center = (int((xv_min + xv_max) * 0.35), int((yv_min + yv_max) * 0.5))
    area1_axes = (eje_horizontal, eje_vertical)

    area2_center = (int((xv_min + xv_max) * 0.65), int((yv_min + yv_max) * 0.5))
    area2_axes = (eje_horizontal, eje_vertical)

    DURACION_MARCA = 2.0

    # Definir las figuras y colores
    figuras_virtuales = ['círculo', 'cuadrado', 'triángulo', 'estrella']
    colores_virtuales = {
        'amarillo': (0, 255, 255),
        'rojo': (0, 0, 255),
        'verde': (0, 255, 0),
        'azul': (255, 0, 0),
        'naranja': (0, 165, 255),
        'morado': (128, 0, 128)
    }

    colores_bgr = {
        'amarillo': (0, 255, 255),
        'rojo': (0, 0, 255),
        'verde': (0, 255, 0),
        'azul': (255, 0, 0),
        'naranja': (0, 165, 255),
        'morado': (128, 0, 128)
    }

    def detect_color_and_shape(image, min_contour_area=500):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            "Rojo": ([170, 50, 50], [180, 255, 255]),
            "Verde": ([40, 50, 50], [90, 255, 255]),
            "Azul": ([90, 50, 50], [130, 255, 255]),
            "Amarillo": ([20, 100, 100], [30, 255, 255]),
            "Naranja": ([0, 100, 100], [10, 255, 255]),
            "Morado": ([130, 50, 50], [160, 255, 255]),
        }

        detected_shapes = []

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) > 4:
                        shape = "círculo"
                    elif len(approx) == 4:
                        shape = "cuadrado"
                    elif len(approx) == 3:
                        shape = "triángulo"
                    else:
                        shape = "desconocido"

                    detected_shapes.append((shape, color_name, cnt))

        return detected_shapes

    # Función para verificar si un punto está dentro de una elipse
    def punto_en_elipse(x, y, centro, ejes):
        h, k = centro
        a, b = ejes
        return ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2) <= 1

    # Función para detectar colisiones entre figuras (solo para figuras virtuales)
    def colisionan(pos1, tamaño1, pos2, tamaño2):
        distancia = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        return distancia < (tamaño1 + tamaño2)

    # Función para dibujar una figura
    def dibujar_figura(screen, figura_info):
        x, y = figura_info['posición']
        tamaño = figura_info['tamaño']
        color = figura_info.get('color_bgr', (255, 255, 255))
        figura = figura_info['figura']

        if figura == 'círculo':
            cv2.circle(screen, (x, y), tamaño, color, -1)
        elif figura == 'cuadrado':
            cv2.rectangle(screen, (x - tamaño, y - tamaño), (x + tamaño, y + tamaño), color, -1)
        elif figura == 'triángulo':
            pts = np.array([
                [x, y - tamaño],
                [x - tamaño, y + tamaño],
                [x + tamaño, y + tamaño]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(screen, [pts], color)
        elif figura == 'estrella':
            # Dibujar una estrella simple de 5 puntas
            pts = []
            for i in range(5):
                angle = i * 4 * math.pi / 5 - math.pi / 2
                xi = x + tamaño * math.cos(angle)
                yi = y + tamaño * math.sin(angle)
                pts.append([int(xi), int(yi)])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(screen, [pts], color)
        else:
            # Si la figura no es reconocida, dibujar un círculo
            cv2.circle(screen, (x, y), tamaño, color, -1)

    # Inicializar 'figuras_a_dibujar' como diccionario
    figuras_a_dibujar = {}

    if not piezas_fisicas:
        # Generar figuras virtuales
        def generar_figuras():
            figuras_a_dibujar.clear()

            if modo_clasificacion == "figuras":
                # Seleccionar 2 tipos de figuras aleatorias
                tipos_figuras = random.sample(figuras_virtuales, 2)
                for _ in range(num_piezas):
                    figura_tipo = random.choice(tipos_figuras)
                    color_name = random.choice(list(colores_virtuales.keys()))
                    color_bgr = colores_virtuales[color_name]
                    tamaño = 30
                    # Generar posición aleatoria dentro del área de trabajo
                    while True:
                        x = random.randint(xv_min + tamaño, xv_max - tamaño)
                        y = random.randint(yv_min + tamaño, yv_max - tamaño)
                        if not any(colisionan((x, y), tamaño, f['posición'], f['tamaño']) for f in figuras_a_dibujar.values()):
                            key = (figura_tipo, color_name, x, y)
                            figuras_a_dibujar[key] = {
                                'figura': figura_tipo,
                                'color': color_name,
                                'color_bgr': color_bgr,
                                'tamaño': tamaño,
                                'posición': (x, y),
                                'moviendo': False,
                                'puntos_toque': [],
                                'clasificada': False,
                                'marcada': None
                            }
                            break
            elif modo_clasificacion == "colores":
                # Seleccionar 2 colores aleatorios
                colores_seleccionados = random.sample(list(colores_virtuales.keys()), 2)
                for _ in range(num_piezas):
                    color_name = random.choice(colores_seleccionados)
                    color_bgr = colores_virtuales[color_name]
                    figura_tipo = random.choice(figuras_virtuales)
                    tamaño = 30
                    # Generar posición aleatoria dentro del área de trabajo
                    while True:
                        x = random.randint(xv_min + tamaño, xv_max - tamaño)
                        y = random.randint(yv_min + tamaño, yv_max - tamaño)
                        if not any(colisionan((x, y), tamaño, f['posición'], f['tamaño']) for f in figuras_a_dibujar.values()):
                            key = (figura_tipo, color_name, x, y)
                            figuras_a_dibujar[key] = {
                                'figura': figura_tipo,
                                'color': color_name,
                                'color_bgr': color_bgr,
                                'tamaño': tamaño,
                                'posición': (x, y),
                                'moviendo': False,
                                'puntos_toque': [],
                                'clasificada': False,
                                'marcada': None
                            }
                            break

        generar_figuras()

    # Iniciar los streams de la cámara
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()

    # Reducir toques fantasma: persistencia y cooldown
    from collections import defaultdict
    touch_history_piezas = defaultdict(list)
    last_valid_touch_time_piezas = time.time()
    touch_cooldown_piezas = 0.2
    min_touch_frames_piezas = 2
    min_touch_area_piezas = 150
    max_touch_area_piezas = 50000
    max_history_age_piezas = 1.0

    while True:
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            continue

        current_time_piezas = time.time()

        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        # Crear la máscara (umbral más estricto 180 y MORPH_CLOSE para reducir fantasmas)
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Un centroide por contorno, filtro por área y persistencia
        raw_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_touch_area_piezas <= area <= max_touch_area_piezas:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                    raw_candidates.append((x_touch, y_touch, area))

        for key in list(touch_history_piezas.keys()):
            touch_history_piezas[key] = [t for t in touch_history_piezas[key] if current_time_piezas - t[3] < max_history_age_piezas]
            if not touch_history_piezas[key]:
                del touch_history_piezas[key]

        touch_points = []
        for (x_touch, y_touch, area) in raw_candidates:
            touch_key = (x_touch // 25, y_touch // 25)
            touch_history_piezas[touch_key].append((x_touch, y_touch, area, current_time_piezas))
            if len(touch_history_piezas[touch_key]) >= min_touch_frames_piezas and current_time_piezas - last_valid_touch_time_piezas > touch_cooldown_piezas:
                recent = touch_history_piezas[touch_key][-min_touch_frames_piezas:]
                if all(current_time_piezas - t[3] < 1.0 for t in recent):
                    areas = [t[2] for t in recent]
                    if min(areas) > 0 and max(areas) / min(areas) < 3.5 and min_touch_area_piezas <= sum(areas) / len(areas) <= max_touch_area_piezas:
                        touch_points.append([x_touch, y_touch])
                        if touch_key in touch_history_piezas:
                            del touch_history_piezas[touch_key]
                        last_valid_touch_time_piezas = current_time_piezas
                        break

        if not piezas_fisicas:
            # Mover las figuras virtuales asociadas a los toques
            for figura in figuras_a_dibujar.values():
                figura['moviendo'] = False
                figura['puntos_toque'] = []  # Reiniciar la lista de puntos de toque

            # Asociar puntos de toque con figuras
            for punto in touch_points:
                x_touch, y_touch = punto[0], punto[1]

                if xv_min <= x_touch < xv_max and yv_min <= y_touch < yv_max:
                    for figura in figuras_a_dibujar.values():
                        if figura['marcada'] is None:  # Solo mover si no está marcada incorrecta
                            # Verificar si el punto está dentro de la figura
                            distancia = math.hypot(x_touch - figura['posición'][0], y_touch - figura['posición'][1])
                            if distancia < figura['tamaño']:
                                figura['moviendo'] = True
                                figura['puntos_toque'].append((x_touch, y_touch))

            for figura in figuras_a_dibujar.values():
                if figura['moviendo'] and figura['puntos_toque']:
                    # Calcular la nueva posición como el promedio de los puntos de toque asociados
                    x_prom = int(sum(p[0] for p in figura['puntos_toque']) / len(figura['puntos_toque']))
                    y_prom = int(sum(p[1] for p in figura['puntos_toque']) / len(figura['puntos_toque']))
                    nueva_posición = (x_prom, y_prom)

                    # Verificar si la nueva posición está dentro del área válida
                    if 0 <= x_prom < view_width and 0 <= y_prom < view_height:
                        # Verificar si la nueva posición colisiona con otras figuras
                        colisiona_con_otra_figura = False
                        for otra_figura in figuras_a_dibujar.values():
                            if otra_figura is not figura:
                                if colisionan(nueva_posición, figura['tamaño'], otra_figura['posición'], otra_figura['tamaño']):
                                    colisiona_con_otra_figura = True
                                    break

                        if not colisiona_con_otra_figura:
                            figura['posición'] = nueva_posición
        else:
            # Procesar las figuras físicas
            detected_shapes = detect_color_and_shape(bgr_data)

            # Crear una copia de bgr_data para mostrar las detecciones
            deteccion_visual = bgr_data.copy()

            figuras_actualizadas = set()

            for shape, color_name, cnt in detected_shapes:
                # Obtener el centro y tamaño de la figura
                x, y, w, h = cv2.boundingRect(cnt)
                centro_x = x + w // 2
                centro_y = y + h // 2

                # Factores de escala
                sx = float(xv_max - xv_min) / (xw_max - xw_min)
                sy = float(yv_max - yv_min) / (yw_max - yw_min)

                # Mapeo de coordenadas de la ventana al viewport con ajuste
                x_viewport = int(xv_min + (centro_x * sx))
                y_viewport = int(yv_min + (centro_y * sy))

                figura_tipo = shape.lower()
                color_name_lower = color_name.lower()
                color_bgr = colores_bgr.get(color_name_lower, (255, 255, 255))

                key = (figura_tipo, color_name_lower)

                if key in figuras_a_dibujar:
                    figura = figuras_a_dibujar[key]
                    # Actualizar posición y tamaño
                    figura['posición'] = (x_viewport, y_viewport)
                    figura['tamaño'] = max(w, h) // 2
                else:
                    # Agregar nueva figura
                    figura = {
                        'figura': figura_tipo,
                        'color': color_name_lower,
                        'color_bgr': color_bgr,
                        'tamaño': max(w, h) // 2,
                        'posición': (x_viewport, y_viewport),
                        'marcada': None  # Inicializar como None
                    }
                    figuras_a_dibujar[key] = figura

                # Marcar que esta figura fue actualizada
                figuras_actualizadas.add(key)

                # Dibujar el contorno y etiqueta en deteccion_visual
                cv2.drawContours(deteccion_visual, [cnt], -1, (0, 255, 0), 2)
                put_text_ubuntu(deteccion_visual, f"{figura_tipo}, {color_name_lower}", (x, y - 10), 0.5, (0, 255, 0), 2)

            # Eliminar figuras que no fueron actualizadas
            keys_to_remove = set(figuras_a_dibujar.keys()) - figuras_actualizadas
            for key in keys_to_remove:
                del figuras_a_dibujar[key]

        # Dibujar las áreas y las figuras en la pantalla del videobeam
        videobeam_screen.fill(0)  # Limpiar la pantalla antes de redibujar

        if mostrar_ovalos:
            cv2.ellipse(videobeam_screen, area1_center, area1_axes, 0, 0, 360, (255, 255, 255), thickness=3)
            cv2.ellipse(videobeam_screen, area2_center, area2_axes, 0, 0, 360, (255, 255, 255), thickness=3)

        if not piezas_fisicas:
            # Dibujar las figuras virtuales
            for figura_info in figuras_a_dibujar.values():
                dibujar_figura(videobeam_screen, figura_info)

        # Dibujar las equis en las figuras marcadas (tanto virtuales como físicas)
        for figura_info in figuras_a_dibujar.values():
            if figura_info['marcada']:
                tiempo_transcurrido = time.time() - figura_info['marcada']
                if tiempo_transcurrido < DURACION_MARCA:
                    x, y = figura_info['posición']
                    tamaño = figura_info['tamaño']
                    factor = 1.5  # Ajusta este factor según sea necesario
                    delta = int(tamaño * factor)
                    # Dibujar la equis en el videobeam
                    cv2.line(videobeam_screen, (x - delta, y - delta), (x + delta, y + delta), (0, 0, 255), thickness=5)
                    cv2.line(videobeam_screen, (x - delta, y + delta), (x + delta, y - delta), (0, 0, 255), thickness=5)
                else:
                    figura_info['marcada'] = None  # Limpiar la marca después de que pase la duración

        key = cv2.waitKey(1) & 0xFF

        if key == ord('h'):
            mostrar_ovalos = not mostrar_ovalos  # Alternar la visualización de los óvalos
            if mostrar_ovalos:
                print("Óvalos visibles")
            else:
                print("Óvalos ocultos")

        if key == ord('v'):
            print("Verificando clasificación...")
            verificar_clasificacion = True

            # Listas para almacenar las figuras según su ubicación
            figuras_fuera = []
            figuras_en_area1 = []
            figuras_en_area2 = []

            for figura in figuras_a_dibujar.values():
                x, y = figura['posición']
                if punto_en_elipse(x, y, area1_center, area1_axes):
                    figuras_en_area1.append(figura)
                elif punto_en_elipse(x, y, area2_center, area2_axes):
                    figuras_en_area2.append(figura)
                else:
                    figuras_fuera.append(figura)

            if figuras_fuera:
                # Reproducir sonido de error por estar fuera de las áreas
                pygame.mixer.music.load('./sounds/incorrect.mp3')
                pygame.mixer.music.play()

                # Marcar las figuras fuera de los óvalos
                for figura in figuras_fuera:
                    figura['marcada'] = time.time()
            else:
                figuras_marcadas = []
                for figuras_en_area in [figuras_en_area1, figuras_en_area2]:
                    if figuras_en_area:
                        if modo_clasificacion == "figuras":
                            tipos = [f['figura'] for f in figuras_en_area]
                            tipo_mayor = max(set(tipos), key=tipos.count)
                            for figura in figuras_en_area:
                                if figura['figura'] != tipo_mayor:
                                    figura['marcada'] = time.time()
                                    figuras_marcadas.append(figura)
                        elif modo_clasificacion == "colores":
                            colores = [f['color'] for f in figuras_en_area]
                            color_mayor = max(set(colores), key=colores.count)
                            for figura in figuras_en_area:
                                if figura['color'] != color_mayor:
                                    figura['marcada'] = time.time()
                                    figuras_marcadas.append(figura)

                if figuras_marcadas:
                    # Reproducir sonido de error
                    pygame.mixer.music.load('./sounds/incorrect.mp3')
                    pygame.mixer.music.play()
                    # Las marcas se dibujarán en el bucle principal
                else:
                    # Reproducir sonido de victoria si todo está correcto
                    pygame.mixer.music.load('./sounds/victory.mp3')
                    pygame.mixer.music.play()
                    # Esperar unos segundos y salir
                    time.sleep(5)
                    break

        # Mostrar el resultado
        cv2.imshow("Mascara", touch_mask_final)
        cv2.namedWindow("Clasificación", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Clasificación", 1920, 0)
        cv2.setWindowProperty("Clasificación", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Clasificación", videobeam_screen)

        if piezas_fisicas:
            # Mostrar la detección en una ventana separada
            cv2.imshow("Detección", deteccion_visual)

        # Detectar la tecla 'q' para salir
        if key == ord('q'):
            break

    rgb_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()

def juego_handprint(device, offset=10):
    # Tamaño de la pantalla del videobeam
    view_width = 1280
    view_height = 800
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)  # Pantalla negra

    # Cargar las coordenadas de calibración desde el archivo JSON
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    # Cargar y validar dmax_map
    dmax_map, w, h = load_and_validate_dmax_map(coordenadas)
    if dmax_map is None:
        return

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Ajuste del rango de detección de profundidad
    dmax_map = dmax_map - 7
    dmin_map = dmax_map - 50

    # Crear una imagen para mantener las huellas de manos
    handprint_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    # Iniciar el stream de la cámara
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()


    # Cálculo de los factores de escala
    sx = float(xv_max - xv_min) / (xw_max - xw_min)
    sy = float(yv_max - yv_min) / (yw_max - yw_min)

    button_colors = [
        ("Rojo", (0, 0, 255)),
        ("Verde", (0, 255, 0)),
        ("Azul", (255, 0, 0)),
        ("Amarillo", (0, 255, 255))
    ]

    # Tamaño y posicion de los botones
    num_buttons = len(button_colors)
    button_width = 100  # Ancho del botón
    button_height = 60  # Alto del botón
    button_margin = xv_min + 50  # Margen desde el borde izquierdo del videobeam
    spacing = (view_height - (yv_min)) // (num_buttons + 1)  # Espaciado entre botones, ajustado a la altura visible

    draw_color = None

    buttons = []
    for i, (color_name, color_bgr) in enumerate(button_colors):
        y = yv_min + spacing * (i + 1)  # posicion vertical ajustada dentro del área de trabajo
        x =  button_margin  # posicion horizontal ajustada a los límites del área de trabajo
        buttons.append({
            "name": color_name,
            "color": color_bgr,
            "rect": (x, y, x + button_width, y + button_height)  # Coordenadas del botón (x1, y1, x2, y2)
        })

    # Función para mapear las coordenadas de la ventana (calibrada) al viewport (proyección)
    def window_to_viewport(x_touch, y_touch):
        x_viewport = int(xv_min + (x_touch) * sx)
        y_viewport = int(yv_min + (y_touch) * sy)
        return x_viewport, y_viewport

    # Función para dibujar los botones en la pantalla del videobeam
    def draw_buttons(videobeam_screen, buttons):
        for button in buttons:
            x1, y1, x2, y2 = button["rect"]
            color_bgr = button["color"]
            name = button["name"]
            cv2.rectangle(videobeam_screen, (x1, y1), (x2, y2), color_bgr, -1)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x1 + (button_width - text_size[0]) // 2
            text_y = y1 + (button_height + text_size[1]) // 2
            put_text_ubuntu(videobeam_screen, name, (text_x, text_y), 0.6, (255, 255, 255), 2)

    # Función para detectar toques en los botones
    def detect_color_touch(touch_mask, buttons):
        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Ajusta este umbral según tus necesidades
                M_contour = cv2.moments(contour)
                if M_contour["m00"] != 0:
                    cx = int(M_contour["m10"] / M_contour["m00"])
                    cy = int(M_contour["m01"] / M_contour["m00"])
                    # Mapeo de coordenadas de la ventana al viewport
                    tX, tY = window_to_viewport(cx, cy)
                    # Verificar si el punto está dentro de algún botón
                    for button in buttons:
                        x1, y1, x2, y2 = button["rect"]
                        if x1 <= tX <= x2 and y1 <= tY <= y2:
                            return button["color"]
        return None

    while True:
        # Leer el frame de la cámara RGB
        frame = rgb_stream.read_frame()
        if frame is None:
            continue

        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

        # Leer el frame de profundidad
        depth_frame = depth_stream.read_frame()
        if depth_frame is None:
            continue

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        # Crear la máscara de toques (umbral 180 y MORPH_CLOSE para reducir fantasmas)
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        touched_color = detect_color_touch(touch_mask_final, buttons)
        if touched_color is not None:
            draw_color = touched_color
       
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                for point in contour:
                    cx, cy = point[0]
                    # Mapeo de coordenadas de la ventana al viewport
                    tX, tY = window_to_viewport(cx, cy)
                    # Limitar el dibujo al área de trabajo (coordenadas calibradas)
                    if xv_min <= tX < xv_max and yv_min <= tY < yv_max:
                        # Asegurarse de que no se dibuje sobre los botones
                        is_on_button = False
                        for button in buttons:
                            x1, y1, x2, y2 = button["rect"]
                            if x1 <= tX <= x2 and y1 <= tY <= y2:
                                is_on_button = True
                                break
                        if not is_on_button and draw_color is not None:
                            cv2.circle(handprint_screen, (tX, tY), 3, draw_color, -1)


        # Dibujar las huellas y botones en la pantalla del videobeam
        videobeam_screen[:] = handprint_screen
        draw_buttons(videobeam_screen, buttons)

        # Mostrar la pantalla del videobeam
        cv2.namedWindow("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Pantalla de Videobeam", 1920, 0)
        cv2.setWindowProperty("Pantalla de Videobeam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Pantalla de Videobeam", videobeam_screen)

        # Si se presiona 'q', salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rgb_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()

def juego_personalizacion(device):
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    # Cargar y validar dmax_map
    dmax_map, w, h = load_and_validate_dmax_map(coordenadas)
    if dmax_map is None:
        return

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Ajustar el dmax_map (restar offset)
    dmax_map = dmax_map - 7
    dmin_map = dmax_map - 15

    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800

    # Inicializar pygame
    pygame.init()
    pygame.mixer.init()

    # Crear una pantalla negra para el videobeam
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    # Diccionario de imágenes de dibujos disponibles
    drawing_images = {
        "Dinosaurio": "./drawings/dinosaurio.png",
        "Mago": "./drawings/mago.png",
        "Robot": "./drawings/robot.png",
        "Sirena": "./drawings/sirena.png",
        "Superheroe": "./drawings/superheroe.png",
        "Superheroina": "./drawings/superheroina.png"
    }

    # Cargar todas las imágenes de dibujos
    loaded_drawing_images = {}
    for name, path in drawing_images.items():
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Redimensionar las imágenes a un tamaño uniforme para la selección
            resized_image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)  # Cambio 1
            loaded_drawing_images[name] = resized_image
        else:
            print(f"Error: No se pudo cargar el dibujo '{name}' desde {path}")

    # Estado del juego
    STATE_SELECTION = 0
    STATE_COLORING = 1
    game_state = STATE_SELECTION

    # Variables para la selección de dibujos
    drawing_positions = {}  # Mapeo de nombres de dibujos a sus posiciones en la pantalla
    selected_drawing = None  # Nombre del dibujo seleccionado

    work_area_width = xv_max - xv_min
    work_area_height = yv_max - yv_min

    # Calcular posiciones para las imágenes de selección (por ejemplo, en una cuadrícula 3x2)
    num_drawings = len(loaded_drawing_images)
    cols = 3
    rows = (num_drawings + cols - 1) // cols  # Redondear hacia arriba
    padding = 170  # Espacio reservado para los botones (barra de colores)
    padding_x = 20  # Reducir el padding para acomodar imágenes más pequeñas  # Cambio 2
    padding_y = 20  # Reducir el padding para acomodar imágenes más pequeñas  # Cambio 2
    image_size = 150  # Nuevo tamaño de las imágenes de selección  # Cambio 3
    spacing_x = (work_area_width - 2 * padding_x - cols * image_size) // (cols - 1) if cols > 1 else 0  # Cambio 3
    spacing_y = (work_area_height - 2 * padding_y - rows * image_size) // (rows - 1) if rows > 1 else 0  # Cambio 3

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= num_drawings:
                break
            name = list(loaded_drawing_images.keys())[idx]
            x = xv_min + padding_x + col * (image_size + spacing_x)  # Cambio 4
            y = yv_min + padding_y + row * (image_size + spacing_y)  # Cambio 4
            drawing_positions[name] = (x, y)
            idx += 1

    # Función para dibujar todas las opciones de dibujos en la pantalla
    def draw_drawing_options(screen, loaded_drawing_images, drawing_positions):
        for name, (x, y) in drawing_positions.items():
            image = loaded_drawing_images[name]
            h, w = image.shape[:2]
            # Insertar la imagen en la pantalla
            screen[y:y+h, x:x+w] = image[:, :, :3]  # Ignorar el canal alfa si existe
            # Dibujar un rectángulo alrededor de la imagen para indicar selección
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # Dibujar el nombre del dibujo
            put_text_ubuntu(screen, name, (x, y - 10), 0.6, (255, 255, 255), 2)

    # Seleccionar el primer dibujo por defecto
    selected_drawing = list(drawing_images.keys())[0]
    dibujo = cv2.imread(drawing_images[selected_drawing], cv2.IMREAD_UNCHANGED)
    if dibujo is None:
        print(f"Error: No se pudo cargar el dibujo '{selected_drawing}'")
        return

    # Redimensionar el dibujo al tamaño del área de dibujo
    print(f"xv_min: {xv_min}, xv_max: {xv_max}, padding: {padding}, xv_max - padding: {xv_max - padding}")

    if (xv_max - padding) <= xv_min:
        print("Error: Padding demasiado grande, el área de dibujo es inexistente.")
        return

    area_width = xv_max - xv_min - padding
    area_height = yv_max - yv_min
    dibujo = cv2.resize(dibujo, (area_width, area_height), interpolation=cv2.INTER_LINEAR)

    # Separar los canales y crear máscaras
    if dibujo.shape[2] == 4:
        b_channel, g_channel, r_channel, alpha_channel = cv2.split(dibujo)
    else:
        # Si no hay canal alfa, crear uno completamente opaco
        b_channel, g_channel, r_channel = cv2.split(dibujo)
        alpha_channel = np.ones((dibujo.shape[0], dibujo.shape[1]), dtype=np.uint8) * 255

    dibujo_rgb = cv2.merge((b_channel, g_channel, r_channel))

    # Máscara para áreas coloreables (donde alpha > 0 y no son líneas negras)
    mask_coloreable = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
    # Cambio 1: Usar Canny para detección de bordes
    gray = cv2.cvtColor(dibujo_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 180)
    mask_lineas = edges  # Las líneas detectadas por Canny
    mask_coloreable = cv2.bitwise_and(mask_coloreable, cv2.bitwise_not(mask_lineas))  # Cambio 3: Asegurar exclusión

    # Verificar la máscara coloreable
    print(f"mask_coloreable shape: {mask_coloreable.shape}, max: {mask_coloreable.max()}, min: {mask_coloreable.min()}")

    # Crear el área de dibujo inicial
    area_dibujo = np.zeros((area_height, area_width, 3), dtype=np.uint8)
    area_dibujo = cv2.bitwise_or(area_dibujo, dibujo_rgb, mask=alpha_channel)

    # Función para superponer las líneas negras sobre el área de dibujo
    def overlay_lines(area_dibujo, dibujo_rgb, mask_lineas):
        area_dibujo_con_lineas = area_dibujo.copy()
        area_dibujo_con_lineas[mask_lineas == 255] = dibujo_rgb[mask_lineas == 255]
        return area_dibujo_con_lineas

    # Inicializar el área de dibujo con líneas negras
    area_dibujo_con_lineas = overlay_lines(area_dibujo, dibujo_rgb, mask_lineas)

    # Iniciar los streams de la cámara
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()

    # Colores disponibles para colorear, incluyendo el morado
    colores = [
        (255, 0, 0),     # Azul
        (0, 255, 0),     # Verde
        (0, 0, 255),     # Rojo
        (0, 255, 255),   # Amarillo
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Cian
        (128, 0, 128),   # Morado
        (152, 194, 229),  # Skin color (light beige)
        (0, 0, 0)        # Negro
    ]
    color_actual = colores[0]
    indice_color = 0

    # Definir Window to Viewport Mapping
    sx = float(xv_max - xv_min) / (xw_max - xw_min)
    sy = float(yv_max - yv_min) / (yw_max - yw_min)

    def window_to_viewport(x_touch, y_touch):
        x_viewport = int(xv_min + (x_touch) * sx)
        y_viewport = int(yv_min + (y_touch) * sy)
        return x_viewport, y_viewport

    # Posiciones de los colores en la barra lateral
    color_buttons = []
    button_size = 30
    button_padding = 10
    barra_ancho = padding  # Ancho de la barra lateral

    # Calcular el espacio disponible en la barra para los botones
    max_buttons = len(colores)
    total_buttons_height = max_buttons * button_size + (max_buttons - 1) * button_padding
    start_y = yv_min + (area_height - total_buttons_height) // 2  # Centrar los botones verticalmente

    for i, color in enumerate(colores):
        x_button = xv_min + (barra_ancho - button_size) // 2  # Mover la barra al lado izquierdo
        y_button = start_y + i * (button_size + button_padding)
        color_buttons.append(((x_button, y_button), color))

    # Verificar las posiciones de los botones
    for i, ((x_button, y_button), color) in enumerate(color_buttons):
        print(f"Color button {i}: position=({x_button}, {y_button}), color={color}")

    # Función para detectar si un punto está dentro de un rectángulo
    def is_point_in_rect(x, y, rect_x, rect_y, rect_w, rect_h):
        return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

    # Función para manejar la selección de dibujo
    def handle_drawing_selection(touch_points, videobeam_screen):
        nonlocal selected_drawing, dibujo, mask_coloreable, mask_lineas, area_dibujo, area_width, area_height, game_state, area_dibujo_con_lineas
        for point in touch_points:
            x_touch_win, y_touch_win = point

            # Aplicar window to viewport mapping
            x_viewport, y_viewport = window_to_viewport(x_touch_win, y_touch_win)
            print(f"Mapped touch view coordinates: ({x_viewport}, {y_viewport})")

            # Verificar si el toque está dentro de alguna de las imágenes de selección
            for name, (x, y) in drawing_positions.items():
                if is_point_in_rect(x_viewport, y_viewport, x, y, 150, 150):
                    selected_drawing = name
                    print(f"Dibujo seleccionado: {selected_drawing}")
                    
                    # Cargar el dibujo seleccionado para colorear
                    selected_image_path = drawing_images[selected_drawing]
                    dibujo = cv2.imread(selected_image_path, cv2.IMREAD_UNCHANGED)
                    if dibujo is None:
                        print(f"Error: No se pudo cargar el dibujo '{selected_drawing}' desde {selected_image_path}")
                        continue

                    # Redimensionar el dibujo al tamaño del área de dibujo
                    dibujo = cv2.resize(dibujo, (area_width, area_height), interpolation=cv2.INTER_LINEAR)

                    # Separar los canales y crear máscaras
                    if dibujo.shape[2] == 4:
                        b_channel, g_channel, r_channel, alpha_channel = cv2.split(dibujo)
                    else:
                        # Si no hay canal alfa, crear uno completamente opaco
                        b_channel, g_channel, r_channel = cv2.split(dibujo)
                        alpha_channel = np.ones((dibujo.shape[0], dibujo.shape[1]), dtype=np.uint8) * 255

                    dibujo_rgb = cv2.merge((b_channel, g_channel, r_channel))

                    # Máscara para áreas coloreables (donde alpha > 0 y no son líneas negras)
                    mask_coloreable = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
                    # Cambio 1: Usar Canny para detección de bordes
                    gray = cv2.cvtColor(dibujo_rgb, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 20, 180)
                    mask_lineas = edges  # Las líneas detectadas por Canny
                    mask_coloreable = cv2.bitwise_and(mask_coloreable, cv2.bitwise_not(mask_lineas))  # Cambio 3: Asegurar exclusión

                    # Verificar la máscara coloreable
                    print(f"mask_coloreable shape: {mask_coloreable.shape}, max: {mask_coloreable.max()}, min: {mask_coloreable.min()}")

                    # Crear el área de dibujo inicial
                    area_dibujo = np.zeros((area_height, area_width, 3), dtype=np.uint8)
                    area_dibujo = cv2.bitwise_or(area_dibujo, dibujo_rgb, mask=alpha_channel)

                    # Restaurar los bordes negros usando overlay_lines
                    area_dibujo_con_lineas = overlay_lines(area_dibujo, dibujo_rgb, mask_lineas)

                    # Cambiar el estado del juego a coloreo
                    game_state = STATE_COLORING

                    # Limpiar la parte de selección en videobeam_screen
                    videobeam_screen[yv_min:yv_max, xv_min:xv_max] = 0

                    # Actualizar la pantalla con el área de dibujo con líneas
                    update_videobeam_screen(videobeam_screen, area_dibujo_con_lineas, xv_min, yv_min, xv_max, yv_max, padding)

                    return True  # Salir después de seleccionar un dibujo
        return False

    # Función para dibujar los botones de colores
    def draw_color_buttons(screen, color_buttons, selected_color):
        for (x_button, y_button), color in color_buttons:
            cv2.rectangle(screen, (x_button, y_button), (x_button + button_size, y_button + button_size), color, -1)
            # Dibujar un borde para resaltar el botón seleccionado
            if color == selected_color:
                cv2.rectangle(screen, (x_button, y_button), (x_button + button_size, y_button + button_size), (255, 255, 255), 2)

    # Función para superponer las líneas negras sobre el área de dibujo
    def overlay_lines(area_dibujo, dibujo_rgb, mask_lineas):
        area_dibujo_con_lineas = area_dibujo.copy()
        area_dibujo_con_lineas[mask_lineas == 255] = dibujo_rgb[mask_lineas == 255]
        return area_dibujo_con_lineas

    # Función para actualizar el área de dibujo en la pantalla principal
    def update_videobeam_screen(screen, area_dibujo_con_lineas, xv_min, yv_min, xv_max, yv_max, padding):
        screen[yv_min:yv_max, xv_min + padding:xv_max] = area_dibujo_con_lineas

    # Función para mostrar la selección de dibujos
    def show_drawing_selection(screen, loaded_drawing_images, drawing_positions):
        # Limpiar solo el área de trabajo
        screen[yv_min:yv_max, xv_min:xv_max] = 0
        draw_drawing_options(screen, loaded_drawing_images, drawing_positions)
        # Configurar la ventana "Dibujo" en pantalla completa sin decoraciones
        cv2.namedWindow("Dibujo", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Dibujo", 1920, 0)
        cv2.setWindowProperty("Dibujo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Dibujo", screen)

    # Reducir toques fantasma en dibujo: persistencia y cooldown
    from collections import defaultdict
    touch_history_draw = defaultdict(list)
    last_valid_touch_time_draw = time.time()
    touch_cooldown_draw = 0.2
    min_touch_frames_draw = 2
    min_touch_area_draw = 150
    max_touch_area_draw = 50000
    max_history_age_draw = 1.0

    while True:
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            continue

        current_time_draw = time.time()

        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        # Máscara más estricta (umbral 180, MORPH_CLOSE) para reducir fantasmas
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates_draw = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_touch_area_draw <= area <= max_touch_area_draw:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    raw_candidates_draw.append((cx, cy, area))

        for key in list(touch_history_draw.keys()):
            touch_history_draw[key] = [t for t in touch_history_draw[key] if current_time_draw - t[3] < max_history_age_draw]
            if not touch_history_draw[key]:
                del touch_history_draw[key]

        touch_points = []
        for (cx, cy, area) in raw_candidates_draw:
            touch_key = (cx // 20, cy // 20)
            touch_history_draw[touch_key].append((cx, cy, area, current_time_draw))
            if len(touch_history_draw[touch_key]) >= min_touch_frames_draw and current_time_draw - last_valid_touch_time_draw > touch_cooldown_draw:
                recent = touch_history_draw[touch_key][-min_touch_frames_draw:]
                if all(current_time_draw - t[3] < 1.0 for t in recent):
                    areas = [t[2] for t in recent]
                    if min(areas) > 0 and max(areas) / min(areas) < 3.5 and min_touch_area_draw <= sum(areas) / len(areas) <= max_touch_area_draw:
                        touch_points.append((cx, cy))
                        if touch_key in touch_history_draw:
                            del touch_history_draw[touch_key]
                        last_valid_touch_time_draw = current_time_draw
                        break

        # Depuración: mostrar touch_points
        # print(f"Touch points (Window coordinates): {touch_points}")

        # Detectar teclas presionadas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Solicitar el nombre del niño y el grupo
            child_name = input("Ingrese el nombre del niño: ").strip()
            group = input("Ingrese el grupo al que pertenece: ").strip()

            # Validar entradas
            if not child_name or not group:
                print("Error: El nombre del niño y el grupo no pueden estar vacíos.")
                continue

            # Crear la carpeta 'avatars' si no existe
            os.makedirs("avatars", exist_ok=True)

            # Construir el nombre del archivo
            sanitized_child_name = "".join(c for c in child_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            sanitized_group = "".join(c for c in group if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            filename = f"avatars/{sanitized_child_name}_{sanitized_group}.png"

            # Guardar la imagen en la carpeta 'avatars', sobrescribiendo si existe
            # Extraer la región de dibujo (excluir la barra de colores)
            drawing_area = videobeam_screen[yv_min: yv_max, xv_min + padding: xv_max]

            # Guardar la imagen de dibujo con líneas negras
            success = cv2.imwrite(filename, drawing_area)
            if success:
                # Mostrar una confirmación en la pantalla
                put_text_ubuntu(videobeam_screen, f"Dibujo guardado como {sanitized_child_name}_{sanitized_group}.png", 
                            (50, 50), 1, (0, 255, 0), 2)
                print(f"Dibujo guardado exitosamente en {filename}")
            else:
                print(f"Error al guardar el dibujo en {filename}")
            break  # Salir del bucle después de guardar

        # Manejar el estado del juego
        if game_state == STATE_SELECTION:
            # Mostrar todas las opciones de dibujos en el videobeam_screen
            show_drawing_selection(videobeam_screen, loaded_drawing_images, drawing_positions)

            # Manejar la selección de dibujo
            if handle_drawing_selection(touch_points, videobeam_screen):
                print(f"Cambiando al estado de coloreo con el dibujo: {selected_drawing}")
        elif game_state == STATE_COLORING:
            # Detectar si se toca algún color o el área de dibujo
            for point in touch_points:
                x_touch_win, y_touch_win = point

                # Aplicar window to viewport mapping
                x_viewport, y_viewport = window_to_viewport(x_touch_win, y_touch_win)
                print(f"Mapped touch view coordinates: ({x_viewport}, {y_viewport})")

                # Verificar si el toque está dentro de la barra de colores
                if (x_viewport <= xv_min + padding) and (yv_min <= y_viewport < yv_max):
                    print(f"Touch on color bar at: ({x_viewport}, {y_viewport})")
                    for i, ((x_button, y_button), color) in enumerate(color_buttons):
                        if is_point_in_rect(x_viewport, y_viewport, x_button, y_button, button_size, button_size):
                            color_actual = color
                            indice_color = i
                            print(f"Cambiado a color: {color_actual}")
                            break

                # Corrección en la detección de toques dentro del área de dibujo
                elif ((xv_min + padding) <= x_viewport < xv_max) and (yv_min <= y_viewport < yv_max):
                    x_dibujo = x_viewport - (xv_min + padding)  # Ajuste para tener en cuenta el padding a la izquierda
                    y_dibujo = y_viewport - yv_min
                    color_en_punto = dibujo_rgb[y_dibujo, x_dibujo]
                    print(f"Color en el punto de toque ({x_dibujo}, {y_dibujo}): {color_en_punto}")
                    print(f"Mapped to dibujo: ({x_dibujo}, {y_dibujo})")
                    if 0 <= x_dibujo < area_width and 0 <= y_dibujo < area_height:
                        print(f"mask_coloreable[y_dibujo, x_dibujo]: {mask_coloreable[y_dibujo, x_dibujo]}")
                        if mask_coloreable[y_dibujo, x_dibujo] == 255 and mask_lineas[y_dibujo, x_dibujo] == 0:
                            mask = np.zeros((area_height + 2, area_width + 2), np.uint8)
                            flags = 4 | (255 << 8)
                            # Aplicar floodFill
                            cv2.floodFill(area_dibujo, mask, (x_dibujo, y_dibujo), color_actual, flags=flags)
                            
                            # Restaurar las líneas negras después del floodFill
                            # area_dibujo_con_lineas = overlay_lines(area_dibujo, dibujo_rgb, mask_lineas)

            # Dibujar los botones de colores en la barra lateral
            draw_color_buttons(videobeam_screen, color_buttons, color_actual)

            # Superponer las líneas negras sobre el área de dibujo
            area_dibujo_con_lineas = overlay_lines(area_dibujo, dibujo_rgb, mask_lineas)

            # Actualizar la pantalla principal con el área de dibujo actualizada
            update_videobeam_screen(videobeam_screen, area_dibujo_con_lineas, xv_min, yv_min, xv_max, yv_max, padding)

            # Mostrar el resultado en la ventana "Dibujo"
            cv2.imshow("Dibujo", videobeam_screen)

    rgb_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

def simon_dice(device):
    # Cargar la configuración de coordenadas
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    # Cargar y validar dmax_map
    dmax_map, w, h = load_and_validate_dmax_map(coordenadas)
    if dmax_map is None:
        return

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Ajustar el dmax_map (restar offset)
    dmax_map = dmax_map - 5
    dmin_map = dmax_map - 7

    # Preguntar por el número de grupo
    group_number = input("Por favor ingrese el número de grupo: ").strip()

    # Listar los archivos en la carpeta 'avatars'
    avatar_files = [f for f in os.listdir("avatars") if f.endswith(".png") and f.split('_')[-1].replace('.png', '') == group_number]

    if not avatar_files:
        print(f"No se encontraron avatares para el grupo {group_number}.")
        return

    # Mostrar los nombres de los niños disponibles
    print("Niños disponibles en este grupo:")
    for idx, file in enumerate(avatar_files):
        name = file.split('_')[0]
        print(f"{idx + 1}. {name}")

    # Pedir la selección de los dos jugadores
    selected_players = []
    while len(selected_players) < 2:
        try:
            selection = int(input(f"Seleccione al jugador {len(selected_players) + 1} (1-{len(avatar_files)}): ")) - 1
            if 0 <= selection < len(avatar_files):
                selected_players.append(avatar_files[selection])
            else:
                print("Selección inválida.")
        except ValueError:
            print("Por favor, ingrese un número válido.")

    # Cargar los avatares seleccionados
    avatars = []
    for player_file in selected_players:
        avatar = cv2.imread(os.path.join("avatars", player_file))
        if avatar is not None:
            avatars.append(avatar)
        else:
            print(f"Error al cargar el avatar {player_file}")

    if not avatars:
        print("No se pudieron cargar los avatares.")
        return

    # Dimensiones del área de trabajo
    work_area_width = xv_max - xv_min
    work_area_height = yv_max - yv_min

    # Crear la pantalla del videobeam
    view_width = 1280
    view_height = 800
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

    # Redimensionar los avatares para que encajen en el área de trabajo
    avatar_height = work_area_height // 4  # Ajustamos el tamaño a un cuarto del área de trabajo
    avatar_width = avatar_height  # Mantener relación de aspecto cuadrada

    resized_avatars = []
    for avatar in avatars:
        resized_avatar = cv2.resize(avatar, (avatar_width, avatar_height), interpolation=cv2.INTER_LINEAR)
        resized_avatars.append(resized_avatar)

    # Rotar ambos avatares 90 grados en sentidos opuestos
    resized_avatars[0] = cv2.rotate(resized_avatars[0], cv2.ROTATE_90_CLOCKWISE)
    resized_avatars[1] = cv2.rotate(resized_avatars[1], cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Posicionar los avatares uno debajo del otro en el centro del área de trabajo
    avatar_x = xv_min + (work_area_width - avatar_width) // 2  # Centrado horizontalmente
    avatar_y1 = yv_min + (work_area_height - avatar_height) // 2 - avatar_height - 10  # Centrado verticalmente, parte superior
    avatar_y2 = yv_min + (work_area_height - avatar_height) // 2 + 10  # Centrado verticalmente, parte inferior

    low_brightness_factor = 0.5
    avatar_brightness_low = [cv2.convertScaleAbs(avatar, alpha=low_brightness_factor, beta=0) for avatar in resized_avatars]

    # Colocar los avatares en la pantalla
    videobeam_screen[avatar_y1:avatar_y1 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar_brightness_low[0]
    videobeam_screen[avatar_y2:avatar_y2 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar_brightness_low[1]

    # Posicionar los botones (figuras) en las esquinas
    button_size = 85  # Tamaño de las figuras
    button_padding = 40  # Espacio entre los botones
    buttons_left = []
    buttons_right = []

    sx = float(xv_max - xv_min) / (xw_max - xw_min)
    sy = float(yv_max - yv_min) / (yw_max - yw_min)

    def window_to_viewport(x_touch, y_touch):
        x_viewport = int(xv_min + (x_touch) * sx)
        y_viewport = int(yv_min + (y_touch) * sy)
        return x_viewport, y_viewport

    def rotate_point(point, angle_deg):
        angle_rad = math.radians(angle_deg)
        x, y = point
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        return (x_rot, y_rot)

    def draw_heart(screen, center, size, color, rotation=0):
        points = []
        num_points = 100
        for i in range(num_points):
            # Parámetro t que varía de 0 a 2*pi
            t = math.pi - (i * (2 * math.pi) / num_points)
            
            # Ecuaciones paramétricas ajustadas para tamaños pequeños
            x = math.sin(t) ** 3
            y = math.cos(t) - 0.4 * math.cos(2 * t) - 0.2 * math.cos(3 * t)
            
            # Escalar los puntos según el tamaño deseado
            x_scaled = size * x
            y_scaled = size * y

            x_rot, y_rot = rotate_point((x_scaled, y_scaled), rotation)
            
            # Trasladar los puntos al centro especificado
            x_final = int(center[0] + x_rot)
            y_final = int(center[1] - y_rot)  # Invertir el eje Y para que el corazón apunte hacia arriba tras la rotación
            
            points.append([x_final, y_final])
        
        # Convertir la lista de puntos a un arreglo de NumPy de tipo entero 32
        points = np.array(points, dtype=np.int32)
        
        # Dibujar y rellenar el polígono del corazón
        cv2.fillPoly(screen, [points], color)

    def draw_star(screen, center, size, color, rotation=0):
        base_points = [
            (0, -size),
            (size / 2, size / 2),
            (-size, -size / 4),
            (size, -size / 4),
            (-size / 2, size / 2),
        ]
        rotated_points = [rotate_point(p, rotation) for p in base_points]
        translated_points = [(center[0] + p[0], center[1] - p[1]) for p in rotated_points]
        points = np.array(translated_points, dtype=np.int32)
        cv2.fillPoly(screen, [points], color)

    def draw_diamond(screen, center, size, color, rotation=0):
        base_points = [
            (0, -size),
            (-size, 0),
            (0, size),
            (size, 0),
        ]
        rotated_points = [rotate_point(p, rotation) for p in base_points]
        translated_points = [(center[0] + p[0], center[1] - p[1]) for p in rotated_points]
        points = np.array(translated_points, dtype=np.int32)
        cv2.fillPoly(screen, [points], color)

    def draw_circle(screen, center, size, color, rotation=0):
        cv2.circle(screen, center, size, color, -1)

    # Posiciones de los botones (figuras) para cada lado
    avatar_center_y = yv_min + (work_area_height // 2) - (avatar_height // 2)

    # Alinear las posiciones de los botones con la altura del avatar
    left_button_positions = [
        (xv_min + button_padding, avatar_center_y - button_size - button_padding),
        (xv_min + button_padding, avatar_center_y + button_padding),
        (xv_min + button_padding + button_size + button_padding, avatar_center_y - button_size - button_padding),
        (xv_min + button_padding + button_size + button_padding, avatar_center_y + button_padding),
    ]

    right_button_positions = [
        (xv_max - button_padding - button_size, avatar_center_y - button_size - button_padding),
        (xv_max - button_padding - button_size, avatar_center_y + button_padding),
        (xv_max - button_padding - 2*button_size - button_padding, avatar_center_y - button_size - button_padding),
        (xv_max - button_padding - 2*button_size - button_padding, avatar_center_y + button_padding),
    ]

    # Dibujar las figuras iniciales
    button_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    shapes = [draw_heart, draw_star, draw_diamond, draw_circle]  # Tipos de formas

    low_brightness_factor = 0.6

    def apply_low_brightness(color):
        return tuple([int(c * low_brightness_factor) for c in color])

    for i, pos in enumerate(left_button_positions):
        low_brightness_color = apply_low_brightness(button_colors[i])
        shapes[i](videobeam_screen, pos, button_size // 2, low_brightness_color, rotation=-90)
        buttons_left.append((pos, shapes[i]))

    for i, pos in enumerate(right_button_positions):
        low_brightness_color = apply_low_brightness(button_colors[i])
        shapes[i](videobeam_screen, pos, button_size // 2, low_brightness_color, rotation=90)
        buttons_right.append((pos, shapes[i]))

    # Mostrar la pantalla en el videobeam
    cv2.imshow("Videobeam", videobeam_screen)
    base_screen = videobeam_screen.copy() 

    # Secuencia del juego
    sequence = []
    current_input = []
    current_player = random.choice([0, 1])  # Jugador inicial aleatorio

    # Aumentar el brillo del avatar del jugador actual
    def highlight_avatar(player_index):
        if player_index == 0:
            avatar = resized_avatars[0]
            videobeam_screen[avatar_y1:avatar_y1 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar
        else:
            avatar = resized_avatars[1]
            videobeam_screen[avatar_y2:avatar_y2 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar
        cv2.imshow("Videobeam", videobeam_screen)
        cv2.waitKey(500)
        base_screen = videobeam_screen.copy() 

    def dim_avatar(player_index):
        if player_index == 0:
            avatar = avatar_brightness_low[0]
            videobeam_screen[avatar_y1:avatar_y1 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar
        else:
            avatar = avatar_brightness_low[1]
            videobeam_screen[avatar_y2:avatar_y2 + avatar_height, avatar_x:avatar_x + avatar_width] = avatar
        cv2.imshow("Videobeam", videobeam_screen)
        cv2.waitKey(500)
        base_screen = videobeam_screen.copy() 

    # Mostrar la secuencia actual en el tablero (ilumina la secuencia)
    def show_sequence():
        for index in sequence:
            if index < 4:
                pos, shape = buttons_left[index]
            else:
                pos, shape = buttons_right[index - 4]
            shape(videobeam_screen, pos, button_size // 2, (255, 255, 255))  # Iluminar
            cv2.imshow("Videobeam", videobeam_screen)
            cv2.waitKey(500)
            shape(videobeam_screen, pos, button_size // 2, button_colors[index % 4])  # Restaurar color
            cv2.waitKey(500)

    highlight_avatar(current_player)

    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()

    turn_counter = 0  # Para contar el número de turnos y validar el número de botones
    buttons_touched = []
    button_pressed = False

    pygame.mixer.init()

    def reproducir_sonido(i):
        # Asignar diferentes sonidos a los botones
        sonidos = ['sounds/simon/F.mp3', 'sounds/simon/E.mp3', 'sounds/simon/D.mp3', 'sounds/simon/C.mp3', 'sounds/incorrect.mp3']
        if 0 <= i < len(sonidos):
            sound = pygame.mixer.Sound(sonidos[i])
            threading.Thread(target=sound.play).start()

    def mostrar_error():
        if current_player == 0:  # Jugador superior (izquierda)
            x_position = avatar_x + avatar_width // 2  # Centrar la X en el avatar superior
            y_position = avatar_y1 + avatar_height // 2
        else:  # Jugador inferior (derecha)
            x_position = avatar_x + avatar_width // 2  # Centrar la X en el avatar inferior
            y_position = avatar_y2 + avatar_height // 2
        
        # Dibujar la "X" sobre el avatar
        put_text_ubuntu(videobeam_screen, "X", (x_position, y_position), 5, (0, 0, 255), 10)
        cv2.imshow("Videobeam", videobeam_screen)
        cv2.waitKey(2000)  # Mostrar la X por 2 segundos

    # Reiniciar el juego después de un error
    def reiniciar_juego(current_player):
        nonlocal current_input, sequence, turn_counter, button_pressed, buttons_touched, videobeam_screen, base_screen
        print("Reiniciando el juego...")

        # Limpiar todos los estados y secuencias
        current_input = []
        sequence = []
        turn_counter = 0
        buttons_touched = []
        button_pressed = False
        videobeam_screen = base_screen.copy()

        # Cambiar de jugador
        dim_avatar(current_player)
        current_player = random.choice([0, 1])
        highlight_avatar(current_player)

        # Restaurar la pantalla base

        cv2.imshow("Videobeam", videobeam_screen)
        cv2.waitKey(500)

        return current_player, sequence

    # Reducir toques fantasma en Simon: persistencia y cooldown
    from collections import defaultdict
    touch_history_simon = defaultdict(list)
    last_valid_touch_time_simon = time.time()
    touch_cooldown_simon = 0.2
    min_touch_frames_simon = 2
    min_touch_area_simon = 150
    max_touch_area_simon = 50000
    max_history_age_simon = 1.0

    while True:
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            continue

        current_time_simon = time.time()

        # Extraer los datos de las imágenes
        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        # Máscara más estricta (umbral 180, MORPH_CLOSE) para reducir fantasmas
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates_simon = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_touch_area_simon <= area <= max_touch_area_simon:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    raw_candidates_simon.append((cx, cy, area))

        for key in list(touch_history_simon.keys()):
            touch_history_simon[key] = [t for t in touch_history_simon[key] if current_time_simon - t[3] < max_history_age_simon]
            if not touch_history_simon[key]:
                del touch_history_simon[key]

        touch_points = []
        for (cx, cy, area) in raw_candidates_simon:
            touch_key = (cx // 20, cy // 20)
            touch_history_simon[touch_key].append((cx, cy, area, current_time_simon))
            if len(touch_history_simon[touch_key]) >= min_touch_frames_simon and current_time_simon - last_valid_touch_time_simon > touch_cooldown_simon:
                recent = touch_history_simon[touch_key][-min_touch_frames_simon:]
                if all(current_time_simon - t[3] < 1.0 for t in recent):
                    areas = [t[2] for t in recent]
                    if min(areas) > 0 and max(areas) / min(areas) < 3.5 and min_touch_area_simon <= sum(areas) / len(areas) <= max_touch_area_simon:
                        touch_points.append((cx, cy))
                        if touch_key in touch_history_simon:
                            del touch_history_simon[touch_key]
                        last_valid_touch_time_simon = current_time_simon
                        break

        if len(touch_points) == 0 and button_pressed:
            button_pressed = False
            buttons_touched.clear()

        # Procesar los toques solo si no hay un botón presionado
        if not button_pressed:
            for point in touch_points:
                x_touch_win, y_touch_win = point
                x_viewport, y_viewport = window_to_viewport(x_touch_win, y_touch_win)
                # cv2.circle(videobeam_screen, (x_viewport, y_viewport), 10, (0, 255, 0), -1)

                # Evitar toques múltiples
                if not button_pressed:  # Solo permitir un toque cuando no hay un botón presionado
                    if current_player == 0:  # Jugador superior, puede tocar botones de la izquierda
                        for i, (pos, shape) in enumerate(buttons_left):
                            if (pos[0] - button_size//2 <= x_viewport <= pos[0] + button_size//2 and
                                pos[1] - button_size//2 <= y_viewport <= pos[1] + button_size//2):
                                if i not in buttons_touched:
                                    reproducir_sonido(i)
                                    shape(videobeam_screen, pos, button_size // 2, button_colors[i], rotation = -90)  # Iluminar con color completo
                                    corresponding_right_pos, corresponding_right_shape = buttons_right[i]  # Encuentra el botón correspondiente
                                    corresponding_right_shape(videobeam_screen, corresponding_right_pos, button_size // 2, button_colors[i], rotation = 90)  # Iluminar en el lado derecho
                                    cv2.imshow("Videobeam", videobeam_screen)
                                    cv2.waitKey(200)

                                    # Restaurar el brillo bajo
                                    low_brightness_color = apply_low_brightness(button_colors[i])
                                    shape(videobeam_screen, pos, button_size // 2, low_brightness_color, rotation = -90)  # Restaurar brillo bajo
                                    corresponding_right_shape(videobeam_screen, corresponding_right_pos, button_size // 2, low_brightness_color, rotation = 90)
                                    
                                    # Normalizar los valores de los botones para que estén en el rango de 0 a 3
                                    pressed_button = i  # Los botones de la izquierda ya están en el rango de 0 a 3
                                    
                                    # Comparar la secuencia existente
                                    if len(current_input) < len(sequence):
                                        if sequence[len(current_input)] == pressed_button:
                                            print(f"Jugador superior tocó correctamente el botón: {pressed_button}")
                                            current_input.append(pressed_button)
                                        else:
                                            print("¡Error en la secuencia!")
                                            reproducir_sonido(4)
                                            mostrar_error()  # Mostrar la X
                                            current_player, sequence = reiniciar_juego(current_player)  # Reiniciar el juego
                                            print(f"Limpiada la secuencia: {sequence}")
                                            break

                                    elif len(current_input) == len(sequence):  # Agregar un nuevo botón
                                        print(f"Jugador superior agregó un nuevo botón: {pressed_button}")
                                        current_input.append(pressed_button)

                                    buttons_touched.append(i)
                                    button_pressed = True
                                    break

                    elif current_player == 1:  # Jugador inferior, puede tocar botones de la derecha
                        for i, (pos, shape) in enumerate(buttons_right):
                            if (pos[0] - button_size//2 <= x_viewport <= pos[0] + button_size//2 and
                                pos[1] - button_size//2 <= y_viewport <= pos[1] + button_size//2):
                                if i not in buttons_touched:
                                    reproducir_sonido(i)
                                    shape(videobeam_screen, pos, button_size // 2, button_colors[i], rotation = 90)  # Iluminar con color completo
                                    corresponding_left_pos, corresponding_left_shape = buttons_left[i]  # Encuentra el botón correspondiente
                                    corresponding_left_shape(videobeam_screen, corresponding_left_pos, button_size // 2, button_colors[i], rotation = -90)  # Iluminar en el lado izquierdo
                                    cv2.imshow("Videobeam", videobeam_screen)
                                    cv2.waitKey(200)

                                    # Restaurar el brillo bajo
                                    low_brightness_color = apply_low_brightness(button_colors[i])
                                    shape(videobeam_screen, pos, button_size // 2, low_brightness_color, rotation = 90)  # Restaurar brillo bajo
                                    corresponding_left_shape(videobeam_screen, corresponding_left_pos, button_size // 2, low_brightness_color, rotation = -90)
                                    
                                    # Normalizar los valores de los botones de la derecha para que estén en el rango de 0 a 3
                                    pressed_button = i  # Alineamos el rango al de 0-3
                                    
                                    # Comparar la secuencia existente
                                    if len(current_input) < len(sequence):
                                        if sequence[len(current_input)] == pressed_button:
                                            print(f"Jugador inferior tocó correctamente el botón: {pressed_button}")
                                            current_input.append(pressed_button)
                                        else:
                                            print("¡Error en la secuencia!")
                                            reproducir_sonido(4)
                                            mostrar_error()  # Mostrar la X
                                            current_player, sequence = reiniciar_juego(current_player)
                                            print(f"Limpiada la secuencia: {sequence}")
                                            break

                                    elif len(current_input) == len(sequence):  # Agregar un nuevo botón
                                        print(f"Jugador inferior agregó un nuevo botón: {pressed_button}")
                                        current_input.append(pressed_button)

                                    buttons_touched.append(i)
                                    button_pressed = True
                                    break

                # Verificar si la secuencia está completa o si el jugador agregó un nuevo botón
                if len(current_input) == len(sequence) + 1:
                    print(f"Secuencia completa y válida. Secuencia actual: {current_input}")
                    sequence = current_input.copy()  # Actualizar la secuencia
                    current_input = []
                    turn_counter += 1
                    dim_avatar(current_player)
                    current_player = 1 - current_player
                    highlight_avatar(current_player)
                    buttons_touched.clear()
                    button_pressed = False

        # Mostrar la pantalla actualizada
        cv2.imshow("Videobeam", videobeam_screen)

        # Condición de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Al final del juego
    rgb_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()

def detect_color_and_shape(image, min_contour_area=250):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            "Rojo": ([170, 50, 50], [180, 255, 255]),
            "Verde": ([40, 50, 50], [90, 255, 255]),
            "Azul": ([90, 50, 50], [130, 255, 255]),
            "Amarillo": ([20, 100, 100], [30, 255, 255]),
            "Naranja": ([0, 100, 100], [10, 255, 255]),
            "Morado": ([130, 50, 50], [160, 255, 255]),
        }

        detected_shapes = []

        for color_name, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_contour_area:
                    shape = None
                    color_name = None
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    if len(approx) > 4:
                        shape = "círculo"
                    elif len(approx) == 4:
                        shape = "cuadrado"
                    elif len(approx) == 3:
                        shape = "triángulo"

                    detected_shapes.append((shape, color_name, cnt))

        return detected_shapes

def juego_tic_tac_toe(device):
    # Cargar las coordenadas desde el archivo JSON
    with open("config/ultima_configuracion_coordenadas.json", "r") as file:
        coordenadas = json.load(file)

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Dimensiones del área de trabajo
    w = xw_max - xw_min
    h = yw_max - yw_min

    # Configuración del tablero de Tic-Tac-Toe
    num_rows = 3
    num_cols = 3
    cell_width = (xv_max - xv_min) // num_cols
    cell_height = (yv_max - yv_min) // num_rows

    # Inicializar el estado del tablero (vacío)
    tablero = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    tablero_cambiado = False  # Para rastrear si el tablero se actualiza

    # Variables para mantener el estado de la victoria
    victoria_detectada = False
    coordenadas_linea = None  # Almacena (start_pos, end_pos) de la línea ganadora

    # Variables para rastrear el turno y las figuras permitidas
    turno_jugador = None  # Inicialmente sin turno
    player1_shape = None
    player2_shape = None
    figuras_permitidas = set()

    # Variables para detección de movimiento
    previous_frame = None
    movement_threshold = 250  # Umbral ajustado según las necesidades

    # Flag para indicar si el juego está en estado de espera para reiniciar
    esperar_reinicio = False

    # Función para mapear las coordenadas de la imagen a una casilla de la cuadrícula
    def detectar_casilla(x, y):
        # Asegurarnos que las coordenadas están dentro del área de trabajo
        if xw_min <= x <= xw_max and yw_min <= y <= yw_max:
            # Escalar las coordenadas para que coincidan con la cuadrícula
            x_scaled = (x - xw_min) * (xv_max - xv_min) / (xw_max - xw_min) + xv_min
            y_scaled = (y - yw_min) * (yv_max - yv_min) / (yw_max - yw_min) + yv_min

            # Calcular la columna y fila correspondientes
            col = int((x_scaled - xv_min) / cell_width)
            row = int((y_scaled - yv_min) / cell_height)

            # Verificar si la casilla está dentro del tablero
            if 0 <= col < num_cols and 0 <= row < num_rows:
                print(f"Coordenadas ({x},{y}) mapeadas a casilla ({row}, {col})")
                return row, col
            else:
                print(f"Coordenadas ({x},{y}) fuera del rango de casillas.")
                return None
        else:
            print(f"Coordenadas ({x},{y}) fuera del área de trabajo.")
            return None

    # Función para detectar figuras y colores
    def detectar_figuras_y_colores(frame):
        shapes = detect_color_and_shape(frame)  # Asume que esta función está definida y detecta 'círculo', 'triángulo', 'cuadrado'
        return shapes

    # Función para dibujar la línea ganadora
    def dibujar_linea_ganadora(image, start_pos, end_pos):
        cv2.line(image, start_pos, end_pos, (0, 255, 0), thickness=5)

    # Verificar si hay una victoria
    def verificar_victoria(matriz_figuras):
        # Verificar filas, columnas y diagonales
        for i in range(3):
            # Verificar filas
            if (matriz_figuras[i][0] is not None and 
                matriz_figuras[i][1] is not None and 
                matriz_figuras[i][2] is not None and 
                matriz_figuras[i][0][0].lower() == matriz_figuras[i][1][0].lower() == matriz_figuras[i][2][0].lower()):
                return True, (i, 0), (i, 2)
            
            # Verificar columnas
            if (matriz_figuras[0][i] is not None and 
                matriz_figuras[1][i] is not None and 
                matriz_figuras[2][i] is not None and 
                matriz_figuras[0][i][0].lower() == matriz_figuras[1][i][0].lower() == matriz_figuras[2][i][0].lower()):
                return True, (0, i), (2, i)
        
        # Verificar diagonal principal
        if (matriz_figuras[0][0] is not None and 
            matriz_figuras[1][1] is not None and 
            matriz_figuras[2][2] is not None and 
            matriz_figuras[0][0][0].lower() == matriz_figuras[1][1][0].lower() == matriz_figuras[2][2][0].lower()):
            return True, (0, 0), (2, 2)
        
        # Verificar diagonal inversa
        if (matriz_figuras[0][2] is not None and 
            matriz_figuras[1][1] is not None and 
            matriz_figuras[2][0] is not None and 
            matriz_figuras[0][2][0].lower() == matriz_figuras[1][1][0].lower() == matriz_figuras[2][0][0].lower()):
            return True, (0, 2), (2, 0)
        
        return False, None, None

    def dibujar_tablero(image):
        for i in range(1, num_rows):
            # Dibujar líneas horizontales
            cv2.line(image, (xv_min, yv_min + i * cell_height), (xv_max, yv_min + i * cell_height), (255, 255, 255), 2)
        for i in range(1, num_cols):
            # Dibujar líneas verticales
            cv2.line(image, (xv_min + i * cell_width, yv_min), (xv_min + i * cell_width, yv_max), (255, 255, 255), 2)
        

    # Iniciar los streams de la cámara
    rgb_stream = device.create_color_stream()
    rgb_stream.start()

    videobeam_screen = np.zeros((800, 1280, 3), dtype=np.uint8)
    time.sleep(2)
    dibujar_tablero(videobeam_screen)

    while True:
        frame = rgb_stream.read_frame()
        if frame is None:
            continue

        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)

        # Recortar el área de trabajo
        area_trabajo = bgr_data[yw_min:yw_max, xw_min:xw_max]

        # Implementación de la detección de movimiento
        # Convertir a escala de grises para comparar la diferencia
        gray_current = cv2.cvtColor(area_trabajo, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray_current
            continue

        # Calcular la diferencia entre el frame actual y el anterior
        frame_diff = cv2.absdiff(previous_frame, gray_current)
        _, thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Contar los píxeles que han cambiado significativamente
        movement_area = np.sum(thresh_diff > 0)

        if movement_area > movement_threshold:
            print("Movimiento detectado, pausa en el stream")
            time.sleep(0.2)  # Pausar por 0.2 segundos si hay movimiento
            # Actualizar el frame anterior después de la pausa
            previous_frame = gray_current
            continue
        else:
            # Se sigue la lógica del juego
            previous_frame = gray_current

        # Detectar las figuras y colores en el área de trabajo
        figuras_detectadas = detectar_figuras_y_colores(area_trabajo)

        # Actualizar el tablero con las nuevas detecciones
        for shape, color, contour in figuras_detectadas:
            # Verificación adicional para ver si 'figuras_detectadas' contiene datos válidos
            if shape is None:
                print(f"Error: Figura no  detectado correctamente. Figura: {shape}")
                continue

            # Calcular el centro de la figura detectada
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + xw_min
                cy = int(M['m01'] / M['m00']) + yw_min
                casilla = detectar_casilla(cx, cy)
                if casilla:
                    row, col = casilla
                    print(f"Intentando colocar {shape}, {color} en casilla ({row}, {col})")
                    if tablero[row][col] is None:
                        # Asignar las figuras permitidas si aún no están definidas
                        if player1_shape is None:
                            player1_shape = shape.lower()
                            figuras_permitidas.add(player1_shape)
                            turno_jugador = player1_shape  # Comienza el primer jugador
                            print(f"Figura del jugador 1 establecida como: {player1_shape}")
                        
                        elif player2_shape is None and shape.lower() != player1_shape:
                            player2_shape = shape.lower()
                            figuras_permitidas.add(player2_shape)
                            turno_jugador = player2_shape  # Cambia el turno al segundo jugador
                            print(f"Figura del jugador 2 establecida como: {player2_shape}")
                        
                        # Verificar si la figura detectada está permitida
                        if shape.lower() in figuras_permitidas:
                            # Solo permitir colocar si es el turno correcto
                            if shape.lower() == turno_jugador:
                                tablero[row][col] = (shape, color)
                                tablero_cambiado = True
                                print(f"Casilla ({row}, {col}) actualizada a {tablero[row][col]}")
                                # Cambiar el turno al otro jugador
                                if turno_jugador == player1_shape:
                                    turno_jugador = player2_shape
                                elif turno_jugador == player2_shape:
                                    turno_jugador = player1_shape
                                print(f"Turno cambiado a: {turno_jugador}")
                            else:
                                print(f"No es el turno para colocar un {shape}. Turno actual: {turno_jugador}")
                        else:
                            print(f"Figura {shape} no está permitida en este juego.")
                    else:
                        print(f"Casilla ({row}, {col}) ya está ocupada por {tablero[row][col]}")
                else:
                    print(f"Coordenadas ({cx},{cy}) no mapeadas a ninguna casilla válida.")
            else:
                print("Error: No se pudo calcular el centro de la figura detectada.")

        # Verificar si hay un ganador o si el tablero está completo
        if tablero_cambiado:
            victoria, inicio, fin = verificar_victoria(tablero)
            tablero_cambiado = False  # Restablecer la variable

            if victoria:
                victoria_detectada = True
                # Convertir las posiciones de la cuadrícula a coordenadas de pantalla
                start_x = xv_min + inicio[1] * cell_width + cell_width // 2
                start_y = yv_min + inicio[0] * cell_height + cell_height // 2
                end_x = xv_min + fin[1] * cell_width + cell_width // 2
                end_y = yv_min + fin[0] * cell_height + cell_height // 2
                coordenadas_linea = ((start_x, start_y), (end_x, end_y))
                print(f"Victoria detectada! Línea desde {coordenadas_linea[0]} hasta {coordenadas_linea[1]}")

            # Verificar si el tablero está lleno y no hay victoria (empate)
            tablero_lleno = all(all(cell is not None for cell in row_cells) for row_cells in tablero)
            if tablero_lleno and not victoria_detectada:
                print("Tablero completo sin victorias. Empate.")
                victoria_detectada = True  # Considerar empate como estado de victoria para la línea
                coordenadas_linea = None  # No dibujar línea en caso de empate

        # Detectar si el juego ha terminado y limpiar el tablero
        if victoria_detectada or (all(all(cell is not None for cell in row) for row in tablero)):
            # Verificar si ya se está esperando reinicio para evitar múltiples mensajes
            if not esperar_reinicio:
                print("Juego finalizado. Retira todas las figuras para reiniciar.")
                esperar_reinicio = True

            # Revisar si ya no hay figuras en el área de trabajo
            if not figuras_detectadas:
                print("No hay figuras detectadas. Reiniciando juego.")
                # Reiniciar el juego
                tablero = [[None for _ in range(num_cols)] for _ in range(num_rows)]
                tablero_cambiado = False
                victoria_detectada = False
                coordenadas_linea = None
                turno_jugador = None
                player1_shape = None
                player2_shape = None
                figuras_permitidas = set()
                esperar_reinicio = False
                print("Juego reiniciado.")
        else:
            esperar_reinicio = False  # Resetear el flag si el juego no está finalizado

        # Redibujar el videobeam_screen en cada iteración
        videobeam_screen = np.zeros((800, 1280, 3), dtype=np.uint8)
        dibujar_tablero(videobeam_screen)

        # Dibujar las figuras en el tablero
        for row in range(num_rows):
            for col in range(num_cols):
                if tablero[row][col] is not None:
                    shape, color = tablero[row][col]
                    # Calcular la posición central de la celda
                    center_x = xv_min + col * cell_width + cell_width // 2
                    center_y = yv_min + row * cell_height + cell_height // 2
                    # Dibujar la figura según su tipo con tamaño reducido
                    if shape.lower() == 'círculo':
                        cv2.circle(videobeam_screen, (center_x, center_y), min(cell_width, cell_height)//4, (0, 255, 0), thickness=2)
                    elif shape.lower() == 'triángulo':
                        # Dibujar un triángulo más pequeño
                        side_length = min(cell_width, cell_height) // 3
                        pt1 = (center_x, center_y - side_length)
                        pt2 = (center_x - side_length, center_y + side_length)
                        pt3 = (center_x + side_length, center_y + side_length)
                        pts = np.array([pt1, pt2, pt3], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(videobeam_screen, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                    elif shape.lower() == 'cuadrado':
                        # Dibujar un cuadrado más pequeño
                        side_length = min(cell_width, cell_height) // 3
                        pt1 = (center_x - side_length, center_y - side_length)
                        pt2 = (center_x + side_length, center_y - side_length)
                        pt3 = (center_x + side_length, center_y + side_length)
                        pt4 = (center_x - side_length, center_y + side_length)
                        pts = np.array([pt1, pt2, pt3, pt4], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(videobeam_screen, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    # Añade más formas si es necesario

        # Dibujar la línea ganadora si se ha detectado una victoria
        if victoria_detectada and coordenadas_linea is not None:
            dibujar_linea_ganadora(videobeam_screen, coordenadas_linea[0], coordenadas_linea[1])

        # Mostrar la ventana de salida
        cv2.imshow("Área de Trabajo", area_trabajo)
        cv2.namedWindow("Tic-Tac-Toe", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Tic-Tac-Toe", 1920, 0)
        cv2.setWindowProperty("Tic-Tac-Toe", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Tic-Tac-Toe", videobeam_screen)

        # Verificar si se presiona la tecla 'q' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    rgb_stream.stop()
    cv2.destroyAllWindows()

def mostrar_seleccion_historias(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de historias con 6 cards.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        str o None: Nombre de la historia seleccionada o None si se canceló
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
    historias_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        historias_screen[y, :] = [b, g, r]
    
    # Dibujar el logo (un poco más pequeño - 95% del tamaño original)
    def draw_logo_smaller_historias(screen):
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
            # Si no hay logo, usar la función original
            draw_logo_func(screen)
    
    draw_logo_smaller_historias(historias_screen)
    
    # Función para dibujar card redonda con X (estilo infantil) - en la parte superior derecha
    def draw_close_card_historias(screen, elevated=False):
        """
        Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca
        en la parte superior derecha
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
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
    close_card_radius_historias = 50
    close_card_margin_x_historias = 180
    close_card_margin_y_historias = 80
    close_card_center_x_historias = view_width - close_card_margin_x_historias - close_card_radius_historias
    close_card_center_y_historias = close_card_margin_y_historias + close_card_radius_historias
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_historias = close_card_radius_historias * 2.4
    close_card_detection_x_historias = close_card_center_x_historias - close_card_radius_historias * 1.2
    close_card_detection_y_historias = close_card_center_y_historias - close_card_radius_historias * 1.2
    close_card_detection_w_historias = close_card_detection_size_historias
    close_card_detection_h_historias = close_card_detection_size_historias
    
    # Función para detectar si se tocó la card de cerrar
    def detectar_close_card_touch_historias(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar"""
        return (close_card_detection_x_historias <= x_touch <= close_card_detection_x_historias + close_card_detection_w_historias and
                close_card_detection_y_historias <= y_touch <= close_card_detection_y_historias + close_card_detection_h_historias)
    
    # Definir las 6 historias con las imágenes de sujetos
    historias = [
        {"nombre": "Niño", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/niño.png"},
        {"nombre": "Niña", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Niña.png"},
        {"nombre": "Doctor", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Doctor.png"},
        {"nombre": "Maestra", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Maestra.png"},
        {"nombre": "Policia", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/policia.png"},
        {"nombre": "Perro", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/perro.png"}
    ]
    
    # Cargar imágenes de las historias si existen
    historia_images = {}
    for historia in historias:
        if "imagen" in historia:
            imagen_path = historia["imagen"]
            # Intentar cargar la imagen con la ruta exacta
            if os.path.exists(imagen_path):
                img = cv2.imread(imagen_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    historia_images[historia["nombre"]] = img
                    print(f"✓ Imagen cargada para {historia['nombre']}: {imagen_path}")
                else:
                    print(f"⚠ No se pudo cargar la imagen para {historia['nombre']}: {imagen_path}")
            else:
                # Si no existe con la ruta exacta, intentar buscar variaciones
                base_path = os.path.dirname(imagen_path)
                base_name = os.path.basename(imagen_path)
                base_name_no_ext = os.path.splitext(base_name)[0]
                ext = os.path.splitext(base_name)[1]
                
                # Intentar diferentes variaciones de mayúsculas/minúsculas
                posibles_nombres = [
                    base_name,  # Original
                    base_name.lower(),  # Todo minúsculas
                    base_name.upper(),  # Todo mayúsculas
                    base_name.capitalize(),  # Primera mayúscula
                    base_name_no_ext.lower() + ext,  # Nombre minúsculas
                    base_name_no_ext.upper() + ext,  # Nombre mayúsculas
                    base_name_no_ext.capitalize() + ext,  # Nombre capitalizado
                ]
                
                imagen_encontrada = False
                for nombre_variante in posibles_nombres:
                    ruta_variante = os.path.join(base_path, nombre_variante)
                    if os.path.exists(ruta_variante):
                        img = cv2.imread(ruta_variante, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            historia_images[historia["nombre"]] = img
                            print(f"✓ Imagen cargada para {historia['nombre']}: {ruta_variante} (variante encontrada)")
                            imagen_encontrada = True
                            break
                
                if not imagen_encontrada:
                    print(f"⚠ No se encontró la imagen para {historia['nombre']}: {imagen_path}")
    
    # Título
    titulo_texto = ""
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 120  # Posición del título
    
    # Función para dibujar el título
    def draw_titulo_historias(screen):
        """Dibuja el título de la pantalla de selección de historias"""
        # Sombra del título
        put_text_ubuntu(screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
                   font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
        # Título principal
        put_text_ubuntu(screen, titulo_texto, (text_x_titulo, text_y_titulo), 
                   font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Función para dibujar el texto de sujetos seleccionados
    def draw_sujetos_seleccionados(screen, num_seleccionados):
        """
        Dibuja el texto que muestra el número de sujetos seleccionados
        """
        texto = f"Sujetos seleccionados: {num_seleccionados}"
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
    
    # Dibujar el título en la pantalla inicial
    draw_titulo_historias(historias_screen)
    
    # Dimensiones de las cards (mismas que en acciones: 6 cards en 2 filas de 3)
    card_width = 240
    card_height = 210
    card_spacing = 25
    
    # Calcular posiciones (igual que acciones: centradas con mismo offset)
    total_width = 3 * card_width + 2 * card_spacing
    start_x = (view_width - total_width) // 2 - 30  # Mismo que acciones
    start_y = 180  # Mismo que acciones
    
    historia_positions = {}
    for idx, historia in enumerate(historias):
        row = idx // 3
        col = idx % 3
        x = start_x + col * (card_width + card_spacing)
        y = start_y + row * (card_height + card_spacing)
        historia_positions[historia["nombre"]] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': historia["color"],
            'imagen': historia_images.get(historia["nombre"])
        }
    
    # Función para dibujar las cards de historias
    def draw_historia_cards(screen, historia_positions, selected_cards=None):
        if selected_cards is None:
            selected_cards = []
        for nombre, pos in historia_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            color = pos['color']
            is_selected = (nombre in selected_cards)
            
            # Mantener el mismo tamaño siempre (sin expansión)
            # Efecto visual si está seleccionada: borde verde más grueso
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
                            (x + offset_layer, y + offset_layer), 
                            (x + w + offset_layer, y + h + offset_layer), 
                            shadow_color_layer, -1)
            
            # Dibujar la imagen como fondo completo de la card si existe
            if pos['imagen'] is not None:
                img = pos['imagen'].copy()
                img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                
                if x >= 0 and y >= 0 and x + w <= screen.shape[1] and y + h <= screen.shape[0]:
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
            
            # Dibujar borde (verde si está seleccionada)
            cv2.rectangle(screen, (x, y), (x + w, y + h), border_color, border_thickness)
    
    # Función para dibujar el botón "Siguiente"
    def draw_siguiente_button(screen, elevated=False, blocked=False):
        """
        Dibuja un botón "Siguiente" en la parte inferior de la pantalla
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, el botón se dibuja elevado (efecto de levantarse)
            blocked: Si True, el botón se dibuja bloqueado (gris, deshabilitado)
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
        
        # Texto "Siguiente" - aumentado y centrado
        texto = "Siguiente"
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
    
    # Variables para el botón "Siguiente"
    siguiente_button_width = 200
    siguiente_button_height = 60
    siguiente_button_margin_bottom = 90  # Ajustado para bajar un poco el botón
    siguiente_button_x = (view_width - siguiente_button_width) // 2
    siguiente_button_y = view_height - siguiente_button_margin_bottom - siguiente_button_height
    
    # Función para detectar si se tocó el botón "Siguiente"
    def detectar_siguiente_button_touch(x_touch, y_touch):
        """Detecta si el toque está dentro del área del botón Siguiente"""
        return (siguiente_button_x <= x_touch <= siguiente_button_x + siguiente_button_width and
                siguiente_button_y <= y_touch <= siguiente_button_y + siguiente_button_height)
    
    # Dibujar las cards y la X (sin selección inicial)
    draw_titulo_historias(historias_screen)
    draw_historia_cards(historias_screen, historia_positions, selected_cards=[])
    draw_sujetos_seleccionados(historias_screen, 0)  # Inicialmente 0 seleccionados
    draw_close_card_historias(historias_screen, elevated=False)
    draw_siguiente_button(historias_screen, elevated=False, blocked=True)  # Bloqueado inicialmente
    
    # Configurar ventana
    window_name = existing_window_name if existing_window_name else "Selección de Historias"
    window_exists = False
    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if prop >= 0:
            window_exists = True
    except:
        window_exists = False
    
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.waitKey(50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(50)
    else:
        try:
            cv2.moveWindow(window_name, 1920, 0)
            cv2.waitKey(10)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except:
            pass
    
    # Mostrar en pantalla
    historias_screen_scaled = scale_to_videobeam(historias_screen)
    cv2.imshow(window_name, historias_screen_scaled)
    
    # Iniciar streams de cámara para detección de toques
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    historias_seleccionadas = []  # Lista de historias seleccionadas
    siguiente_elevated = False
    siguiente_pressed = False
    siguiente_press_frames = 0
    close_elevated = False
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
            draw_logo_smaller_historias(temp_screen)
            draw_titulo_historias(temp_screen)
            draw_historia_cards(temp_screen, historia_positions, selected_cards=historias_seleccionadas)
            draw_sujetos_seleccionados(temp_screen, len(historias_seleccionadas))
            draw_close_card_historias(temp_screen, elevated=False)
            # Bloquear botón si hay menos de 1 o más de 2 selecciones
            is_blocked = len(historias_seleccionadas) < 1 or len(historias_seleccionadas) > 2
            draw_siguiente_button(temp_screen, elevated=False, blocked=is_blocked)
            historias_screen_scaled = scale_to_videobeam(temp_screen)
            cv2.imshow(window_name, historias_screen_scaled)
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
            
            # Verificar si se tocó la X
            if detectar_close_card_touch_historias(x_touch, y_touch):
                print("Card de cerrar tocada en selección de historias")
                # Mostrar efecto de elevación en la X
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_smaller_historias(temp_screen)
                draw_titulo_historias(temp_screen)
                draw_historia_cards(temp_screen, historia_positions, selected_cards=historias_seleccionadas)
                draw_sujetos_seleccionados(temp_screen, len(historias_seleccionadas))
                draw_close_card_historias(temp_screen, elevated=True)
                # Bloquear botón si hay menos de 1 o más de 2 selecciones
                is_blocked = len(historias_seleccionadas) < 1 or len(historias_seleccionadas) > 2
                draw_siguiente_button(temp_screen, elevated=False, blocked=is_blocked)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver al menú principal (retornar None)
                rgb_stream.stop()
                depth_stream.stop()
                return None
            
            # Verificar si se está tocando el botón "Siguiente"
            if detectar_siguiente_button_touch(x_touch, y_touch):
                if 1 <= len(historias_seleccionadas) <= 2:  # Requiere mínimo 1 y máximo 2 sujetos
                    if not siguiente_pressed:
                        # Iniciar el efecto de elevación
                        siguiente_pressed = True
                        siguiente_press_frames = 0
                        siguiente_elevated = True
                        print("Botón Siguiente presionado")
                else:
                    siguiente_elevated = False
                    siguiente_pressed = False
                    siguiente_press_frames = 0
                    print("Botón Siguiente bloqueado - selecciona entre 1 y 2 sujetos")
            else:
                # Si no se está tocando el botón, resetear estado de elevación pero mantener pressed si está procesándose
                if siguiente_press_frames == 0:  # Solo resetear si no está en proceso de ser procesado
                    siguiente_elevated = False
                # No resetear siguiente_pressed aquí, se resetea después de procesar
                
                # Verificar si se seleccionó una historia
                historia_seleccionada_temp = None
                for nombre, pos in historia_positions.items():
                    x, y = pos['x'], pos['y']
                    w, h = pos['width'], pos['height']
                    if x <= x_touch <= x + w and y <= y_touch <= y + h:
                        historia_seleccionada_temp = nombre
                        break
                
                if historia_seleccionada_temp:
                    # Toggle de selección: si ya está seleccionada, deseleccionarla; si no, seleccionarla
                    if historia_seleccionada_temp in historias_seleccionadas:
                        historias_seleccionadas.remove(historia_seleccionada_temp)
                        print(f"Historia deseleccionada: {historia_seleccionada_temp}")
                    else:
                        # Limitar a máximo 2 sujetos
                        if len(historias_seleccionadas) < 2:
                            historias_seleccionadas.append(historia_seleccionada_temp)
                            print(f"Historia seleccionada: {historia_seleccionada_temp}")
                            _historia_tts_speak(historia_seleccionada_temp, tipo="sujeto")
                        else:
                            print(f"Ya has seleccionado el máximo de 2 sujetos")
                    print(f"Historias seleccionadas: {historias_seleccionadas}")
                    # Continuar en el bucle para mostrar la selección actualizada
        
        # Procesar el botón "Siguiente" si está presionado
        if siguiente_pressed and 1 <= len(historias_seleccionadas) <= 2:  # Asegurar que hay entre 1 y 2 sujetos seleccionados
            siguiente_press_frames += 1
            if siguiente_press_frames >= 10:  # Después de 10 frames, procesar la acción
                print("Botón Siguiente procesado - pasando a selección de acciones")
                # Resetear el estado del botón antes de cambiar de vista
                siguiente_pressed = False
                siguiente_press_frames = 0
                siguiente_elevated = False
                # Detener streams de historias antes de ir a acciones
                rgb_stream.stop()
                depth_stream.stop()
                
                # Llamar a la vista de selección de acciones
                acciones_seleccionadas = mostrar_seleccion_acciones(
                    device, coordenadas, dmax_map, dmin_map, draw_logo_func,
                    existing_window_name=window_name
                )
                
                # Si se canceló o se presionó la X (retornó None o "MENU")
                if acciones_seleccionadas is None:
                    # Reiniciar streams para volver a la vista de historias
                    rgb_stream = device.create_color_stream()
                    depth_stream = device.create_depth_stream()
                    rgb_stream.start()
                    depth_stream.start()
                    # Reiniciar el contador de frames
                    frame_count = 0
                    siguiente_elevated = False
                    siguiente_pressed = False
                    siguiente_press_frames = 0
                    # Continuar en el bucle de selección de historias
                    continue
                elif acciones_seleccionadas == "MENU":
                    # Si se presionó la X en acciones, volver al menú principal
                    rgb_stream.stop()
                    depth_stream.stop()
                    return None
                elif acciones_seleccionadas == "BACK":
                    # Si se presionó la flecha de retroceso, volver a la vista de historias manteniendo las selecciones
                    rgb_stream = device.create_color_stream()
                    depth_stream = device.create_depth_stream()
                    rgb_stream.start()
                    depth_stream.start()
                    # Reiniciar el contador de frames
                    frame_count = 0
                    siguiente_elevated = False
                    siguiente_pressed = False
                    siguiente_press_frames = 0
                    # Continuar en el bucle de selección de historias (las selecciones se mantienen)
                    continue
                else:
                    # Si acciones_seleccionadas es un diccionario, significa que ya pasó por lugares
                    if isinstance(acciones_seleccionadas, dict):
                        # Llamar a la vista final con todas las selecciones
                        resultado_final = mostrar_vista_final(
                            device, coordenadas, dmax_map, dmin_map, draw_logo_func,
                            historias_seleccionadas,
                            acciones_seleccionadas.get('acciones', []),
                            acciones_seleccionadas.get('lugares', []),
                            existing_window_name=window_name
                        )
                        
                        # Si se canceló desde la vista final, retornar "MENU" o None
                        if resultado_final is None or resultado_final == "MENU":
                            return resultado_final
                        elif resultado_final == "RESTART_FULL":
                            # Volver a la primera selección del juego (limpiar todas las selecciones)
                            historias_seleccionadas = []
                            acciones_seleccionadas = []  # Reset to empty list (will be a list, not a dict)
                            # Reiniciar streams para volver a la vista de historias
                            try:
                                rgb_stream.stop()
                                depth_stream.stop()
                            except:
                                pass
                            rgb_stream = device.create_color_stream()
                            depth_stream = device.create_depth_stream()
                            rgb_stream.start()
                            depth_stream.start()
                            # Reiniciar el contador de frames
                            frame_count = 0
                            siguiente_elevated = False
                            siguiente_pressed = False
                            siguiente_press_frames = 0
                            continue  # Continuar en el bucle para empezar de nuevo
                        else:
                            # Retornar todas las selecciones
                            return {
                                'sujetos': historias_seleccionadas,
                                'acciones': acciones_seleccionadas.get('acciones', []),
                                'lugares': acciones_seleccionadas.get('lugares', [])
                            }
                    else:
                        # Si es una lista, solo retornar las historias y acciones (caso antiguo)
                        return {
                            'sujetos': historias_seleccionadas,
                            'acciones': acciones_seleccionadas
                        }
        
        # Redibujar la pantalla en cada frame
        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            temp_screen[y, :] = [b, g, r]
        draw_logo_smaller_historias(temp_screen)
        draw_titulo_historias(temp_screen)
        draw_historia_cards(temp_screen, historia_positions, selected_cards=historias_seleccionadas)
        draw_sujetos_seleccionados(temp_screen, len(historias_seleccionadas))
        draw_close_card_historias(temp_screen, elevated=close_elevated)
        # Bloquear botón si hay menos de 1 o más de 2 sujetos seleccionados
        is_blocked = len(historias_seleccionadas) < 1 or len(historias_seleccionadas) > 2
        draw_siguiente_button(temp_screen, elevated=siguiente_elevated, blocked=is_blocked)
        historias_screen_scaled = scale_to_videobeam(temp_screen)
        cv2.imshow(window_name, historias_screen_scaled)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Detener streams y cerrar
    rgb_stream.stop()
    depth_stream.stop()
    return historias_seleccionadas if historias_seleccionadas else None  # Retornar las historias seleccionadas (o None si no hay ninguna)

def mostrar_seleccion_acciones(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de acciones con 6 cards.
    
    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)
    
    Returns:
        list o None: Lista de acciones seleccionadas o None si se canceló
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
    acciones_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        acciones_screen[y, :] = [b, g, r]
    
    # Dibujar el logo (un poco más pequeño - 95% del tamaño original)
    def draw_logo_smaller_acciones(screen):
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
    
    draw_logo_smaller_acciones(acciones_screen)
    
    # Función para dibujar card redonda con flecha hacia la izquierda (estilo infantil) - en la parte superior izquierda
    def draw_back_card_acciones(screen, elevated=False):
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
    def draw_close_card_acciones(screen, elevated=False):
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
    back_card_radius_acciones = 50
    back_card_margin_x_acciones = 120  # Movido más a la izquierda
    back_card_margin_y_acciones = 80
    back_card_center_x_acciones = back_card_margin_x_acciones + back_card_radius_acciones
    back_card_center_y_acciones = back_card_margin_y_acciones + back_card_radius_acciones
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    back_card_detection_size_acciones = back_card_radius_acciones * 2.4
    back_card_detection_x_acciones = back_card_center_x_acciones - back_card_radius_acciones * 1.2
    back_card_detection_y_acciones = back_card_center_y_acciones - back_card_radius_acciones * 1.2
    back_card_detection_w_acciones = back_card_detection_size_acciones
    back_card_detection_h_acciones = back_card_detection_size_acciones
    
    # Función para detectar si se tocó la card de retroceso
    def detectar_back_card_touch_acciones(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de retroceso"""
        return (back_card_detection_x_acciones <= x_touch <= back_card_detection_x_acciones + back_card_detection_w_acciones and
                back_card_detection_y_acciones <= y_touch <= back_card_detection_y_acciones + back_card_detection_h_acciones)
    
    # Variables para la card de cerrar (necesarias para la detección)
    close_card_radius_acciones = 50
    close_card_margin_x_acciones = 180
    close_card_margin_y_acciones = 80
    close_card_center_x_acciones = view_width - close_card_margin_x_acciones - close_card_radius_acciones
    close_card_center_y_acciones = close_card_margin_y_acciones + close_card_radius_acciones
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
    close_card_detection_size_acciones = close_card_radius_acciones * 2.4
    close_card_detection_x_acciones = close_card_center_x_acciones - close_card_radius_acciones * 1.2
    close_card_detection_y_acciones = close_card_center_y_acciones - close_card_radius_acciones * 1.2
    close_card_detection_w_acciones = close_card_detection_size_acciones
    close_card_detection_h_acciones = close_card_detection_size_acciones
    
    # Función para detectar si se tocó la card de cerrar
    def detectar_close_card_touch_acciones(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar"""
        return (close_card_detection_x_acciones <= x_touch <= close_card_detection_x_acciones + close_card_detection_w_acciones and
                close_card_detection_y_acciones <= y_touch <= close_card_detection_y_acciones + close_card_detection_h_acciones)
    
    # Definir las 6 acciones
    acciones = [
        {"nombre": "Dar", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Dar.png"},
        {"nombre": "Ayudar", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Ayudar.png"},
        {"nombre": "Correr", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Correr.png"},
        {"nombre": "Jugar", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Jugar.png"},
        {"nombre": "Llamar", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Llamar.png"},
        {"nombre": "Trabajar", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Trabajar.png"}
    ]
    
    # Cargar imágenes de las acciones si existen
    accion_images = {}
    for accion in acciones:
        if "imagen" in accion and os.path.exists(accion["imagen"]):
            img = cv2.imread(accion["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                accion_images[accion["nombre"]] = img
                print(f"✓ Imagen cargada para {accion['nombre']}: {accion['imagen']}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {accion['nombre']}: {accion['imagen']}")
    
    # Título
    titulo_texto = "Selecciona una accion"
    font_titulo = cv2.FONT_HERSHEY_DUPLEX
    font_scale_titulo = 1.2
    thickness_titulo = 3
    text_size_titulo, _ = cv2.getTextSize(titulo_texto, font_titulo, font_scale_titulo, thickness_titulo)
    text_x_titulo = (view_width - text_size_titulo[0]) // 2
    text_y_titulo = 150  # Mover título más arriba
    # Sombra del título
    put_text_ubuntu(acciones_screen, titulo_texto, (text_x_titulo + 2, text_y_titulo + 2), 
               font_scale_titulo, (0, 0, 0), thickness_titulo + 2)
    # Título principal
    put_text_ubuntu(acciones_screen, titulo_texto, (text_x_titulo, text_y_titulo), 
               font_scale_titulo, (255, 255, 255), thickness_titulo)
    
    # Función para dibujar el texto de acciones seleccionadas
    def draw_acciones_seleccionadas(screen, num_seleccionados):
        """
        Dibuja el texto que muestra el número de acciones seleccionadas
        """
        texto = f"Acciones seleccionadas: {num_seleccionados}"
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
    
    accion_positions = {}
    for idx, accion in enumerate(acciones):
        row = idx // 3
        col = idx % 3
        x = start_x + col * (card_width + card_spacing)
        y = start_y + row * (card_height + card_spacing)
        accion_positions[accion["nombre"]] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': accion["color"],
            'imagen': accion_images.get(accion["nombre"])
        }
    
    # Función para dibujar las cards de acciones
    def draw_accion_cards(screen, accion_positions, selected_cards=None):
        if selected_cards is None:
            selected_cards = []
        for nombre, pos in accion_positions.items():
            x = pos['x']
            y = pos['y']
            w = pos['width']
            h = pos['height']
            color = pos['color']
            is_selected = (nombre in selected_cards)
            
            # Efecto visual si está seleccionada: hacer la card más ancha (como en escenarios)
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
            if pos['imagen'] is not None:
                img = pos['imagen'].copy()
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
    
    # Función para dibujar el botón "Siguiente"
    def draw_siguiente_button_acciones(screen, elevated=False, blocked=False):
        """
        Dibuja un botón "Siguiente" en la parte inferior de la pantalla
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
        
        # Texto "Siguiente" - aumentado y centrado
        texto = "Siguiente"
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
    
    # Variables para el botón "Siguiente"
    siguiente_button_width_acciones = 200
    siguiente_button_height_acciones = 60
    siguiente_button_margin_bottom_acciones = 90  # Ajustado para bajar un poco el botón
    siguiente_button_x_acciones = (view_width - siguiente_button_width_acciones) // 2
    siguiente_button_y_acciones = view_height - siguiente_button_margin_bottom_acciones - siguiente_button_height_acciones
    
    # Función para detectar si se tocó el botón "Siguiente"
    def detectar_siguiente_button_touch_acciones(x_touch, y_touch):
        """Detecta si el toque está dentro del área del botón Siguiente"""
        return (siguiente_button_x_acciones <= x_touch <= siguiente_button_x_acciones + siguiente_button_width_acciones and
                siguiente_button_y_acciones <= y_touch <= siguiente_button_y_acciones + siguiente_button_height_acciones)
    
    # Dibujar las cards y la X (sin selección inicial)
    draw_accion_cards(acciones_screen, accion_positions, selected_cards=[])
    draw_acciones_seleccionadas(acciones_screen, 0)  # Inicialmente 0 seleccionados
    draw_close_card_acciones(acciones_screen, elevated=False)
    draw_siguiente_button_acciones(acciones_screen, elevated=False, blocked=True)  # Bloqueado inicialmente
    
    # Configurar ventana
    window_name = existing_window_name if existing_window_name else "Selección de Acciones"
    window_exists = False
    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if prop >= 0:
            window_exists = True
    except:
        window_exists = False
    
    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.waitKey(50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(50)
    else:
        try:
            cv2.moveWindow(window_name, 1920, 0)
            cv2.waitKey(10)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except:
            pass
    
    # Mostrar en pantalla
    acciones_screen_scaled = scale_to_videobeam(acciones_screen)
    cv2.imshow(window_name, acciones_screen_scaled)
    
    # Iniciar streams de cámara para detección de toques
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    
    acciones_seleccionadas = []  # Lista de acciones seleccionadas
    siguiente_elevated_acciones = False
    siguiente_pressed_acciones = False
    siguiente_press_frames_acciones = 0
    back_elevated_acciones = False
    close_elevated_acciones = False
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
            draw_logo_smaller_acciones(temp_screen)
            draw_accion_cards(temp_screen, accion_positions, selected_cards=acciones_seleccionadas)
            draw_acciones_seleccionadas(temp_screen, len(acciones_seleccionadas))
            draw_back_card_acciones(temp_screen, elevated=False)
            draw_close_card_acciones(temp_screen, elevated=False)
            is_blocked = len(acciones_seleccionadas) != 1
            draw_siguiente_button_acciones(temp_screen, elevated=False, blocked=is_blocked)
            acciones_screen_scaled = scale_to_videobeam(temp_screen)
            cv2.imshow(window_name, acciones_screen_scaled)
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
            if detectar_back_card_touch_acciones(x_touch, y_touch):
                print("Card de retroceso tocada en selección de acciones")
                # Mostrar efecto de elevación en la flecha
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_func(temp_screen)
                draw_accion_cards(temp_screen, accion_positions, selected_cards=acciones_seleccionadas)
                draw_acciones_seleccionadas(temp_screen, len(acciones_seleccionadas))
                draw_back_card_acciones(temp_screen, elevated=True)
                draw_close_card_acciones(temp_screen, elevated=False)
                is_blocked = len(acciones_seleccionadas) != 1
                draw_siguiente_button_acciones(temp_screen, elevated=False, blocked=is_blocked)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver a la vista anterior (retornar "BACK" para indicar retroceso)
                rgb_stream.stop()
                depth_stream.stop()
                return "BACK"
            
            # Verificar si se tocó la X
            if detectar_close_card_touch_acciones(x_touch, y_touch):
                print("Card de cerrar tocada en selección de acciones")
                # Mostrar efecto de elevación en la X
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_func(temp_screen)
                draw_accion_cards(temp_screen, accion_positions, selected_cards=acciones_seleccionadas)
                draw_acciones_seleccionadas(temp_screen, len(acciones_seleccionadas))
                draw_back_card_acciones(temp_screen, elevated=False)
                draw_close_card_acciones(temp_screen, elevated=True)
                is_blocked = len(acciones_seleccionadas) != 1
                draw_siguiente_button_acciones(temp_screen, elevated=False, blocked=is_blocked)
                temp_screen_scaled = scale_to_videobeam(temp_screen)
                cv2.imshow(window_name, temp_screen_scaled)
                cv2.waitKey(200)
                # Volver al menú principal (retornar "MENU" para indicar que se debe volver al menú)
                rgb_stream.stop()
                depth_stream.stop()
                return "MENU"
            
            # Verificar si se está tocando el botón "Siguiente"
            if detectar_siguiente_button_touch_acciones(x_touch, y_touch):
                if len(acciones_seleccionadas) == 1:  # Requiere exactamente 1 acción
                    if not siguiente_pressed_acciones:
                        # Iniciar el efecto de elevación
                        siguiente_pressed_acciones = True
                        siguiente_press_frames_acciones = 0
                        siguiente_elevated_acciones = True
                        print("Botón Siguiente presionado en selección de acciones")
                else:
                    siguiente_elevated_acciones = False
                    siguiente_pressed_acciones = False
                    siguiente_press_frames_acciones = 0
                    print("Botón Siguiente bloqueado - selecciona al menos una acción")
            else:
                # Si no se está tocando el botón, resetear estado
                if siguiente_pressed_acciones:
                    siguiente_pressed_acciones = False
                    siguiente_press_frames_acciones = 0
                siguiente_elevated_acciones = False
                
                # Verificar si se seleccionó una acción
                accion_seleccionada_temp = None
                for nombre, pos in accion_positions.items():
                    x, y = pos['x'], pos['y']
                    w, h = pos['width'], pos['height']
                    if x <= x_touch <= x + w and y <= y_touch <= y + h:
                        accion_seleccionada_temp = nombre
                        break
                
                if accion_seleccionada_temp:
                    # Toggle de selección: si ya está seleccionada, deseleccionarla; si no, seleccionarla
                    if accion_seleccionada_temp in acciones_seleccionadas:
                        acciones_seleccionadas.remove(accion_seleccionada_temp)
                        print(f"Acción deseleccionada: {accion_seleccionada_temp}")
                    else:
                        # Limitar a máximo 1 acción
                        if len(acciones_seleccionadas) < 1:
                            acciones_seleccionadas.append(accion_seleccionada_temp)
                            print(f"Acción seleccionada: {accion_seleccionada_temp}")
                            _historia_tts_speak(accion_seleccionada_temp, tipo="accion")
                        else:
                            # Si ya hay una acción seleccionada, reemplazarla
                            acciones_seleccionadas.clear()
                            acciones_seleccionadas.append(accion_seleccionada_temp)
                            print(f"Acción seleccionada: {accion_seleccionada_temp} (reemplazando selección anterior)")
                            _historia_tts_speak(accion_seleccionada_temp, tipo="accion")
                    print(f"Acciones seleccionadas: {acciones_seleccionadas}")
                    # Continuar en el bucle para mostrar la selección actualizada
        
        # Procesar el botón "Siguiente" si está presionado
        if siguiente_pressed_acciones:
            siguiente_press_frames_acciones += 1
            if siguiente_press_frames_acciones >= 10:  # Después de 10 frames, procesar la acción
                print("Botón Siguiente procesado - pasando a selección de lugares")
                # Detener streams de acciones antes de ir a lugares
                rgb_stream.stop()
                depth_stream.stop()
                
                # Llamar a la vista de selección de lugares
                lugares_seleccionados = mostrar_seleccion_lugares(
                    device, coordenadas, dmax_map, dmin_map, draw_logo_func,
                    existing_window_name=window_name
                )
                
                # Si se canceló (retornó None o "MENU"), retornar None o "MENU"
                if lugares_seleccionados is None or lugares_seleccionados == "MENU":
                    return lugares_seleccionados
                else:
                    # Retornar diccionario con acciones y lugares
                    return {
                        'acciones': acciones_seleccionadas,
                        'lugares': lugares_seleccionados
                    }
        
        # Redibujar la pantalla en cada frame
        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            temp_screen[y, :] = [b, g, r]
        draw_logo_smaller_acciones(temp_screen)
        draw_accion_cards(temp_screen, accion_positions, selected_cards=acciones_seleccionadas)
        draw_acciones_seleccionadas(temp_screen, len(acciones_seleccionadas))
        draw_back_card_acciones(temp_screen, elevated=back_elevated_acciones)
        draw_close_card_acciones(temp_screen, elevated=close_elevated_acciones)
        # Bloquear botón si no hay exactamente 1 acción seleccionada
        is_blocked = len(acciones_seleccionadas) != 1
        draw_siguiente_button_acciones(temp_screen, elevated=siguiente_elevated_acciones, blocked=is_blocked)
        acciones_screen_scaled = scale_to_videobeam(temp_screen)
        cv2.imshow(window_name, acciones_screen_scaled)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Solo cerrar si se presiona 'q' explícitamente
            rgb_stream.stop()
            depth_stream.stop()
            return None
    
    # Detener streams y cerrar (esto no debería ejecutarse normalmente)
    rgb_stream.stop()
    depth_stream.stop()
    return acciones_seleccionadas if acciones_seleccionadas else None  # Retornar las acciones seleccionadas (o None si no hay ninguna)

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
    
    # Crear un área rectangular de detección (más grande que el círculo para facilitar el toque)
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
    lugares = [
        {"nombre": "Calle", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Calle.png"},
        {"nombre": "Clinica", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Clinica.png"},
        {"nombre": "Estacion-Policia", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Estacion-Policia.png"},
        {"nombre": "Escuela", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Escuela.png"},
        {"nombre": "Casa", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Casa.png"},
        {"nombre": "Parque", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Parque.png"}
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
            
            # Efecto de elevación si está seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 5
            
            if is_selected:
                elevation_offset = -8
                scale_factor = 1.08
                shadow_offset_base = 8
            
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
            
            # Color del borde (verde si está seleccionada)
            border_color = (0, 255, 0) if is_selected else (100, 100, 100)
            border_thickness = 5 if is_selected else 2
            
            # Dibujar la card
            if img is not None:
                # Redimensionar imagen para que quepa en la card
                img_resized = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                
                # Si la imagen tiene canal alpha, compositar
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
                # Si no hay imagen, dibujar card con color
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
            
            # Dibujar borde (verde si está seleccionada)
            cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
    
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
                # Mostrar efecto de elevación en la flecha
                temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    temp_screen[y, :] = [b, g, r]
                draw_logo_func(temp_screen)
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
                draw_logo_func(temp_screen)
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
            
            # Verificar si se está tocando el botón de retroceso
            if detectar_back_card_touch_lugares(x_touch, y_touch):
                back_elevated_lugares = True
            else:
                back_elevated_lugares = False
            
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
                # Si no se está tocando el botón, resetear estado
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
                    # Toggle de selección: si ya está seleccionado, deseleccionarlo; si no, seleccionarlo
                    if lugar_seleccionado_temp in lugares_seleccionados:
                        lugares_seleccionados.remove(lugar_seleccionado_temp)
                        print(f"Lugar deseleccionado: {lugar_seleccionado_temp}")
                    else:
                        # Limitar a máximo 1 lugar
                        if len(lugares_seleccionados) < 1:
                            lugares_seleccionados.append(lugar_seleccionado_temp)
                            print(f"Lugar seleccionado: {lugar_seleccionado_temp}")
                            _historia_tts_speak(lugar_seleccionado_temp, tipo="lugar")
                        else:
                            # Si ya hay un lugar seleccionado, reemplazarlo
                            lugares_seleccionados.clear()
                            lugares_seleccionados.append(lugar_seleccionado_temp)
                            print(f"Lugar seleccionado: {lugar_seleccionado_temp} (reemplazando selección anterior)")
                            _historia_tts_speak(lugar_seleccionado_temp, tipo="lugar")
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
    historias = [
        {"nombre": "Niño", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Niño.png"},
        {"nombre": "Niña", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Niña.png"},
        {"nombre": "Doctor", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/doctor.png"},
        {"nombre": "Maestra", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Maestra.png"},
        {"nombre": "Policia", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Policia.png"},
        {"nombre": "Perro", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Perro.png"}
    ]
    
    acciones = [
        {"nombre": "Dar", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Dar.png"},
        {"nombre": "Ayudar", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Ayudar.png"},
        {"nombre": "Correr", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Correr.png"},
        {"nombre": "Jugar", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Jugar.png"},
        {"nombre": "Llamar", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Llamar.png"},
        {"nombre": "Trabajar", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Trabajar.png"}
    ]
    
    lugares = [
        {"nombre": "Calle", "color": (255, 150, 200), "imagen": "src/features/juego-historia/assets/images/Calle.png"},
        {"nombre": "Clinica", "color": (200, 150, 255), "imagen": "src/features/juego-historia/assets/images/Clinica.png"},
        {"nombre": "Estacion-Policia", "color": (150, 255, 200), "imagen": "src/features/juego-historia/assets/images/Estacion-Policia.png"},
        {"nombre": "Escuela", "color": (255, 200, 150), "imagen": "src/features/juego-historia/assets/images/Escuela.png"},
        {"nombre": "Casa", "color": (200, 255, 150), "imagen": "src/features/juego-historia/assets/images/Casa.png"},
        {"nombre": "Parque", "color": (150, 200, 255), "imagen": "src/features/juego-historia/assets/images/Parque.png"}
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
                ret = _run_historia_voice_flow(
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


def _run_historia_voice_flow(window_name, view_width, view_height, scale_to_videobeam, draw_logo_func,
                             sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados,
                             device, coordenadas, dmax_map, dmin_map,
                             draw_close_card_final_fn, detectar_close_card_touch_final_fn,
                             close_card_detection_x_final, close_card_detection_y_final,
                             close_card_detection_w_final, close_card_detection_h_final,
                             VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT):
    """
    Run listen -> transcribe -> thinking -> Ollama -> result screen. Returns "MENU" when user presses X on result.
    """
    def _gradient_screen():
        screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            screen[y, :] = [b, g, r]
        return screen

    sv = _get_historia_story_voice()
    get_required_words = sv.get_required_words
    listen_and_transcribe = sv.listen_and_transcribe
    verify_story_ollama = sv.verify_story_ollama

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "src", "features", "juego-historia", "config", "palabras_imagenes.json")
    required_words = get_required_words(config_path, sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados)

    # 1) Show "Preparando..." first; then "Ahora puedes hablar" only after listening has started
    font = cv2.FONT_HERSHEY_DUPLEX
    fs_prep, th_prep = 1.5, 3  # Mismo tamaño y grosor que "Comprobando tu historia"
    prep_screen = _gradient_screen()
    draw_logo_func(prep_screen)
    msg_prep = "Preparando micrófono..."
    
    # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
    try:
        from PIL import Image, ImageDraw
        from src.core.font_utils import get_ubuntu_font
        font_ubuntu = get_ubuntu_font(font_scale=fs_prep, bold=True)
        img_pil = Image.fromarray(cv2.cvtColor(prep_screen, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            bbox = draw.textbbox((0, 0), msg_prep, font=font_ubuntu)
            text_width_prep = bbox[2] - bbox[0]
        except AttributeError:
            bbox = font_ubuntu.getbbox(msg_prep) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
            text_width_prep = bbox[2] - bbox[0]
    except:
        text_size_prep, _ = cv2.getTextSize(msg_prep, font, fs_prep, th_prep)
        text_width_prep = text_size_prep[0]
    
    # Centrar texto horizontalmente
    tx_prep = (view_width - text_width_prep) // 2
    ty_prep = view_height // 2
    
    # Dibujar texto con sombra y más grande
    _put_text_safe_historia(prep_screen, msg_prep, (tx_prep + 2, ty_prep + 2), font, fs_prep, (0, 0, 0), th_prep + 1, bold=True)
    _put_text_safe_historia(prep_screen, msg_prep, (tx_prep, ty_prep), font, fs_prep, (255, 255, 255), th_prep, bold=True)
    cv2.imshow(window_name, scale_to_videobeam(prep_screen))
    cv2.waitKey(100)

    def _show_ahora_puedes_hablar():
        esc_screen = _gradient_screen()
        draw_logo_func(esc_screen)
        msg = "Ahora puedes hablar"
        
        # Usar mismo tamaño y estilo que "Comprobando tu historia"
        fs_esc, th_esc = 1.5, 3  # Mismo tamaño y grosor que "Comprobando tu historia"
        
        # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_ubuntu = get_ubuntu_font(font_scale=fs_esc, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(esc_screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), msg, font=font_ubuntu)
                text_width_esc = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font_ubuntu.getbbox(msg) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                text_width_esc = bbox[2] - bbox[0]
        except:
            text_size_esc, _ = cv2.getTextSize(msg, font, fs_esc, th_esc)
            text_width_esc = text_size_esc[0]
        
        # Centrar texto horizontalmente
        tx = (view_width - text_width_esc) // 2
        ty = view_height // 2
        
        # Dibujar texto con sombra y más grande
        _put_text_safe_historia(esc_screen, msg, (tx + 2, ty + 2), font, fs_esc, (0, 0, 0), th_esc + 1, bold=True)
        _put_text_safe_historia(esc_screen, msg, (tx, ty), font, fs_esc, (255, 255, 255), th_esc, bold=True)
        cv2.imshow(window_name, scale_to_videobeam(esc_screen))
        cv2.waitKey(1)

    text = listen_and_transcribe(timeout=10, phrase_time_limit=10, on_listening_started=_show_ahora_puedes_hablar)
    if not text:
        text = ""

    # 2) Transcription screen with typewriter animation
    trans_screen = _gradient_screen()
    draw_logo_func(trans_screen)
    tit = "Tu historia:"
    _put_text_safe_historia(trans_screen, tit, (50, 120), font, 0.8, (255, 255, 255), 2)
    for i in range(1, len(text) + 1):
        trans_screen = _gradient_screen()
        draw_logo_func(trans_screen)
        _put_text_safe_historia(trans_screen, tit, (50, 120), font, 0.8, (255, 255, 255), 2)
        # Word wrap: draw text[:i] in a box
        line = text[:i]
        font_scale = 0.7
        thickness = 2
        y_pos = 180
        max_width = view_width - 100
        words = line.split()
        line_cur = ""
        for w in words:
            test = line_cur + (" " if line_cur else "") + w
            (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw > max_width and line_cur:
                _put_text_safe_historia(trans_screen, line_cur, (50, y_pos), font, font_scale, (255, 255, 255), thickness)
                y_pos += 35
                line_cur = w
            else:
                line_cur = test
        if line_cur:
            _put_text_safe_historia(trans_screen, line_cur, (50, y_pos), font, font_scale, (255, 255, 255), thickness)
        cv2.imshow(window_name, scale_to_videobeam(trans_screen))
        cv2.waitKey(max(20, 400 // max(len(text), 1)))
    cv2.waitKey(800)

    # 3) Thinking indicator + Ollama
    result_holder = [None]
    done_holder = [False]
    def _ollama_thread():
        result_holder[0] = verify_story_ollama(
                text,
                subjects=sujetos_seleccionados,
                actions=acciones_seleccionadas,
                places=lugares_seleccionados,
                model="deepseek-r1",
            )
        done_holder[0] = True
    thr = threading.Thread(target=_ollama_thread, daemon=True)
    thr.start()
    start_thinking = time.time()
    while not done_holder[0]:
        elapsed = int((time.time() - start_thinking) * 1000)
        dots = "." * ((elapsed // 500) % 4)
        think_screen = _gradient_screen()
        draw_logo_func(think_screen)
        msg_think = "Comprobando tu historia" + dots
        
        # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
        font_scale_think = 1.5  # Aumentado de 1.0 a 1.5
        thickness_think = 3  # Aumentado de 2 a 3
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_ubuntu = get_ubuntu_font(font_scale=font_scale_think, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(think_screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), msg_think, font=font_ubuntu)
                text_width_think = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font_ubuntu.getbbox(msg_think) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                text_width_think = bbox[2] - bbox[0]
        except:
            text_size_think, _ = cv2.getTextSize(msg_think, font, font_scale_think, thickness_think)
            text_width_think = text_size_think[0]
        
        # Centrar texto horizontalmente
        tx_think = (view_width - text_width_think) // 2
        ty_think = view_height // 2
        
        # Dibujar texto con sombra y más grande
        _put_text_safe_historia(think_screen, msg_think, (tx_think + 2, ty_think + 2), font, font_scale_think, (0, 0, 0), thickness_think + 1, bold=True)
        _put_text_safe_historia(think_screen, msg_think, (tx_think, ty_think), font, font_scale_think, (255, 255, 255), thickness_think, bold=True)
        cv2.imshow(window_name, scale_to_videobeam(think_screen))
        cv2.waitKey(1)  # Cambiar a 1ms para que no se quede pegado
    result = result_holder[0] or {"correct": False, "tips": ["Revisa tu oración."]}

    # Initialize confetti system if story is correct
    confetti_system = None
    if result.get("correct", False):
        confetti_system = ConfettiSystem(view_width, view_height, num_particles=200)
        confetti_system.start(multiple_bursts=True, num_burst_points=3)

    # 4) Result screen: LingoBien + sentence (highlighted) + tips + X
    lingo_path = os.path.join(project_root, "images", "LingoBien.png")
    if not os.path.exists(lingo_path):
        lingo_path = "images/LingoBien.png"
    lingo_img = cv2.imread(lingo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(lingo_path) else None

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    def _draw_result_screen(screen, elevated_close=False):
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            screen[y, :] = [b, g, r]
        draw_logo_func(screen)
        # LingoBien
        if lingo_img is not None:
            h_img, w_img = lingo_img.shape[:2]
            max_h = 220
            scale = max_h / h_img
            w_new = int(w_img * scale)
            h_new = int(h_img * scale)
            resized = cv2.resize(lingo_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            lx = (view_width - w_new) // 2
            ly = 140
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    screen[ly:ly + h_new, lx:lx + w_new, c] = (
                        alpha * resized[:, :, c] + (1 - alpha) * screen[ly:ly + h_new, lx:lx + w_new, c])
            else:
                screen[ly:ly + h_new, lx:lx + w_new] = resized[:, :, :3]
        
        # Sentence with highlights (word-by-word with line wrap) and underlines for parts
        display_text = text if text else "No se pudo entender. Intenta de nuevo."
        y_pos = 400
        font_scale = 0.65
        thickness = 2
        max_width = view_width - 80
        
        # Get parts from result
        parts = result.get("parts", {})
        subjects_list = parts.get("subjects", [])
        actions_list = parts.get("actions", [])
        predicates_list = parts.get("predicates", [])
        
        # Create normalized sets for matching
        def normalize_word(word):
            """Normalize word by removing punctuation and converting to lowercase"""
            return re.sub(r"[^\wáéíóúñü]", "", word.lower())
        
        def word_matches(word_norm, word_list):
            """Check if normalized word matches any word in the list (exact or partial match)"""
            for w in word_list:
                w_norm = normalize_word(w)
                if word_norm == w_norm or word_norm.startswith(w_norm) or w_norm.startswith(word_norm):
                    return True
            return False
        
        # Colors for different parts
        subject_color = (0, 0, 255)  # Red (BGR)
        action_color = (0, 255, 0)  # Green (BGR)
        predicate_color = (255, 0, 0)  # Blue (BGR)
        
        # Calculate word positions using utility function
        word_positions_data = calculate_word_positions(display_text, 40, y_pos, font_scale, max_width, bold=False)
        
        required_set = set(required_words)
        
        # Function to match phrases (multi-word sequences) and single words in the text
        def find_phrase_positions(phrase_list, word_positions_data):
            """Find positions of phrases (which may span multiple words) or single words in the text"""
            phrase_matches = []
            for phrase in phrase_list:
                if not phrase or not phrase.strip():
                    continue
                phrase_words = phrase.split()
                
                # Handle single word
                if len(phrase_words) == 1:
                    phrase_norm = normalize_word(phrase_words[0])
                    for wp_data in word_positions_data:
                        word_norm = normalize_word(wp_data['word'])
                        if word_norm == phrase_norm or word_norm.startswith(phrase_norm) or phrase_norm.startswith(word_norm):
                            phrase_matches.append({
                                'x_start': wp_data['x_start'],
                                'x_end': wp_data['x_end'],
                                'y': wp_data['y'],
                                'phrase': phrase
                            })
                else:
                    # Handle multi-word phrase
                    phrase_norms = [normalize_word(w) for w in phrase_words]
                    
                    # Try to find the phrase in consecutive words
                    for i in range(len(word_positions_data) - len(phrase_words) + 1):
                        # Check if consecutive words match the phrase
                        match = True
                        for j, phrase_word_norm in enumerate(phrase_norms):
                            word_norm = normalize_word(word_positions_data[i + j]['word'])
                            if word_norm != phrase_word_norm and not (word_norm.startswith(phrase_word_norm) or phrase_word_norm.startswith(word_norm)):
                                match = False
                                break
                        
                        if match:
                            # Found a match - get the span
                            start_wp = word_positions_data[i]
                            end_wp = word_positions_data[i + len(phrase_words) - 1]
                            phrase_matches.append({
                                'x_start': start_wp['x_start'],
                                'x_end': end_wp['x_end'],
                                'y': start_wp['y'],  # Use first word's y position
                                'phrase': phrase
                            })
            return phrase_matches
        
        # Find phrase positions for each category
        subject_phrases = find_phrase_positions(subjects_list, word_positions_data)
        action_phrases = find_phrase_positions(actions_list, word_positions_data)
        predicate_phrases = find_phrase_positions(predicates_list, word_positions_data)
        
        # Build underline info from phrase matches
        underline_info = []
        
        # Combine all phrase matches with their categories
        all_phrases = []
        for sp in subject_phrases:
            all_phrases.append({**sp, 'is_subject': True, 'is_action': False, 'is_predicate': False})
        for ap in action_phrases:
            # Check if this phrase is already in the list (might overlap with subject/predicate)
            existing = next((p for p in all_phrases if p['x_start'] == ap['x_start'] and p['x_end'] == ap['x_end']), None)
            if existing:
                existing['is_action'] = True
            else:
                all_phrases.append({**ap, 'is_subject': False, 'is_action': True, 'is_predicate': False})
        for pp in predicate_phrases:
            existing = next((p for p in all_phrases if p['x_start'] == pp['x_start'] and p['x_end'] == pp['x_end']), None)
            if existing:
                existing['is_predicate'] = True
            else:
                all_phrases.append({**pp, 'is_subject': False, 'is_action': False, 'is_predicate': True})
        
        underline_info = all_phrases
        
        # Draw action highlights FIRST (before text, so text appears on top)
        for ui in underline_info:
            if ui['is_action']:
                x_start = ui['x_start']
                x_end = ui['x_end']
                y_text = ui['y']
                
                # Calculate text height for highlight
                try:
                    from PIL import Image, ImageDraw
                    from src.core.font_utils import get_ubuntu_font
                    font_ubuntu = get_ubuntu_font(font_scale=font_scale, bold=False)
                    img_pil = Image.new('RGB', (100, 100), (0, 0, 0))
                    draw = ImageDraw.Draw(img_pil)
                    try:
                        bbox = draw.textbbox((0, 0), "Ag", font=font_ubuntu)
                        word_height = bbox[3] - bbox[1]
                    except AttributeError:
                        bbox = font_ubuntu.getbbox("Ag") if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 20)
                        word_height = bbox[3] - bbox[1]
                except:
                    (_, word_height), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
                    word_height = word_height + baseline
                
                # Draw highlight rectangle behind the text
                highlight_y_top = y_text - word_height + 2  # Slight padding
                highlight_y_bottom = y_text + 2
                highlight_color = action_color
                # Draw semi-transparent highlight
                overlay = screen.copy()
                cv2.rectangle(overlay, (x_start, highlight_y_top), (x_end, highlight_y_bottom), highlight_color, -1)
                cv2.addWeighted(overlay, 0.3, screen, 0.7, 0, screen)
        
        # Draw text AFTER highlights
        for wp_data in word_positions_data:
            word = wp_data['word']
            norm = normalize_word(word)
            is_required = norm in required_set or any(norm.startswith(rw) or rw.startswith(norm) for rw in required_set)
            
            # Text color (white by default, or cyan if it's a required word)
            text_color = (0, 255, 255) if is_required else (255, 255, 255)
            
            # Draw text at calculated position
            _put_text_safe_historia(screen, word, (wp_data['x_start'], wp_data['y']), font, font_scale, text_color, thickness)
        
        # Draw underlines and highlights - positioned lower below the text
        # First, we need to get the actual text height to position underlines correctly
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_ubuntu = get_ubuntu_font(font_scale=font_scale, bold=False)
            img_pil = Image.new('RGB', (100, 100), (0, 0, 0))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), "Ag", font=font_ubuntu)  # Sample text to get height
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                bbox = font_ubuntu.getbbox("Ag") if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 20)
                text_height = bbox[3] - bbox[1]
        except:
            (_, text_height), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
            text_height = text_height + baseline
        
        underline_offset = 20# Increased distance from text baseline (lower position)
        underline_thickness = 3
        
        # Draw underlines for subjects and predicates (actions are already highlighted)
        for ui in underline_info:
            x_start = ui['x_start']
            x_end = ui['x_end']
            y_text = ui['y']
            
            # Calculate baseline position (y_text is the baseline position from PIL)
            y_baseline = y_text
            
            # Subjects and Predicates: Draw underlines at the same level (no stepping)
            underline_categories = []
            if ui['is_subject']:
                underline_categories.append(('subject', subject_color))
            if ui['is_predicate']:
                underline_categories.append(('predicate', predicate_color))
            
            # Draw all underlines at the same y position (no stepping)
            if underline_categories:
                y_underline = y_baseline + underline_offset
                for category, color in underline_categories:
                    cv2.line(screen, (x_start, y_underline), (x_end, y_underline), color, underline_thickness)
        
        # Draw legend in bottom right (always show all three categories)
        legend_x = view_width - 250
        legend_y = view_height - 120
        legend_font_scale = 0.5
        legend_thickness = 2
        legend_line_height = 25
        
        # Legend background (semi-transparent) - always show 3 items
        legend_bg_height = 3 * legend_line_height + 20
        legend_bg = np.zeros((legend_bg_height, 240, 3), dtype=np.uint8)
        legend_bg[:] = (40, 40, 40)  # Dark gray
        screen[legend_y-10:legend_y-10+legend_bg_height, legend_x-10:legend_x-10+240] = \
            cv2.addWeighted(screen[legend_y-10:legend_y-10+legend_bg_height, legend_x-10:legend_x-10+240], 0.7, legend_bg, 0.3, 0)
        
        legend_y_current = legend_y
        # Always show all three categories in the legend
        # Draw subject color line
        cv2.line(screen, (legend_x, legend_y_current), (legend_x + 30, legend_y_current), subject_color, underline_thickness)
        _put_text_safe_historia(screen, "Sujeto", (legend_x + 40, legend_y_current), font, legend_font_scale, (255, 255, 255), legend_thickness)
        legend_y_current += legend_line_height
        
        # Draw action color highlight (rectangle instead of line)
        highlight_rect_size = 20
        highlight_y_top = legend_y_current - highlight_rect_size // 2
        highlight_y_bottom = legend_y_current + highlight_rect_size // 2
        overlay_legend = screen.copy()
        cv2.rectangle(overlay_legend, (legend_x, highlight_y_top), (legend_x + 30, highlight_y_bottom), action_color, -1)
        cv2.addWeighted(overlay_legend, 0.3, screen, 0.7, 0, screen)
        _put_text_safe_historia(screen, "Verbo", (legend_x + 40, legend_y_current), font, legend_font_scale, (255, 255, 255), legend_thickness)
        legend_y_current += legend_line_height
        
        # Draw predicate color line
        cv2.line(screen, (legend_x, legend_y_current), (legend_x + 30, legend_y_current), predicate_color, underline_thickness)
        _put_text_safe_historia(screen, "Predicado", (legend_x + 40, legend_y_current), font, legend_font_scale, (255, 255, 255), legend_thickness)
        # Tips - mostrar como lista numerada con fuente más grande
        tips = result.get("tips") or []
        
        # Calcular posición inicial para tips
        y_tips = y_pos + 50
        tips_font_scale = 0.75  # Aumentado de 0.55 a 0.75
        tips_thickness = 3  # Aumentado de 2 a 3
        tips_max_width = view_width - 100  # Ancho máximo para el texto de tips
        
        # Dibujar el título "Consejos:" con fuente más grande
        _put_text_safe_historia(screen, "Consejos:", (40, y_tips), font, 0.8, (200, 200, 255), 3, bold=True)
        
        # Dibujar tips como lista numerada
        tips_x = 60  # Indentación para la lista
        tips_y = y_tips + 35
        tips_line_height = 35  # Espacio entre líneas (aumentado)
        
        for idx, tip in enumerate(tips, 1):
            # Crear texto con número de lista: "1. [tip text]"
            tip_text = f"{idx}. {tip}"
            
            # Dividir el tip en palabras para word wrapping
            tip_words = tip_text.split()
            current_line = ""
            
            for word in tip_words:
                test_line = current_line + (" " if current_line else "") + word
                # Calcular ancho del texto de prueba
                try:
                    from PIL import Image, ImageDraw
                    from src.core.font_utils import get_ubuntu_font
                    font_ubuntu = get_ubuntu_font(font_scale=tips_font_scale, bold=False)
                    img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    try:
                        bbox = draw.textbbox((0, 0), test_line, font=font_ubuntu)
                        test_width = bbox[2] - bbox[0]
                    except AttributeError:
                        bbox = font_ubuntu.getbbox(test_line) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                        test_width = bbox[2] - bbox[0]
                except:
                    test_size, _ = cv2.getTextSize(test_line, font, tips_font_scale, tips_thickness)
                    test_width = test_size[0]
                
                # Si el texto excede el ancho máximo, dibujar la línea actual y empezar una nueva
                if test_width > tips_max_width and current_line:
                    _put_text_safe_historia(screen, current_line, (tips_x, tips_y), font, tips_font_scale, (255, 255, 255), tips_thickness)
                    tips_y += tips_line_height
                    current_line = word
                else:
                    current_line = test_line
            
            # Dibujar la última línea del tip si hay contenido
            if current_line:
                _put_text_safe_historia(screen, current_line, (tips_x, tips_y), font, tips_font_scale, (255, 255, 255), tips_thickness)
                tips_y += tips_line_height
        draw_close_card_final_fn(screen, elevated=elevated_close)
        # Draw confetti if story is correct (will be drawn after this function returns)

    # Mostrar la vista de resultados con tips primero
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    result_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    _draw_result_screen(result_screen, elevated_close=False)
    # Don't draw confetti here - it will be drawn in the final overlay
    cv2.imshow(window_name, scale_to_videobeam(result_screen))
    cv2.waitKey(1)  # Actualizar la ventana inmediatamente para que se muestre rápido
    
    # Leer los tips por voz DESPUÉS de mostrar la pantalla de resultados
    tips = result.get("tips") or []
    tips_thread = None
    if tips:
        tips_text = ". ".join(tips)  # Unir los tips con puntos
        def _speak_tips():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                for v in engine.getProperty("voices"):
                    if "spanish" in v.name.lower() or "español" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(tips_text)
                engine.runAndWait()
            except Exception as e:
                print(f"⚠ Error al reproducir TTS de tips: {e}")
        
        # Reproducir tips en un hilo separado (no daemon para poder esperar a que termine)
        tips_thread = threading.Thread(target=_speak_tips, daemon=False)
        tips_thread.start()
    
    # Esperar a que el TTS termine y luego 2 segundos adicionales antes de mostrar el overlay
    close_elevated = False
    frame_count_res = 0
    touch_history_res = {}
    from collections import defaultdict
    touch_history_res = defaultdict(list)
    
    # Esperar a que el hilo de TTS termine (si existe)
    if tips_thread is not None:
        tips_thread.join()  # Esperar a que termine el TTS
    
    # Esperar 2 segundos adicionales después de que termine el TTS
    tips_display_time = 0
    tips_display_duration = 2.0  # Mostrar tips por 2 segundos después del TTS
    
    while tips_display_time < tips_display_duration:
        tips_display_time += 0.1
        time.sleep(0.1)
        
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            # Redibujar la pantalla de resultados
            result_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
            _draw_result_screen(result_screen, elevated_close=False)
            if confetti_system is not None:
                confetti_system.draw(result_screen)
            cv2.imshow(window_name, scale_to_videobeam(result_screen))
            cv2.waitKey(1)
            continue
        
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask = cv2.medianBlur(touch_mask, 3)
        kernel = np.ones((2, 2), np.uint8)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 <= area <= 50000:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x_touch = int(xv_min + cx * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + cy * (yv_max - yv_min) / (yw_max - yw_min))
                    if detectar_close_card_touch_final_fn(x_touch, y_touch):
                        rgb_stream.stop()
                        depth_stream.stop()
                        return "MENU"
        
        # Actualizar confetti si está activo
        if confetti_system is not None:
            confetti_system.update()
        
        # Redibujar la pantalla de resultados
        result_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        _draw_result_screen(result_screen, elevated_close=False)
        if confetti_system is not None:
            confetti_system.draw(result_screen)
        cv2.imshow(window_name, scale_to_videobeam(result_screen))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rgb_stream.stop()
            depth_stream.stop()
            return None
    
    # Ahora mostrar la nueva vista con overlay claro sobre la vista de resultados
    is_correct = result.get("correct", False)
    message_text = "Muy bien" if is_correct else "Vamos inténtalo de nuevo"
    
    def _draw_final_screen(base_screen, elevated_salir=False, elevated_reintentar=False):
        # Usar la vista de resultados como base (ya tiene la historia y consejos)
        # Trabajar directamente sobre base_screen para que los cambios se reflejen
        
        # Agregar overlay oscuro/transparente sobre la vista de resultados
        # Crear una capa oscura (negro semi-transparente) para que se vea la vista de fondo
        dark_overlay = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        alpha = 0.4  # 40% de opacidad oscura para que se vea mejor la vista de fondo
        cv2.addWeighted(base_screen, 1 - alpha, dark_overlay, alpha, 0, base_screen)
        
        # Dibujar imagen LingoBien centrada
        if lingo_img is not None:
            h_img, w_img = lingo_img.shape[:2]
            max_h = 220
            scale = max_h / h_img
            w_new = int(w_img * scale)
            h_new = int(h_img * scale)
            resized = cv2.resize(lingo_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            lx = (view_width - w_new) // 2
            ly = 140
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                alpha_img = resized[:, :, 3] / 255.0
                for c in range(3):
                    base_screen[ly:ly + h_new, lx:lx + w_new, c] = (
                        alpha_img * resized[:, :, c] + (1 - alpha_img) * base_screen[ly:ly + h_new, lx:lx + w_new, c])
            else:
                base_screen[ly:ly + h_new, lx:lx + w_new] = resized[:, :, :3]
            
            # Dibujar texto "Muy bien" o "Vamos inténtalo de nuevo" más grande y centrado abajo de la imagen
            message_y = ly + h_new + 40  # 40 píxeles abajo de la imagen
            message_font_scale = 1.5  # Más grande
            message_thickness = 3
            message_color = (0, 255, 0) if is_correct else (0, 200, 255)  # Verde si correcto, naranja si no
            
            # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
            try:
                from PIL import Image, ImageDraw
                from src.core.font_utils import get_ubuntu_font
                font_ubuntu = get_ubuntu_font(font_scale=message_font_scale, bold=True)
                img_pil = Image.fromarray(cv2.cvtColor(base_screen, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                try:
                    bbox = draw.textbbox((0, 0), message_text, font=font_ubuntu)
                    message_width = bbox[2] - bbox[0]
                except AttributeError:
                    bbox = font_ubuntu.getbbox(message_text) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                    message_width = bbox[2] - bbox[0]
            except:
                message_size, _ = cv2.getTextSize(message_text, font, message_font_scale, message_thickness)
                message_width = message_size[0]
            
            # Centrar texto horizontalmente
            message_x = (view_width - message_width) // 2
            
            # Dibujar texto con sombra
            _put_text_safe_historia(base_screen, message_text, (message_x + 2, message_y + 2), 
                                   font, message_font_scale, (0, 0, 0), message_thickness + 1, bold=True)
            _put_text_safe_historia(base_screen, message_text, (message_x, message_y), 
                                   font, message_font_scale, message_color, message_thickness, bold=True)
        
        # Dibujar botones rectangulares: "Salir" y "Volver a intentarlo"
        button_width_salir = 200
        button_width_reintentar = 280  # Más ancho para "Volver a intentarlo"
        button_height = 60
        button_spacing = 50
        button_y = view_height - 90 - button_height  # Misma posición que botones "Siguiente" y "Jugar"
        button_center_x = view_width // 2
        
        # Botón "Salir" (izquierda) - rojo
        button_salir_x = button_center_x - button_width_salir - button_spacing // 2
        button_salir_bounds = draw_rectangular_button(
            base_screen,
            button_salir_x, button_y, button_width_salir, button_height,
            "Salir",
            bg_color=(0, 0, 200),  # Rojo
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.0,  # Tamaño reducido
            bold=True,
            shadow=True
        )
        
        # Botón "Volver a intentarlo" o "Volver a jugar" (derecha) - verde
        button_reintentar_x = button_center_x + button_spacing // 2
        # Cambiar el texto del botón según si la historia es correcta o no
        button_text_reintentar = "Volver a jugar" if is_correct else "Volver a intentarlo"
        button_reintentar_bounds = draw_rectangular_button(
            base_screen,
            button_reintentar_x, button_y, button_width_reintentar, button_height,
            button_text_reintentar,
            bg_color=(100, 255, 100),  # Verde
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.0,  # Tamaño reducido
            bold=True,
            shadow=True
        )
        
        return button_salir_bounds, button_reintentar_bounds
    
    # Mostrar la nueva vista final usando la vista de resultados como base
    # Crear la vista de resultados final (con historia y consejos)
    result_screen_final = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    _draw_result_screen(result_screen_final, elevated_close=False)
    # Don't draw confetti here - it will be drawn in the final overlay
    
    # Ahora agregar el overlay oscuro con imagen Lingo, texto y botones
    button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_screen_final, elevated_salir=False, elevated_reintentar=False)
    
    # Draw confetti in the final overlay if active
    if confetti_system is not None:
        confetti_system.draw(result_screen_final)
    
    cv2.imshow(window_name, scale_to_videobeam(result_screen_final))
    
    salir_elevated = False
    reintentar_elevated = False
    frame_count_final = 0
    touch_history_final = defaultdict(list)
    
    while True:
        frame_count_final += 1
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            # Recrear la vista de resultados como base
            result_screen_base = np.zeros((view_height, view_width, 3), dtype=np.uint8)
            _draw_result_screen(result_screen_base, elevated_close=False)
            if confetti_system is not None:
                confetti_system.update()
            button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_screen_base, elevated_salir=salir_elevated, elevated_reintentar=reintentar_elevated)
            # Draw confetti in the final overlay
            if confetti_system is not None:
                confetti_system.draw(result_screen_base)
            cv2.imshow(window_name, scale_to_videobeam(result_screen_base))
            cv2.waitKey(1)
            continue
        
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask = cv2.medianBlur(touch_mask, 3)
        kernel = np.ones((2, 2), np.uint8)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Procesar toques
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 <= area <= 50000:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x_touch = int(xv_min + cx * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + cy * (yv_max - yv_min) / (yw_max - yw_min))
                    
                    # Verificar si se tocó el botón "Salir"
                    if button_salir_bounds and is_point_in_rectangular_button(x_touch, y_touch, button_salir_bounds):
                        print("Botón 'Salir' presionado - Volviendo al menú principal")
                        rgb_stream.stop()
                        depth_stream.stop()
                        return "MENU"
                    
                    # Verificar si se tocó el botón "Volver a intentarlo" o "Volver a jugar"
                    if button_reintentar_bounds and is_point_in_rectangular_button(x_touch, y_touch, button_reintentar_bounds):
                        button_text = "Volver a jugar" if is_correct else "Volver a intentarlo"
                        print(f"Botón '{button_text}' presionado")
                        rgb_stream.stop()
                        depth_stream.stop()
                        # Si la historia es correcta, volver a la primera selección del juego
                        # Si no es correcta, volver a la vista final (donde está el botón Hablar)
                        if is_correct:
                            return "RESTART_FULL"  # Volver al inicio del juego
                        else:
                            return "RESTART"  # Volver a la vista final
        
        # Redibujar la pantalla usando la vista de resultados como base
        result_screen_base = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        _draw_result_screen(result_screen_base, elevated_close=False)
        # Actualizar confetti si está activo
        if confetti_system is not None:
            confetti_system.update()
        button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_screen_base, elevated_salir=salir_elevated, elevated_reintentar=reintentar_elevated)
        # Draw confetti in the final overlay (after drawing the overlay)
        if confetti_system is not None:
            confetti_system.draw(result_screen_base)
        cv2.imshow(window_name, scale_to_videobeam(result_screen_base))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rgb_stream.stop()
            depth_stream.stop()
            return None
    return "MENU"

def mostrar_menu_juegos(device):
    # Inicializar Pygame para reproducir sonidos (opcional)
    pygame.init()
    pygame.mixer.init()

    # 1. Cargar Configuraciones
    try:
        with open("config/ultima_configuracion_coordenadas.json", "r") as file:
            coordenadas = json.load(file)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'ultima_configuracion_coordenadas.json'.")
        return
    except json.JSONDecodeError:
        print("Error: El archivo JSON tiene un formato inválido.")
        return

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Cargar y validar dmax_map (opcional - solo necesario para detección de toques)
    dmax_map, w, h = load_and_validate_dmax_map(coordenadas)
    dmax_map_available = dmax_map is not None
    
    if dmax_map_available:
        # Ajustar el dmax_map (restar offset)
        dmax_map = dmax_map - 5
        dmin_map = dmax_map - 7
    else:
        print("\n[ADVERTENCIA] No se pudo cargar dmax_map. El menú se mostrará pero la detección de toques no funcionará.")
        print("Ejecuta 'python calibrate_area.py' o 'python calibrate_area_mejorado.py' para calibrar.\n")
        dmax_map = None
        dmin_map = None

    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800
    
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

    # Función auxiliar para dibujar el logo
    def draw_logo(screen):
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
            put_text_ubuntu(screen, logo_text, (text_x + 3, text_y + 3), 
                       font_scale, (0, 0, 0), thickness + 2)
            put_text_ubuntu(screen, logo_text, (text_x, text_y), 
                       font_scale, (0, 255, 255), thickness)
            return False
    
    # Crear fondo colorido con degradado para niños
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores pastel (azul a morado a rosa)
    for y in range(view_height):
        ratio = y / view_height
        # Degradado de azul claro a morado a rosa
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        videobeam_screen[y, :] = [b, g, r]  # BGR
    
    # Dibujar el logo usando la función auxiliar
    logo_loaded = draw_logo(videobeam_screen)
    
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
    
    # Estado de la card de bocina (muteada o no)
    # Usar las variables globales del módulo niveles_clasificacion para mantener el estado del audio entre vistas
    # Cargar y configurar el audio de fondo usando mixer.music (solo si no está cargado)
    if not niveles_clasificacion._background_music_loaded:
        archivo = "relax-meditate-gentle-peaceful-291162.mp3"
        if os.path.exists(archivo):
            try:
                pygame.mixer.music.load(archivo)
                # El módulo music suele responder mejor a volúmenes bajos
                pygame.mixer.music.set_volume(0.05)  # 5% de volumen
                niveles_clasificacion._background_music_loaded = True
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3 (volumen al 5%)")
                # Iniciar el audio automáticamente si no está muteado
                if not niveles_clasificacion._bocina_muted_global:
                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
                    print("✓ Audio de fondo iniciado automáticamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el audio de fondo: {e}")
        else:
            print("⚠ No se encontró el archivo de audio: relax-meditate-gentle-peaceful-291162.mp3")
    
    # Usar el estado global del audio
    bocina_muted = niveles_clasificacion._bocina_muted_global
    
    # Función para dibujar card cuadrada con icono de bocina (estilo infantil) - amarilla con bocina
    def draw_close_card_main(screen, elevated=False, muted=False):
        """
        Dibuja una card cuadrada con icono de bocina en el centro, estilo infantil, amarilla con bocina
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
            muted: Si True, muestra la imagen de bocina muteada (BocinaMute.png), si False muestra Bocina.png
        """
        # Posición base del lado derecho (parte inferior, misma posición X que tenía la card amarilla)
        base_card_size = 100  # Tamaño del cuadrado (2 * radio de 50 para mantener el mismo tamaño)
        card_margin_x = 180  # Mismo margen X que tenía la card amarilla
        card_margin_y = 50  # Margen desde el borde inferior (misma posición Y que tenía la card amarilla)
        
        # Sin efecto de elevación - siempre el mismo tamaño y posición
        card_size = base_card_size
        # Calcular posición del cuadrado (esquina superior izquierda)
        card_x = view_width - card_margin_x - card_size
        card_y = view_height - card_margin_y - card_size
        
        # Dibujar la imagen completa como fondo de la card (usar imagen mute si está muteada)
        # NOTA: Se eliminó la sombra negra como se solicitó
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
    
    # Variables para la card de cerrar (necesarias para la detección)
    # Usar un área rectangular para la detección, similar a las otras cards
    close_card_size_main = 100  # Tamaño del cuadrado
    close_card_margin_x_main = 180
    close_card_margin_y_main = 50  # Margen desde el borde inferior
    close_card_x_main = view_width - close_card_margin_x_main - close_card_size_main
    close_card_y_main = view_height - close_card_margin_y_main - close_card_size_main  # Posición desde abajo
    
    # Crear un área rectangular de detección (más grande que el cuadrado para facilitar el toque)
    close_card_detection_size_main = int(close_card_size_main * 1.2)  # Área más grande para facilitar el toque
    close_card_detection_x_main = close_card_x_main - int(close_card_size_main * 0.1)
    close_card_detection_y_main = close_card_y_main - int(close_card_size_main * 0.1)
    close_card_detection_w_main = close_card_detection_size_main
    close_card_detection_h_main = close_card_detection_size_main
    
    # Función para detectar si se tocó la card de cerrar (usando área rectangular como las otras cards)
    def detectar_close_card_touch_main(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar (usando área rectangular)"""
        return (close_card_detection_x_main <= x_touch <= close_card_detection_x_main + close_card_detection_w_main and
                close_card_detection_y_main <= y_touch <= close_card_detection_y_main + close_card_detection_h_main)

    # 2. Definir los 3 juegos ficticios (sin acentos en los nombres)
    juegos = [
        {
            "nombre": "Juego de Clasificacion",
            "color": (100, 200, 255),  # Azul claro (BGR)
            "descripcion": "Clasifica objetos",
            "icono": "📦",
            "imagen": "images/LogoClasificacion.png"  # Ruta de la imagen del logo
        },
        {
            "nombre": "Absurdos Logicos",
            "color": (100, 255, 150),  # Verde claro (BGR)
            "descripcion": "Encuentra lo absurdo",
            "icono": "🤔",
            "imagen": "images/LogoAbsurdosVisuales.png"  # Ruta de la imagen del logo
        },
        {
            "nombre": "Historias",
            "color": (255, 150, 200),  # Rosa claro (BGR)
            "descripcion": "Crea historias",
            "icono": "📚",
            "imagen": "images/LogoHistoria.png"  # Ruta de la imagen del logo
        }
    ]
    
    # Cargar imágenes de los juegos si existen
    juego_images = {}
    for juego in juegos:
        if "imagen" in juego and os.path.exists(juego["imagen"]):
            img = cv2.imread(juego["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                juego_images[juego["nombre"]] = img
                print(f"✓ Imagen cargada para {juego['nombre']}: {juego['imagen']}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {juego['nombre']}: {juego['imagen']}")
    
    # 3. Calcular posiciones de las cards (centradas verticalmente)
    card_width = 320
    card_height = 400
    card_spacing = 50
    total_cards_width = len(juegos) * card_width + (len(juegos) - 1) * card_spacing
    start_x = (view_width - total_cards_width) // 2
    start_y = 200  # Posición vertical para las cards
    
    game_positions = {}
    for idx, juego in enumerate(juegos):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        game_positions[juego["nombre"]] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': juego["color"],
            'descripcion': juego["descripcion"],
            'icono': juego["icono"],
            'imagen': juego_images.get(juego["nombre"])  # Imagen cargada si existe
        }
    
    # 4. Función para dibujar las cards de juegos
    def draw_game_cards(screen, game_positions, elevated_card=None):
        """
        Dibuja las cards de juegos en la pantalla.
        
        Args:
            screen: La imagen donde dibujar
            game_positions: Diccionario con las posiciones de las cards
            elevated_card: Nombre de la card que debe estar elevada (None si ninguna)
        """
        for nombre, pos in game_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            color = pos['color']
            
            # Efecto de elevación si esta card está seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 8
            
            if elevated_card == nombre:
                elevation_offset = -20  # Mover hacia arriba
                scale_factor = 1.05  # Aumentar tamaño ligeramente
                shadow_offset_base = 15  # Sombra más grande para mayor profundidad
                # Hacer el color más brillante cuando está elevada
                color = tuple(min(255, int(c * 1.15)) for c in color)
            
            # Aplicar escala
            w_scaled = int(w * scale_factor)
            h_scaled = int(h * scale_factor)
            x_scaled = x - (w_scaled - w) // 2  # Centrar el escalado
            y_scaled = y + elevation_offset - (h_scaled - h) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, screen.shape[0] - h_scaled))
            
            # Dibujar sombra (más grande si está elevada)
            shadow_offset = int(shadow_offset_base * scale_factor)
            shadow_color = (50, 50, 50)
            cv2.rectangle(screen, (x_scaled + shadow_offset, y_scaled + shadow_offset), 
                        (x_scaled + w_scaled + shadow_offset, y_scaled + h_scaled + shadow_offset), 
                        shadow_color, -1)
            
            # Verificar si esta card tiene imagen para usar como fondo completo
            if 'imagen' in pos and pos['imagen'] is not None:
                # Usar la imagen como fondo completo de la card
                img = pos['imagen'].copy()
                
                # Redimensionar la imagen para que llene completamente la card
                img_resized = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                
                # Verificar límites antes de dibujar
                if x_scaled >= 0 and y_scaled >= 0 and x_scaled + w_scaled <= screen.shape[1] and y_scaled + h_scaled <= screen.shape[0]:
                    # Si la imagen tiene canal alfa (transparencia)
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        # Extraer canal alfa
                        alpha = img_resized[:, :, 3] / 255.0
                        # Convertir BGR de la imagen
                        img_bgr = img_resized[:, :, :3]
                        # Mezclar con el fondo
                        for c in range(3):
                            screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        # Sin canal alfa, copiar directamente
                        screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
                
                # No dibujar nombre ni descripción si la imagen ya los incluye
                # (Opcional: puedes comentar estas líneas si quieres que NO se muestren el nombre y descripción)
                continue  # Saltar el resto del dibujado para esta card si tiene imagen
            else:
                # Comportamiento original para cards sin imagen
                # Dibujar card con bordes redondeados (simulado con rectángulos)
                # Fondo de la card
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
                
                # Borde de la card (más oscuro, más grueso si está elevada)
                border_color = tuple(max(0, c - 30) for c in color)
                border_thickness = 7 if elevated_card == nombre else 5
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
                
                # Dibujar efecto de brillo en la parte superior (más brillante si está elevada)
                highlight_boost = 60 if elevated_card == nombre else 40
                highlight_color = tuple(min(255, c + highlight_boost) for c in color)
                highlight_margin = int(10 * scale_factor)
                cv2.rectangle(screen, (x_scaled + highlight_margin, y_scaled + highlight_margin), 
                            (x_scaled + w_scaled - highlight_margin, y_scaled + int(50 * scale_factor) + highlight_margin), 
                            highlight_color, -1)
                
                # Ajustar posiciones de texto según el escalado
                icon_x_offset = (w_scaled - w) // 2
                icon_y_offset = (h_scaled - h) // 2
                
                # Dibujar icono emoji (comportamiento original)
                icono = pos['icono']
                font_icon = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_icon = 3.0 * scale_factor  # Escalar el icono también
                thickness_icon = int(3 * scale_factor)
                icon_size, _ = cv2.getTextSize(icono, font_icon, font_scale_icon, thickness_icon)
                icon_x = x_scaled + (w_scaled - icon_size[0]) // 2
                icon_y = y_scaled + int(100 * scale_factor) + icon_y_offset
                put_text_ubuntu(screen, icono, (icon_x, icon_y), font_scale_icon, (255, 255, 255), thickness_icon)
            
            # Dibujar nombre del juego (nombre es la clave del diccionario)
            # Ajustar el texto para que quepa dentro de la card
            nombre_texto = nombre
            font_nombre = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nombre = 0.9
            thickness_nombre = 2
            
            # Calcular el ancho disponible (con margen de 20 píxeles a cada lado)
            available_width = w - 40
            
            # Verificar si el texto cabe, si no, reducir el tamaño de fuente
            nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            while nombre_size[0] > available_width and font_scale_nombre > 0.5:
                font_scale_nombre -= 0.1
                nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            
            # Si aún no cabe, dividir en múltiples líneas
            if nombre_size[0] > available_width:
                # Dividir el texto en palabras y crear líneas
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
                start_y = y_scaled + int(180 * scale_factor) + icon_y_offset - (len(lineas) - 1) * line_height // 2
                for i, linea in enumerate(lineas):
                    linea_size, _ = cv2.getTextSize(linea, font_nombre, font_scale_nombre, thickness_nombre)
                    linea_x = x_scaled + (w_scaled - linea_size[0]) // 2
                    linea_y = start_y + i * line_height
                    # Sombra del texto
                    put_text_ubuntu(screen, linea, (linea_x + 2, linea_y + 2), 
                               font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                    # Texto principal
                    put_text_ubuntu(screen, linea, (linea_x, linea_y), 
                               font_scale_nombre, (255, 255, 255), thickness_nombre)
            else:
                # El texto cabe en una línea, dibujarlo normalmente
                nombre_x = x_scaled + (w_scaled - nombre_size[0]) // 2
                nombre_y = y_scaled + int(180 * scale_factor) + icon_y_offset
                # Sombra del texto
                put_text_ubuntu(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                           font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                # Texto principal
                put_text_ubuntu(screen, nombre_texto, (nombre_x, nombre_y), 
                           font_scale_nombre, (255, 255, 255), thickness_nombre)
            
            # Dibujar descripción (ajustar si es muy larga)
            desc_texto = pos['descripcion']
            font_desc = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_desc = 0.7 * scale_factor
            thickness_desc = int(2 * scale_factor)
            
            # Calcular el ancho disponible (con margen de 20 píxeles a cada lado)
            available_width_desc = int(w_scaled - 40)
            
            # Verificar si el texto cabe, si no, reducir el tamaño de fuente
            desc_size, _ = cv2.getTextSize(desc_texto, font_desc, font_scale_desc, thickness_desc)
            while desc_size[0] > available_width_desc and font_scale_desc > 0.4:
                font_scale_desc -= 0.05
                desc_size, _ = cv2.getTextSize(desc_texto, font_desc, font_scale_desc, thickness_desc)
            
            desc_x = x_scaled + (w_scaled - desc_size[0]) // 2
            desc_y = y_scaled + int(220 * scale_factor) + icon_y_offset
            put_text_ubuntu(screen, desc_texto, (desc_x, desc_y), 
                       font_scale_desc, (255, 255, 255), thickness_desc)
    
    # 5. Crear y configurar la ventana ANTES de dibujar (igual que calibrate_area.py)
    # Crear una ventana para la proyección
    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Menú de Juegos", 1920, 0)
    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Dibujar las cards en la pantalla
    draw_game_cards(videobeam_screen, game_positions)
    # Dibujar card de bocina en la parte inferior derecha
    draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)

    # Escalar a la resolución del videobeam antes de mostrar
    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
    cv2.imshow("Menú de Juegos", videobeam_screen_scaled)

    # 6. Función para detectar el juego seleccionado
    def detectar_juego_seleccionado(x_touch, y_touch, game_positions):
        """
        Dado un punto de toque (x_touch, y_touch), determina qué juego ha sido seleccionado.
        """
        for nombre, pos in game_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            if x <= x_touch <= x + w and y <= y_touch <= y + h:
                return nombre
        return None
    
    # 7. La función de selección de niveles ahora está en vista_niveles_clasificacion.py
    
    # 7b. Funciones placeholder para otros juegos
    def mostrar_mensaje_juego(nombre_juego):
        """Muestra un mensaje cuando se selecciona un juego (placeholder)"""
        mensaje_screen = videobeam_screen.copy()
        
        # Fondo semi-transparente
        overlay = mensaje_screen.copy()
        cv2.rectangle(overlay, (0, 0), (view_width, view_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, mensaje_screen, 0.3, 0, mensaje_screen)
        
        # Mensaje principal
        texto_principal = f"¡{nombre_juego}!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.0
        thickness = 4
        text_size, _ = cv2.getTextSize(texto_principal, font, font_scale, thickness)
        text_x = (view_width - text_size[0]) // 2
        text_y = view_height // 2 - 50
        
        # Sombra del texto
        put_text_ubuntu(mensaje_screen, texto_principal, (text_x + 3, text_y + 3), 
                   font_scale, (0, 0, 0), thickness + 2)
        # Texto principal
        put_text_ubuntu(mensaje_screen, texto_principal, (text_x, text_y), 
                   font_scale, (0, 255, 255), thickness)
        
        # Mensaje secundario
        texto_secundario = "Este juego estará disponible pronto"
        font_sec = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_sec = 1.0
        thickness_sec = 2
        text_size_sec, _ = cv2.getTextSize(texto_secundario, font_sec, font_scale_sec, thickness_sec)
        text_x_sec = (view_width - text_size_sec[0]) // 2
        text_y_sec = view_height // 2 + 50
        
        put_text_ubuntu(mensaje_screen, texto_secundario, (text_x_sec, text_y_sec), 
                   font_scale_sec, (255, 255, 255), thickness_sec)
        
        # Escalar a la resolución del videobeam antes de mostrar
        mensaje_screen_scaled = scale_to_videobeam(mensaje_screen)
        cv2.imshow("Menú de Juegos", mensaje_screen_scaled)
        cv2.waitKey(50)
        cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        cv2.waitKey(2000)  # Mostrar por 2 segundos

    # 8. Iniciar streams de cámara para detección de toques (solo si dmax_map está disponible)
    if dmax_map_available:
        rgb_stream = device.create_color_stream()
        depth_stream = device.create_depth_stream()
        rgb_stream.start()
        depth_stream.start()

        juego_seleccionado_flag = False  # Bandera para evitar múltiples selecciones
        frame_count = 0
        initialization_delay = 10  # Reducido a 10 frames para respuesta más rápida
        
        # Sistema de debounce temporal para evitar falsos positivos
        from collections import defaultdict
        touch_history = defaultdict(list)  # Historial de toques por posición
        min_touch_frames = 2  # Requiere que el toque persista por al menos 2 frames consecutivos
        touch_persistence_threshold = 0.8  # 80% de los frames deben tener el toque
        min_touch_area = 100  # Área mínima para filtrar ruido pequeño pero permitir toques reales
        max_touch_area = 50000  # Área máxima para evitar detecciones de objetos grandes
        last_valid_touch_time = time.time()
        touch_cooldown = 0.15  # Cooldown de 150ms entre toques válidos
        history_cleanup_interval = 30  # Limpiar historial cada 30 frames
        max_history_age = 1.0  # Eliminar entradas del historial más antiguas de 1 segundo
        inactivity_threshold = 5.0  # Si no hay toques válidos en 5 segundos, ser más estricto
        touch_feedback_points = []  # Lista (x, y, timestamp) para feedback visual sutil de toques
        touch_feedback_duration = 0.4  # Segundos que se muestra el feedback de cada toque

        # 9. Bucle principal de detección de toques
        while True:
            juego_seleccionado_flag = False
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

            # Crear la máscara que considera solo los valores entre dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

            # Aplicar filtros más suaves para preservar toques reales (igual que calibrate_area.py)
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
            
            # Aplicar apertura morfológica para eliminar ruido pequeño
            kernel = np.ones((2, 2), np.uint8)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
            
            # También aplicar cierre para conectar áreas cercanas
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)

            # No procesar toques durante el delay inicial
            if frame_count < initialization_delay:
                draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # Encontrar los contornos de los toques (usar la máscara filtrada)
            contours, _ = cv2.findContours(touch_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Limpiar historial antiguo periódicamente
            current_time = time.time()
            if frame_count % history_cleanup_interval == 0:
                # Eliminar entradas del historial más antiguas de max_history_age segundos
                for key in list(touch_history.keys()):
                    touch_history[key] = [
                        touch for touch in touch_history[key] 
                        if current_time - touch[3] < max_history_age
                    ]
                    # Si el historial está vacío, eliminarlo
                    if not touch_history[key]:
                        del touch_history[key]
            
            # Limitar el tamaño del historial por posición
            for key in list(touch_history.keys()):
                if len(touch_history[key]) > min_touch_frames + 5:
                    touch_history[key] = touch_history[key][-(min_touch_frames + 5):]

            # Procesar cada contorno
            valid_touches_this_frame = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Validar que el área esté en el rango esperado
                if min_touch_area <= area <= max_touch_area:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas de ventana a viewport
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Agregar a historial (usar coordenadas discretizadas para agrupar toques cercanos)
                        touch_key = (x_touch // 25, y_touch // 25)  # Agrupar toques dentro de 25 píxeles
                        touch_history[touch_key].append((x_touch, y_touch, area, current_time))
                        
                        # Verificar si el toque ha persistido lo suficiente
                        if len(touch_history[touch_key]) >= min_touch_frames:
                            # Verificar cooldown
                            if current_time - last_valid_touch_time > touch_cooldown:
                                # Determinar si ha habido inactividad reciente
                                time_since_last_touch = current_time - last_valid_touch_time
                                is_inactive = time_since_last_touch > inactivity_threshold
                                
                                # Si hay inactividad, ser más estricto (requerir más frames y área más grande)
                                required_frames = min_touch_frames + (2 if is_inactive else 0)
                                required_min_area = int(min_touch_area * (1.5 if is_inactive else 1.0))
                                
                                # Obtener los últimos required_frames toques (o todos si hay menos)
                                available_touches = len(touch_history[touch_key])
                                if available_touches >= required_frames:
                                    recent_touches = touch_history[touch_key][-required_frames:]
                                    
                                    # Validar que todos los toques sean recientes
                                    time_window = 0.8 if is_inactive else 1.0
                                    all_recent = all(current_time - touch[3] < time_window for touch in recent_touches)
                                    
                                    # Validar que el área sea razonablemente consistente
                                    if all_recent and len(recent_touches) >= required_frames:
                                        areas = [touch[2] for touch in recent_touches]
                                        avg_area = sum(areas) / len(areas)
                                        
                                        # Validación: área promedio debe estar en rango válido
                                        # Si hay inactividad, ser más estricto con la variación
                                        if min(areas) > 0:
                                            area_variance = max(areas) / min(areas)
                                            max_variance = 2.5 if is_inactive else 3.5
                                            if area_variance < max_variance and required_min_area <= avg_area <= max_touch_area:
                                                valid_touches_this_frame.append((x_touch, y_touch, touch_key))
                                        elif required_min_area <= avg_area <= max_touch_area:
                                            # Si no hay variación suficiente, solo permitir si está en rango y no hay inactividad
                                            if not is_inactive:
                                                valid_touches_this_frame.append((x_touch, y_touch, touch_key))
            
            # Procesar solo los toques válidos (que han persistido lo suficiente)
            for x_touch, y_touch, touch_key in valid_touches_this_frame:
                # Limpiar el historial de este toque después de procesarlo
                if touch_key in touch_history:
                    del touch_history[touch_key]
                last_valid_touch_time = current_time
                # Añadir a la lista de feedback visual (posición + tiempo)
                touch_feedback_points.append((x_touch, y_touch, current_time))

                # Primero verificar si se tocó la card de bocina
                if detectar_close_card_touch_main(x_touch, y_touch):
                    print("Card de bocina tocada en menú principal")
                    
                    # Cambiar el estado de mute (usando variable global del módulo niveles_clasificacion)
                    niveles_clasificacion._bocina_muted_global = not niveles_clasificacion._bocina_muted_global
                    bocina_muted = niveles_clasificacion._bocina_muted_global
                    print(f"Bocina {'muteada' if bocina_muted else 'activada'}")
                    
                    # Controlar el audio según el estado
                    if niveles_clasificacion._background_music_loaded:
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
                    
                    # Redibujar la pantalla con el nuevo estado (sin efecto de elevación)
                    temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                    for y in range(view_height):
                        ratio = y / view_height
                        r = int(255 * (0.3 + 0.4 * ratio))
                        g = int(200 * (0.5 + 0.3 * ratio))
                        b = int(255 * (0.8 - 0.3 * ratio))
                        temp_screen[y, :] = [b, g, r]
                    draw_logo(temp_screen)
                    draw_game_cards(temp_screen, game_positions)
                    draw_close_card_main(temp_screen, elevated=False, muted=bocina_muted)
                    videobeam_screen = temp_screen
                    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                    cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                    continue
                
                # Detectar si se seleccionó un juego
                if not juego_seleccionado_flag:
                    juego_seleccionado = detectar_juego_seleccionado(x_touch, y_touch, game_positions)
                    if juego_seleccionado:
                        print(f"Juego seleccionado: {juego_seleccionado}")
                        
                        # Mostrar efecto de elevación de la card (animación rápida)
                        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                        for y in range(view_height):
                            ratio = y / view_height
                            r = int(255 * (0.3 + 0.4 * ratio))
                            g = int(200 * (0.5 + 0.3 * ratio))
                            b = int(255 * (0.8 - 0.3 * ratio))
                            temp_screen[y, :] = [b, g, r]
                        draw_logo(temp_screen)
                        draw_game_cards(temp_screen, game_positions, elevated_card=juego_seleccionado)
                        draw_close_card_main(temp_screen, elevated=False, muted=bocina_muted)
                        temp_screen_scaled = scale_to_videobeam(temp_screen)
                        cv2.imshow("Menú de Juegos", temp_screen_scaled)
                        cv2.waitKey(100)  # Pausa breve antes de cambiar
                        
                        # Si es el juego de Clasificación, mostrar selección de niveles
                        if juego_seleccionado == "Juego de Clasificacion":
                            # No cerrar la ventana, reutilizarla para transición suave
                            nivel_seleccionado = mostrar_seleccion_niveles_clasificacion(
                                device, coordenadas, dmax_map, dmin_map, draw_logo,
                                existing_window_name="Menú de Juegos"
                            )
                            if nivel_seleccionado:
                                print(f"Procesando nivel seleccionado: {nivel_seleccionado}")
                                # Aquí puedes agregar la lógica para iniciar el juego con el nivel seleccionado
                                # Por ejemplo: juego_clasificacion(device, nivel=nivel_seleccionado, ...)
                            
                            # Volver al menú principal después de seleccionar nivel o cancelar (incluyendo cuando se presiona X)
                            # Recrear el menú
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break
                        # Si es Absurdos Logicos, ir directamente al juego de absurdos visuales
                        elif juego_seleccionado == "Absurdos Logicos":
                            # Ir directamente al juego de absurdos visuales (sin vista intermedia)
                            juego_absurdos_reconocimiento_voz(device, coordenadas, dmax_map, dmin_map, sentence_transformer_model)
                            # El juego retornó, volver al menú principal
                            # Recrear el menú
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True  # Establecer flag para salir del bucle de detección
                            break  # Salir del bucle de detección de toques para mostrar el menú principal
                        # Si es Historias, mostrar selección de historias
                        elif juego_seleccionado == "Historias":
                            # Mostrar la vista de selección de historias (sujetos)
                            resultado_seleccion = mostrar_seleccion_historias(
                                device, coordenadas, dmax_map, dmin_map, draw_logo,
                                existing_window_name="Menú de Juegos"
                            )
                            if resultado_seleccion:
                                if isinstance(resultado_seleccion, dict):
                                    # Si retornó un diccionario, puede contener sujetos, acciones y lugares
                                    sujetos_seleccionados = resultado_seleccion.get('sujetos', [])
                                    acciones_seleccionadas = resultado_seleccion.get('acciones', [])
                                    lugares_seleccionados = resultado_seleccion.get('lugares', [])
                                    print(f"Sujetos seleccionados: {sujetos_seleccionados}")
                                    print(f"Acciones seleccionadas: {acciones_seleccionadas}")
                                    print(f"Lugares seleccionados: {lugares_seleccionados}")
                                    # Aquí puedes agregar la lógica para iniciar el juego de historia
                                    # Por ejemplo: juego_historia(device, sujetos=sujetos_seleccionados, acciones=acciones_seleccionadas, lugares=lugares_seleccionados, ...)
                                elif isinstance(resultado_seleccion, list):
                                    # Si retornó una lista, solo se seleccionaron sujetos (caso antiguo, por compatibilidad)
                                    print(f"Sujetos seleccionados: {resultado_seleccion}")
                                    # Aquí puedes agregar la lógica para iniciar el juego de historia
                                    # Por ejemplo: juego_historia(device, sujetos=resultado_seleccion, ...)
                            
                            # Volver al menú principal después de seleccionar historia o cancelar
                            # Recrear el menú
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break
                        else:
                            mostrar_mensaje_juego(juego_seleccionado)
                            
                            # Redibujar el menú después del mensaje
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            # Recrear el fondo degradado
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            # Redibujar logo y cards
                            draw_logo(videobeam_screen)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break

            # Asegurar que la card de bocina esté dibujada en cada frame
            bocina_muted = niveles_clasificacion._bocina_muted_global

            # Copia para dibujar el feedback sutil de toques sin modificar el menú base
            display_screen = videobeam_screen.copy()
            # Mantener solo toques recientes y dibujar un círculo sutil en cada uno
            touch_feedback_points[:] = [(x, y, t) for (x, y, t) in touch_feedback_points if current_time - t < touch_feedback_duration]
            for (x, y, ts) in touch_feedback_points:
                age = current_time - ts
                alpha = 1.0 - (age / touch_feedback_duration)  # Fade out
                radius = int(8 + 6 * (1 - age / touch_feedback_duration))  # 14 -> 8 px
                color = (int(200 + 55 * alpha), int(220 + 35 * alpha), 255)  # Azul claro suave
                cv2.circle(display_screen, (int(x), int(y)), radius, color, 2)
                cv2.circle(display_screen, (int(x), int(y)), max(2, radius - 4), color, -1)

            draw_close_card_main(display_screen, elevated=False, muted=bocina_muted)
            # Escalar a la resolución del videobeam antes de mostrar
            videobeam_screen_scaled = scale_to_videobeam(display_screen)
            # Mostrar la ventana
            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
            # Forzar pantalla completa en cada frame
            cv2.waitKey(10)
            try:
                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
            except:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Detener los streams de la cámara y cerrar las ventanas
        rgb_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()
    else:
        # Si no hay dmax_map, solo mostrar el menú sin detección de toques
        print("\n[INFO] El menú se mostrará pero la detección de toques no estará disponible.")
        print("       Ejecuta la calibración para habilitar la detección de toques.\n")
        
        # Mostrar el menú estático
        videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
        cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
        # Forzar pantalla completa
        cv2.waitKey(50)
        cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        
        # Esperar hasta que se presione 'q'
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    # Función auxiliar (no usada pero mantenida para compatibilidad)
    def seleccionar_opciones_clasificacion():
        """
        Muestra opciones para seleccionar el modo de clasificación y el tipo de piezas.
        Si se selecciona "Virtuales", permite ajustar el número de fichas.
        """
        # Crear una pantalla negra para las opciones
        opciones_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

        # Definir las opciones
        opciones_modo = ["Figuras", "Colores"]
        opciones_tipo = ["Virtuales", "Físicas"]

        # Configuración de botones
        button_width = 200   # Ancho de los botones
        button_height = 80   # Alto de los botones
        padding_x = 50
        padding_y = 50
        spacing_x = 50
        spacing_y = 30

        # Calcular el ancho y alto disponibles dentro del área de trabajo
        available_width = xv_max - xv_min - 2 * padding_x
        available_height = yv_max - yv_min - 2 * padding_y

        # Calcular posición X inicial para centrar las columnas
        total_buttons_width = button_width * 2 + spacing_x
        x_start = xv_min + (available_width - total_buttons_width) // 2 + padding_x

        # Posición inicial en Y para los botones de modo y tipo
        initial_y = yv_min + padding_y + 50  # Añadimos 50 para espacio del encabezado

        # Posiciones de los botones de modo y tipo
        positions = {}
        for idx in range(max(len(opciones_modo), len(opciones_tipo))):
            # Columna 1: Opciones de modo
            if idx < len(opciones_modo):
                opcion = opciones_modo[idx]
                x = x_start
                y = initial_y + idx * (button_height + spacing_y)
                positions[opcion] = (x, y)
            # Columna 2: Opciones de tipo
            if idx < len(opciones_tipo):
                opcion = opciones_tipo[idx]
                x = x_start + button_width + spacing_x
                y = initial_y + idx * (button_height + spacing_y)
                positions[opcion] = (x, y)

        # Inicializamos num_piezas pero no mostramos los controles hasta que se seleccione "Virtuales"
        num_piezas = 5  # Valor inicial
        num_piezas_min = 5
        num_piezas_max = 12

        # Bandera para indicar si se debe mostrar la selección de número de piezas
        mostrar_num_piezas = False

        # Dibujar los elementos en la pantalla
        def draw_elements(screen):
            # Limpiar pantalla
            screen[:] = 0

            fuente_encabezado = cv2.FONT_HERSHEY_SIMPLEX
            escala_fuente_encabezado = 1.0
            color_encabezado = (255, 255, 255)
            grosor_encabezado = 2

            # Dibujar encabezados para los botones de modo y tipo
            # Encabezado de la columna de modo
            encabezado_modo = "Modo de Juego"
            tamaño_texto, _ = cv2.getTextSize(encabezado_modo, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
            x_text_modo = x_start + (button_width - tamaño_texto[0]) // 2
            y_text_modo = initial_y - 20  # Ajustar para que quede encima
            put_text_ubuntu(screen, encabezado_modo, (x_text_modo, y_text_modo), escala_fuente_encabezado, color_encabezado, grosor_encabezado)

            # Encabezado de la columna de tipo
            encabezado_tipo = "Tipo de Piezas"
            tamaño_texto, _ = cv2.getTextSize(encabezado_tipo, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
            x_text_tipo = x_start + button_width + spacing_x + (button_width - tamaño_texto[0]) // 2
            y_text_tipo = initial_y - 20  # Ajustar para que quede encima
            put_text_ubuntu(screen, encabezado_tipo, (x_text_tipo, y_text_tipo), escala_fuente_encabezado, color_encabezado, grosor_encabezado)

            # Dibujar los botones de modo y tipo
            for opcion, (x, y) in positions.items():
                # Dibujar rectángulo del botón
                cv2.rectangle(screen, (x, y), (x + button_width, y + button_height), (255, 255, 255), 2)
                # Escribir el texto de la opción
                texto = opcion
                fuente = cv2.FONT_HERSHEY_SIMPLEX
                escala_fuente = 0.8
                color_texto = (255, 255, 255)  # Blanco
                grosor_texto = 2
                tamaño_texto, _ = cv2.getTextSize(texto, fuente, escala_fuente, grosor_texto)
                text_x = x + (button_width - tamaño_texto[0]) // 2
                text_y = y + (button_height + tamaño_texto[1]) // 2
                put_text_ubuntu(screen, texto, (text_x, text_y), escala_fuente, color_texto, grosor_texto)

            # Si se ha seleccionado "Virtuales", mostramos la selección del número de fichas
            if mostrar_num_piezas:
                # Posiciones para la selección del número de fichas (debajo de los botones)
                num_piezas_area = {
                    'x': x_start,
                    'y': initial_y + 2 * (button_height + spacing_y) + 50,  # Ajustar posición debajo de los botones
                    'width': button_width * 2 + spacing_x,
                    'height': button_height
                }

                # Áreas para los botones de aumentar y disminuir
                decrease_button = {
                    'x1': num_piezas_area['x'],
                    'y1': num_piezas_area['y'],
                    'x2': num_piezas_area['x'] + 80,
                    'y2': num_piezas_area['y'] + button_height
                }

                increase_button = {
                    'x1': num_piezas_area['x'] + num_piezas_area['width'] - 80,
                    'y1': num_piezas_area['y'],
                    'x2': num_piezas_area['x'] + num_piezas_area['width'],
                    'y2': num_piezas_area['y'] + button_height
                }

                # Área donde se muestra el número de piezas actual
                num_display_area = {
                    'x': decrease_button['x2'] + 10,
                    'y': num_piezas_area['y'],
                    'width': num_piezas_area['width'] - 2 * (80 + 10),
                    'height': button_height
                }

                # Dibujar título para la selección de número de fichas
                titulo_piezas = "Numero de Fichas"
                tamaño_texto, _ = cv2.getTextSize(titulo_piezas, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
                x_text_piezas = num_piezas_area['x'] + (num_piezas_area['width'] - tamaño_texto[0]) // 2
                y_text_piezas = num_piezas_area['y'] - 20  # Ajustar para que quede encima
                put_text_ubuntu(screen, titulo_piezas, (x_text_piezas, y_text_piezas), escala_fuente_encabezado, color_encabezado, grosor_encabezado)

                # Dibujar botón de disminuir
                cv2.rectangle(screen, (decrease_button['x1'], decrease_button['y1']),
                            (decrease_button['x2'], decrease_button['y2']), (255, 255, 255), 2)
                put_text_ubuntu(screen, "-", (decrease_button['x1'] + 25, decrease_button['y1'] + 55), 2.0, (255, 255, 255), 2)

                # Dibujar botón de aumentar
                cv2.rectangle(screen, (increase_button['x1'], increase_button['y1']),
                            (increase_button['x2'], increase_button['y2']), (255, 255, 255), 2)
                put_text_ubuntu(screen, "+", (increase_button['x1'] + 20, increase_button['y1'] + 55), 2.0, (255, 255, 255), 2)

                # Dibujar área de visualización del número de piezas
                cv2.rectangle(screen, (num_display_area['x'], num_display_area['y']),
                            (num_display_area['x'] + num_display_area['width'], num_display_area['y'] + num_display_area['height']), (255, 255, 255), 2)
                # Mostrar el número actual
                texto_num = str(num_piezas)
                tamaño_texto, _ = cv2.getTextSize(texto_num, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
                text_x = num_display_area['x'] + (num_display_area['width'] - tamaño_texto[0]) // 2
                text_y = num_display_area['y'] + (num_display_area['height'] + tamaño_texto[1]) // 2
                put_text_ubuntu(screen, texto_num, (text_x, text_y), escala_fuente_encabezado, color_encabezado, grosor_encabezado)

                # Agregar las áreas de los botones de aumentar y disminuir al diccionario de áreas
                areas_opciones["decrease"] = decrease_button
                areas_opciones["increase"] = increase_button

        # Inicialmente dibujamos los elementos
        areas_opciones = {}  # Definir el diccionario aquí para que esté accesible dentro de draw_elements
        draw_elements(opciones_screen)

        # Mostrar la pantalla en la proyección
        cv2.namedWindow("Opciones de Clasificación", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Opciones de Clasificación", 1920, 0)
        cv2.setWindowProperty("Opciones de Clasificación", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Opciones de Clasificación", opciones_screen)

        # Crear áreas de selección para los botones
        for opcion, (x, y) in positions.items():
            areas_opciones[opcion] = {
                'x1': x,
                'y1': y,
                'x2': x + button_width,
                'y2': y + button_height
            }

        # Iniciar los streams de cámara
        rgb_stream = device.create_color_stream()
        depth_stream = device.create_depth_stream()
        rgb_stream.start()
        depth_stream.start()

        modo_seleccionado = None
        tipo_seleccionado = None
        seleccion_realizada = False

        # Variables para controlar un solo incremento/decremento por toque
        increase_button_pressed = False
        decrease_button_pressed = False

        # Definir la función 'detectar_opcion_seleccionada' dentro de 'seleccionar_opciones_clasificacion'
        def detectar_opcion_seleccionada(x_touch, y_touch, areas_opciones):
            """
            Dado un punto de toque (x_touch, y_touch), determina qué opción ha sido seleccionada.
            """
            for opcion, area in areas_opciones.items():
                if area['x1'] <= x_touch <= area['x2'] and area['y1'] <= y_touch <= area['y2']:
                    return opcion
            return None

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

            # Crear la máscara (filtros más estrictos para reducir toques fantasma)
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
            touch_mask = cv2.medianBlur(touch_mask, ksize=5)
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            increase_button_still_pressed = False
            decrease_button_still_pressed = False

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Ajusta el umbral según sea necesario
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"]) + yw_min

                        # Mapeo de coordenadas de ventana a viewport
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))

                        # Detectar si se seleccionó una opción
                        opcion_seleccionada = detectar_opcion_seleccionada(x_touch, y_touch, areas_opciones)
                        if opcion_seleccionada:
                            print(f"Opción seleccionada: {opcion_seleccionada}")
                            # Determinar si es modo o tipo
                            if opcion_seleccionada in opciones_modo and modo_seleccionado is None:
                                modo_seleccionado = opcion_seleccionada
                                _historia_tts_speak(opcion_seleccionada)
                                # Marcar visualmente la selección
                                cv2.rectangle(opciones_screen,
                                            (areas_opciones[modo_seleccionado]['x1'], areas_opciones[modo_seleccionado]['y1']),
                                            (areas_opciones[modo_seleccionado]['x2'], areas_opciones[modo_seleccionado]['y2']),
                                            (0, 255, 0), 3)
                                cv2.imshow("Opciones de Clasificación", opciones_screen)
                            elif opcion_seleccionada in opciones_tipo and tipo_seleccionado is None:
                                tipo_seleccionado = opcion_seleccionada
                                _historia_tts_speak(opcion_seleccionada)
                                # Marcar visualmente la selección
                                cv2.rectangle(opciones_screen,
                                            (areas_opciones[tipo_seleccionado]['x1'], areas_opciones[tipo_seleccionado]['y1']),
                                            (areas_opciones[tipo_seleccionado]['x2'], areas_opciones[tipo_seleccionado]['y2']),
                                            (0, 255, 0), 3)
                                cv2.imshow("Opciones de Clasificación", opciones_screen)
                                if tipo_seleccionado == "Virtuales":
                                    mostrar_num_piezas = True
                                    # Recalcular las áreas de los botones de incremento y decremento
                                    areas_opciones.clear()
                                    for opcion, (x, y) in positions.items():
                                        areas_opciones[opcion] = {
                                            'x1': x,
                                            'y1': y,
                                            'x2': x + button_width,
                                            'y2': y + button_height
                                        }
                                    draw_elements(opciones_screen)
                                    cv2.imshow("Opciones de Clasificación", opciones_screen)
                                else:
                                    mostrar_num_piezas = False
                                    areas_opciones = {}
                                    for opcion, (x, y) in positions.items():
                                        areas_opciones[opcion] = {
                                            'x1': x,
                                            'y1': y,
                                            'x2': x + button_width,
                                            'y2': y + button_height
                                        }
                            elif mostrar_num_piezas:
                                if opcion_seleccionada == "increase":
                                    increase_button_still_pressed = True
                                    if not increase_button_pressed:
                                        if num_piezas < num_piezas_max:
                                            num_piezas += 1
                                            print(f"Número de fichas incrementado a {num_piezas}")
                                            # Redibujar los elementos para actualizar visualmente los cambios
                                            draw_elements(opciones_screen)
                                            cv2.imshow("Opciones de Clasificación", opciones_screen)
                                        increase_button_pressed = True  # Marcar que el botón está presionado
                                elif opcion_seleccionada == "decrease":
                                    decrease_button_still_pressed = True
                                    if not decrease_button_pressed:
                                        if num_piezas > num_piezas_min:
                                            num_piezas -= 1
                                            print(f"Número de fichas decrementado a {num_piezas}")
                                            # Redibujar los elementos para actualizar visualmente los cambios
                                            draw_elements(opciones_screen)
                                            cv2.imshow("Opciones de Clasificación", opciones_screen)
                                        decrease_button_pressed = True  # Marcar que el botón está presionado

                            if modo_seleccionado and tipo_seleccionado:
                                if tipo_seleccionado == "Físicas":
                                    seleccion_realizada = True
                                    break  # Salir del bucle de detección de toques
                                elif tipo_seleccionado == "Virtuales" and mostrar_num_piezas:
                                    # Esperar a que el usuario termine de ajustar el número de piezas y luego toque en algún lugar para continuar
                                    if not (opcion_seleccionada == "increase" or opcion_seleccionada == "decrease"):
                                        seleccion_realizada = True
                                        break

            # Actualizar el estado de los botones presionados
            increase_button_pressed = increase_button_still_pressed
            decrease_button_pressed = decrease_button_still_pressed

            # Mostrar la ventana de opciones
            cv2.imshow("Opciones de Clasificación", opciones_screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Detener los streams de la cámara y cerrar las ventanas
        rgb_stream.stop()
        depth_stream.stop()
        cv2.destroyWindow("Opciones de Clasificación")

        if modo_seleccionado and tipo_seleccionado:
            # Convertir las opciones seleccionadas a los parámetros esperados por juego_clasificacion
            modo_clasificacion = modo_seleccionado.lower()  # "figuras" o "colores"
            tipo_pieza = tipo_seleccionado.lower()  # "virtuales" o "físicas"
            if tipo_pieza == "virtuales":
                print(f"Iniciando juego de clasificación con modo: {modo_clasificacion}, tipo: {tipo_pieza}, número de fichas: {num_piezas}")
                tipo_pieza = False
                juego_clasificacion(device, modo_clasificacion, tipo_pieza, num_piezas)
            else:
                print(f"Iniciando juego de clasificación con modo: {modo_clasificacion}, tipo: {tipo_pieza}")
                tipo_pieza = True
                juego_clasificacion(device, modo_clasificacion, tipo_pieza, 0)
        else:
            print("No se seleccionaron todas las opciones necesarias. Volviendo al menú principal.")

if __name__ == "__main__":
    # Inicializar OpenNI2 - Buscar en múltiples ubicaciones comunes
    openni2_paths = []
    
    # 1. Primero verificar si hay una variable de entorno configurada
    env_path = os.environ.get("OPENNI2_PATH")
    if env_path:
        openni2_paths.append(env_path)
        openni2_paths.append(os.path.join(env_path, "Redist"))
    
    # 2. Agregar la ruta más común primero (prioridad)
    openni2_paths.extend([
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
    ])
    
    # 3. Agregar ubicaciones comunes en Windows usando variables de entorno
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    
    if program_files:
        openni2_paths.append(os.path.join(program_files, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files, "OpenNI2"))
        openni2_paths.append(os.path.join(program_files, "OpenNI2", "Driver"))
    
    if program_files_x86:
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2"))
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2", "Driver"))
    
    # 4. Agregar otras rutas absolutas comunes
    openni2_paths.extend([
        "C:/Program Files (x86)/OpenNI2/Redist",
        "C:/Program Files (x86)/OpenNI2",
        "C:/OpenNI2/Redist",
        "C:/OpenNI2",
        "D:/Program Files/OpenNI2/Redist",
        "D:/Program Files/OpenNI2",
        "D:/Program Files (x86)/OpenNI2/Redist",
        "D:/Program Files (x86)/OpenNI2",
        "D:/OpenNI2/Redist",
        "D:/OpenNI2",
    ])
    
    # Eliminar duplicados y rutas vacías
    openni2_paths = list(dict.fromkeys([p for p in openni2_paths if p]))
    
    openni2_initialized = False
    last_exception = None
    
    for path in openni2_paths:
        if os.path.exists(path):
            try:
                openni2.initialize(path)
                openni2_initialized = True
                print(f"✓ OpenNI2 inicializado desde: {path}")
                break
            except Exception as e:
                last_exception = e
                continue
    
    if not openni2_initialized:
        print("=" * 60)
        print("ERROR: No se pudo encontrar OpenNI2 SDK")
        print("=" * 60)
        print("\nSOLUCIONES:")
        print("\n1. Instala OpenNI2 SDK desde:")
        print("   https://structure.io/openni")
        print("   Descarga: OpenNI 2 SDK for Windows")
        print("\n2. O configura la variable de entorno OPENNI2_PATH:")
        print("   setx OPENNI2_PATH \"C:\\ruta\\a\\OpenNI2\\Redist\"")
        print("\n3. O especifica la ruta manualmente editando main.py")
        print("   (agrega tu ruta al inicio de la lista openni2_paths)")
        print("\nUbicaciones buscadas:")
        for path in openni2_paths[:10]:  # Mostrar solo las primeras 10
            status = "✓ Existe" if os.path.exists(path) else "✗ No existe"
            print(f"  {status}: {path}")
        if len(openni2_paths) > 10:
            print(f"  ... y {len(openni2_paths) - 10} ubicaciones más")
        if last_exception:
            print(f"\nÚltimo error: {last_exception}")
        print("=" * 60)
        exit(1)
    
    device = openni2.Device.open_any()

    # Cargar modelo de sentence-transformers al inicio (para el juego de absurdos)
    print("\n" + "=" * 60)
    print("CARGANDO MODELO DE SENTENCE-TRANSFORMERS")
    print("=" * 60)
    try:
        from sentence_transformers import SentenceTransformer
        print("Cargando modelo 'paraphrase-multilingual-MiniLM-L12-v2'...")
        print("(Esto puede tardar unos minutos la primera vez que se descarga)")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                sentence_transformer_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ Modelo de sentence-transformers cargado correctamente")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"✗ Intento {attempt + 1} fallido: {e}")
                    print(f"  Reintentando en {wait_time} segundos...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Error al cargar modelo después de {max_retries} intentos: {e}")
                    print("  El juego de absurdos puede no funcionar correctamente.")
                    print("  Verifica tu conexión a internet e intenta nuevamente.")
                    sentence_transformer_model = None
    except ImportError:
        print("✗ sentence-transformers no está instalado.")
        print("  Instálalo con: uv pip install sentence-transformers")
        sentence_transformer_model = None
    except Exception as e:
        print(f"✗ Error inesperado al cargar modelo: {e}")
        sentence_transformer_model = None
    
    print("=" * 60 + "\n")

    # Mostrar el menú de juegos automáticamente al ejecutar
    print("=" * 60)
    print("Iniciando MagicboARd - Menú de Juegos")
    print("=" * 60)
    print("\nPresiona 'q' en la ventana del menú para salir")
    print("Toca las cards para seleccionar un juego\n")
    
    try:
        mostrar_menu_juegos(device)
    except KeyboardInterrupt:
        print("\n\nInterrupción del usuario. Cerrando...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        openni2.unload()
        cv2.destroyAllWindows()
        print("\n¡Hasta luego!")
