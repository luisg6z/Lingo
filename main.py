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
import threading
import niveles_clasificacion
from niveles_clasificacion import mostrar_seleccion_niveles_clasificacion
from absurdos_seleccion import mostrar_seleccion_absurdos

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
        y_viewport = int(yv_min + (y_touch - yw_min) * sy)

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
                    cv2.putText(roi, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
            
            # Aplicar un umbral para consolidar las áreas de toque
            _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

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

                # Verificar si hay toque dentro de la figura
                touch_in_shape = cv2.bitwise_and(touch_mask_final, mask_shape)

                if np.any(touch_in_shape > 0):
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
        x_viewport = int(xv_min + ((cX_window - xw_min) * sx ))
        y_viewport = int(yv_min + ((cY_window - yw_min) * sy))

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
                cv2.putText(bgr_data, "Presiona 'c' para asignar animales", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
                                    adjusted_figure = figure_to_draw + np.array([xw_min, yw_min])
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

        # Crear la máscara que considera solo los valores entre dmin y dmax
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))

        # Umbralización
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

        # Encontrar los contornos de los toques
        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Procesar cada punto de los contornos
        touch_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Ajusta el umbral según sea necesario
                for point in contour:
                    cx, cy = point[0]
                    # Transformar el punto al espacio del viewport
                    x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                    touch_points.append([x_touch, y_touch])

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
                y_viewport = int(yv_min + (centro_y * sy) - 30)  # Aplicar offset de 40

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
                cv2.putText(deteccion_visual, f"{figura_tipo}, {color_name_lower}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        y_viewport = int(yv_min + (y_touch - yw_min) * sy)
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
            cv2.putText(videobeam_screen, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        # Crear la máscara de toques
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)

        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))

        # Aplicar umbral para eliminar manchas residuales y consolidar las áreas de toque
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

        # Detectar los toques y mapearlos
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
            cv2.putText(screen, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        # Crear la máscara que considera solo los valores entre dmin y dmax
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

        # Filtrar y procesar la máscara de toques
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))

        # Umbralización
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

        # Encontrar los contornos de los toques
        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Procesar cada punto de los contornos
        touch_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    touch_points.append((cx, cy))

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
                cv2.putText(videobeam_screen, f"Dibujo guardado como {sanitized_child_name}_{sanitized_group}.png", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        y_viewport = int(yv_min + (y_touch - yw_min) * sy)
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
        cv2.putText(videobeam_screen, "X", (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
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

    while True:
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            continue

        # Extraer los datos de las imágenes
        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        bgr_data = cv2.flip(bgr_data, 1)
        bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        # Crear la máscara que considera solo los valores entre dmin y dmax
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask_filtered = cv2.GaussianBlur(touch_mask_filtered, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask_filtered, ddepth=-1, ksize=(3, 3))

        # Umbralización
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 150, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)

        # Encontrar los contornos de los toques
        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        touch_points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                touch_points.append((cx, cy))

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
    # Cargar y configurar el audio de fondo (solo si no está cargado)
    if niveles_clasificacion._background_music_global is None:
        if os.path.exists("relax-meditate-gentle-peaceful-291162.mp3"):
            try:
                niveles_clasificacion._background_music_global = pygame.mixer.Sound("relax-meditate-gentle-peaceful-291162.mp3")
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3")
                # Iniciar el audio automáticamente si no está muteado
                if not niveles_clasificacion._bocina_muted_global:
                    niveles_clasificacion._background_music_global.play(-1)  # -1 significa bucle infinito
                    print("✓ Audio de fondo iniciado automáticamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el audio de fondo: {e}")
        else:
            print("⚠ No se encontró el archivo de audio: relax-meditate-gentle-peaceful-291162.mp3")
    
    # Usar el estado global del audio
    bocina_muted = niveles_clasificacion._bocina_muted_global
    background_music = niveles_clasificacion._background_music_global
    
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
                cv2.putText(screen, icono, (icon_x, icon_y), font_icon, font_scale_icon, (255, 255, 255), thickness_icon)
            
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
                    cv2.putText(screen, linea, (linea_x + 2, linea_y + 2), 
                               font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                    # Texto principal
                    cv2.putText(screen, linea, (linea_x, linea_y), 
                               font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            else:
                # El texto cabe en una línea, dibujarlo normalmente
                nombre_x = x_scaled + (w_scaled - nombre_size[0]) // 2
                nombre_y = y_scaled + int(180 * scale_factor) + icon_y_offset
                # Sombra del texto
                cv2.putText(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                           font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                # Texto principal
                cv2.putText(screen, nombre_texto, (nombre_x, nombre_y), 
                           font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            
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
            cv2.putText(screen, desc_texto, (desc_x, desc_y), 
                       font_desc, font_scale_desc, (255, 255, 255), thickness_desc)
    
    # Dibujar las cards en la pantalla
    draw_game_cards(videobeam_screen, game_positions)
    # Dibujar card de bocina en la parte inferior derecha
    draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)

    # 5. Mostrar la Ventana en la Proyección del Videobeam
    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Menú de Juegos", 1920, 0)
    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Menú de Juegos", videobeam_screen)

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
        cv2.putText(mensaje_screen, texto_principal, (text_x + 3, text_y + 3), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        # Texto principal
        cv2.putText(mensaje_screen, texto_principal, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Mensaje secundario
        texto_secundario = "Este juego estará disponible pronto"
        font_sec = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_sec = 1.0
        thickness_sec = 2
        text_size_sec, _ = cv2.getTextSize(texto_secundario, font_sec, font_scale_sec, thickness_sec)
        text_x_sec = (view_width - text_size_sec[0]) // 2
        text_y_sec = view_height // 2 + 50
        
        cv2.putText(mensaje_screen, texto_secundario, (text_x_sec, text_y_sec), 
                   font_sec, font_scale_sec, (255, 255, 255), thickness_sec)
        
        cv2.imshow("Menú de Juegos", mensaje_screen)
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
                cv2.imshow("Menú de Juegos", videobeam_screen)
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

                # Primero verificar si se tocó la card de bocina
                if detectar_close_card_touch_main(x_touch, y_touch):
                    print("Card de bocina tocada en menú principal")
                    
                    # Cambiar el estado de mute (usando variable global del módulo niveles_clasificacion)
                    niveles_clasificacion._bocina_muted_global = not niveles_clasificacion._bocina_muted_global
                    bocina_muted = niveles_clasificacion._bocina_muted_global
                    print(f"Bocina {'muteada' if bocina_muted else 'activada'}")
                    
                    # Controlar el audio según el estado
                    if niveles_clasificacion._background_music_global is not None:
                        if bocina_muted:
                            # Detener el audio cuando está muteada
                            pygame.mixer.stop()
                            print("Audio de fondo detenido")
                        else:
                            # Reproducir el audio en bucle cuando está activada
                            niveles_clasificacion._background_music_global.play(-1)  # -1 significa bucle infinito
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
                    cv2.imshow("Menú de Juegos", videobeam_screen)
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
                        cv2.imshow("Menú de Juegos", temp_screen)
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
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.imshow("Menú de Juegos", videobeam_screen)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break
                        # Si es Absurdos Logicos, mostrar selección de tipo de absurdos
                        elif juego_seleccionado == "Absurdos Logicos":
                            # No cerrar la ventana, reutilizarla para transición suave
                            tipo_seleccionado = mostrar_seleccion_absurdos(
                                device, coordenadas, dmax_map, dmin_map, draw_logo,
                                existing_window_name="Menú de Juegos"
                            )
                            if tipo_seleccionado:
                                print(f"Tipo de absurdo seleccionado: {tipo_seleccionado}")
                                # Aquí puedes agregar la lógica para iniciar el juego con el tipo seleccionado
                                # Por ejemplo: juego_absurdos(device, tipo=tipo_seleccionado, ...)
                            # Volver al menú principal después de seleccionar tipo o cancelar
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
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.imshow("Menú de Juegos", videobeam_screen)
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
                            cv2.imshow("Menú de Juegos", videobeam_screen)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break

            # Asegurar que la card de bocina esté dibujada en cada frame
            # Actualizar el estado de la bocina desde la variable global (por si cambió en otra vista)
            bocina_muted = niveles_clasificacion._bocina_muted_global
            
            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
            # Mostrar la ventana
            cv2.imshow("Menú de Juegos", videobeam_screen)

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
        cv2.imshow("Menú de Juegos", videobeam_screen)
        
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
            cv2.putText(screen, encabezado_modo, (x_text_modo, y_text_modo), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

            # Encabezado de la columna de tipo
            encabezado_tipo = "Tipo de Piezas"
            tamaño_texto, _ = cv2.getTextSize(encabezado_tipo, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
            x_text_tipo = x_start + button_width + spacing_x + (button_width - tamaño_texto[0]) // 2
            y_text_tipo = initial_y - 20  # Ajustar para que quede encima
            cv2.putText(screen, encabezado_tipo, (x_text_tipo, y_text_tipo), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

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
                cv2.putText(screen, texto, (text_x, text_y), fuente, escala_fuente, color_texto, grosor_texto)

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
                cv2.putText(screen, titulo_piezas, (x_text_piezas, y_text_piezas), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

                # Dibujar botón de disminuir
                cv2.rectangle(screen, (decrease_button['x1'], decrease_button['y1']),
                            (decrease_button['x2'], decrease_button['y2']), (255, 255, 255), 2)
                cv2.putText(screen, "-", (decrease_button['x1'] + 25, decrease_button['y1'] + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

                # Dibujar botón de aumentar
                cv2.rectangle(screen, (increase_button['x1'], increase_button['y1']),
                            (increase_button['x2'], increase_button['y2']), (255, 255, 255), 2)
                cv2.putText(screen, "+", (increase_button['x1'] + 20, increase_button['y1'] + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

                # Dibujar área de visualización del número de piezas
                cv2.rectangle(screen, (num_display_area['x'], num_display_area['y']),
                            (num_display_area['x'] + num_display_area['width'], num_display_area['y'] + num_display_area['height']), (255, 255, 255), 2)
                # Mostrar el número actual
                texto_num = str(num_piezas)
                tamaño_texto, _ = cv2.getTextSize(texto_num, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
                text_x = num_display_area['x'] + (num_display_area['width'] - tamaño_texto[0]) // 2
                text_y = num_display_area['y'] + (num_display_area['height'] + tamaño_texto[1]) // 2
                cv2.putText(screen, texto_num, (text_x, text_y), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

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

            # Crear la máscara que considera solo los valores entre dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

            # Operaciones morfológicas
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)

            # Encontrar los contornos de los toques
            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Reiniciar las banderas al inicio de cada iteración
            increase_button_still_pressed = False
            decrease_button_still_pressed = False

            # Procesar cada contorno
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
                                # Marcar visualmente la selección
                                cv2.rectangle(opciones_screen,
                                            (areas_opciones[modo_seleccionado]['x1'], areas_opciones[modo_seleccionado]['y1']),
                                            (areas_opciones[modo_seleccionado]['x2'], areas_opciones[modo_seleccionado]['y2']),
                                            (0, 255, 0), 3)
                                cv2.imshow("Opciones de Clasificación", opciones_screen)
                            elif opcion_seleccionada in opciones_tipo and tipo_seleccionado is None:
                                tipo_seleccionado = opcion_seleccionada
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
