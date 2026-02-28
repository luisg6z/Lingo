"""
Classification game feature for MagicboARd
"""
import cv2
import numpy as np
import pygame
import time
import os
import json
import math
import random
from openni import openni2

# Import core utilities
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sys
sys.path.insert(0, project_root)
from src.core.calibration import load_and_validate_dmax_map
from src.core.detection import detect_color_and_shape
from src.core.font_utils import put_text_ubuntu

def juego_clasificacion(device, modo_clasificacion, piezas_fisicas, num_piezas):
    # Cargar las coordenadas desde el archivo JSON
    import json
    with open(os.path.join(project_root, "src", "config", "ultima_configuracion_coordenadas.json"), "r") as file:
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