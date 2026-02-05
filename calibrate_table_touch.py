# -*- coding: utf-8 -*-
"""
Sistema de Calibración para Realidad Aumentada Espacial
Calibra el área de la mesa para detectar toques y mapearlos a la proyección

Este script:
1. Proyecta cuadrados de calibración en la mesa
2. Detecta los cuadrados con la cámara Kinect
3. Calcula la transformación de perspectiva (homografía)
4. Calibra el mapa de profundidad para detectar toques
5. Guarda la configuración para usar en el juego principal
"""

import cv2
import numpy as np
from tqdm import tqdm
from openni import openni2
import json
import os
import sys
import io
from datetime import datetime

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Dimensiones de la proyección (ajustar según tu proyector)
PROJECTION_WIDTH = 1280
PROJECTION_HEIGHT = 800

# Tamaño de los cuadrados de calibración
SQUARE_SIZE = 60

# Márgenes desde los bordes de la proyección
MARGIN = 120

# Parámetros de calibración de profundidad
DEPTH_CALIBRATION_FRAMES = 500
MIN_DEPTH = 500
MAX_DEPTH = 4000

# Parámetros de detección de toques
TOUCH_DEPTH_OFFSET = 4  # Offset desde dmax para detectar toques
TOUCH_DEPTH_RANGE = 10  # Rango de profundidad para detectar toques

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def mostrar_mensaje(proyeccion, texto, color=(255, 255, 255), font_scale=1.5, thickness=3):
    """Muestra un mensaje centrado en la proyección"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(texto, font, font_scale, thickness)[0]
    text_x = (PROJECTION_WIDTH - text_size[0]) // 2
    text_y = (PROJECTION_HEIGHT + text_size[1]) // 2
    cv2.putText(proyeccion, texto, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def dibujar_cuadricula(proyeccion, xv_min, yv_min, xv_max, yv_max):
    """Dibuja una cuadrícula en el área calibrada para visualización"""
    color = (100, 100, 100)
    thickness = 1
    
    # Líneas verticales
    for i in range(5):
        x = xv_min + (xv_max - xv_min) * i // 4
        cv2.line(proyeccion, (x, yv_min), (x, yv_max), color, thickness)
    
    # Líneas horizontales
    for i in range(5):
        y = yv_min + (yv_max - yv_min) * i // 4
        cv2.line(proyeccion, (xv_min, y), (xv_max, y), color, thickness)

# ============================================================================
# DETECCIÓN DE CUADRADOS
# ============================================================================

def detectar_cuadrados(frame):
    """
    Detecta cuadrados blancos en el frame de la cámara.
    Retorna una lista de contornos de cuadrados detectados.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro gaussiano para reducir ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Umbral alto para detectar solo áreas muy blancas (los cuadrados proyectados)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar la imagen
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuadrados = []
    
    # Parámetros de validación
    min_area = 1000
    max_area = 20000
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if min_area <= area <= max_area:
            # Aproximar el contorno a polígono
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Debe tener 4 vértices (cuadrado/rectángulo)
            if len(approx) == 4:
                # Calcular relación de aspecto
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Los cuadrados deben tener relación de aspecto cercana a 1
                if 0.7 <= aspect_ratio <= 1.3:
                    # Calcular extensión (área del contorno / área del rectángulo)
                    rect_area = w * h
                    extent = float(area) / rect_area if rect_area > 0 else 0
                    
                    # Los cuadrados bien formados tienen extensión > 0.7
                    if extent > 0.7:
                        cuadrados.append((approx, area))
    
    # Ordenar por área (de menor a mayor)
    cuadrados = sorted(cuadrados, key=lambda x: x[1])
    
    return [cuadrado[0] for cuadrado in cuadrados[:4]]  # Máximo 4 cuadrados

# ============================================================================
# PROYECCIÓN DE CUADRADOS
# ============================================================================

def proyectar_cuadrados_calibracion():
    """
    Proyecta 4 cuadrados blancos en las esquinas del área de calibración.
    Retorna la proyección y las coordenadas de los cuadrados.
    """
    proyeccion = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
    
    # Calcular posiciones de los 4 cuadrados (esquinas)
    cuadrados = []
    
    # Esquina superior izquierda
    x1 = MARGIN
    y1 = MARGIN
    cuadrados.append((x1, y1, x1 + SQUARE_SIZE, y1 + SQUARE_SIZE, (x1 + SQUARE_SIZE//2, y1 + SQUARE_SIZE//2)))
    
    # Esquina superior derecha
    x2 = PROJECTION_WIDTH - MARGIN - SQUARE_SIZE
    y2 = MARGIN
    cuadrados.append((x2, y2, x2 + SQUARE_SIZE, y2 + SQUARE_SIZE, (x2 + SQUARE_SIZE//2, y2 + SQUARE_SIZE//2)))
    
    # Esquina inferior izquierda
    x3 = MARGIN
    y3 = PROJECTION_HEIGHT - MARGIN - SQUARE_SIZE
    cuadrados.append((x3, y3, x3 + SQUARE_SIZE, y3 + SQUARE_SIZE, (x3 + SQUARE_SIZE//2, y3 + SQUARE_SIZE//2)))
    
    # Esquina inferior derecha
    x4 = PROJECTION_WIDTH - MARGIN - SQUARE_SIZE
    y4 = PROJECTION_HEIGHT - MARGIN - SQUARE_SIZE
    cuadrados.append((x4, y4, x4 + SQUARE_SIZE, y4 + SQUARE_SIZE, (x4 + SQUARE_SIZE//2, y4 + SQUARE_SIZE//2)))
    
    # Dibujar los cuadrados
    for x1, y1, x2, y2, centro in cuadrados:
        cv2.rectangle(proyeccion, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # Calcular área de calibración (rectángulo que contiene los 4 cuadrados)
    xv_min = MARGIN
    xv_max = PROJECTION_WIDTH - MARGIN
    yv_min = MARGIN
    yv_max = PROJECTION_HEIGHT - MARGIN
    
    return proyeccion, cuadrados, (xv_min, yv_min, xv_max, yv_max)

# ============================================================================
# CALIBRACIÓN DE PROFUNDIDAD
# ============================================================================

def calcular_mapa_profundidad(device, calibrated_area, xv_min, yv_min, xv_max, yv_max):
    """
    Calcula el mapa de profundidad (dmax_map) para el área calibrada.
    Este mapa representa la profundidad de la superficie de la mesa.
    """
    depth_stream = device.create_depth_stream()
    depth_stream.start()
    
    x, y, w, h = calibrated_area
    print(f"\nCalibrando profundidad en área: {w}x{h} píxeles")
    
    # Acumulador de profundidad
    depth_accum = np.zeros((h, w, MAX_DEPTH - MIN_DEPTH + 1), dtype=int)
    
    # Mostrar mensaje en proyección
    proyeccion = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibrando profundidad...", (0, 255, 255))
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1)
    
    # Acumular profundidad durante varios frames
    print("Capturando frames de profundidad...")
    for _ in tqdm(range(DEPTH_CALIBRATION_FRAMES), desc="Frames", unit="frames"):
        frame = depth_stream.read_frame()
        depth_data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[y:y+h, x:x+w]
        
        # Contar frecuencias de profundidad
        valid_mask = (depth_roi >= MIN_DEPTH) & (depth_roi <= MAX_DEPTH)
        valid_depth = depth_roi[valid_mask] - MIN_DEPTH
        indices = np.where(valid_mask)
        depth_accum[indices[0], indices[1], valid_depth] += 1
    
    depth_stream.stop()
    
    # Generar mapa dmax (moda de la profundidad)
    dmax_map = np.argmax(depth_accum, axis=2) + MIN_DEPTH
    
    # Guardar mapa
    os.makedirs("config", exist_ok=True)
    np.savetxt("config/dmax_map.txt", dmax_map.flatten(), fmt="%d")
    
    # Mostrar mensaje de finalización
    proyeccion = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibracion Completada!", (0, 255, 0))
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(2000)
    
    return dmax_map

# ============================================================================
# CALIBRACIÓN PRINCIPAL
# ============================================================================

def calibrar_mesa(device):
    """
    Función principal de calibración.
    Calibra el área de la mesa y calcula las transformaciones necesarias.
    """
    # Crear ventana de proyección
    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Proyeccion", 1920, 0)  # Mover a segunda pantalla si existe
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Verificar sensores
    if not device.has_sensor(openni2.SENSOR_COLOR):
        print("=" * 60)
        print("ERROR: El dispositivo no tiene sensor de color disponible")
        print("=" * 60)
        return None
    
    # Iniciar streams
    print("Iniciando streams de Kinect...")
    try:
        color_stream = device.create_color_stream()
        color_stream.start()
        print("✓ Stream de color iniciado")
    except Exception as e:
        print(f"ERROR: No se pudo iniciar stream de color: {e}")
        return None
    
    try:
        depth_stream = device.create_depth_stream()
        depth_stream.start()
        print("✓ Stream de profundidad iniciado")
    except Exception as e:
        print(f"ERROR: No se pudo iniciar stream de profundidad: {e}")
        color_stream.stop()
        return None
    
    # Leer primer frame de profundidad
    depth_frame = depth_stream.read_frame()
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_data = cv2.flip(depth_data, 1)
    
    # Variables de calibración
    homography_matrix = None
    calibracion_completada = False
    xw_min = xw_max = yw_min = yw_max = None
    xv_min = xv_max = yv_min = yv_max = None
    
    # ========================================================================
    # PASO 1: DETECCIÓN DE CUADRADOS
    # ========================================================================
    print("\n" + "=" * 60)
    print("PASO 1: Detección de cuadrados de calibración")
    print("=" * 60)
    print("Asegúrate de que los cuadrados blancos sean visibles en la cámara")
    print("Presiona 'q' para salir, cualquier otra tecla cuando veas los cuadrados")
    
    proyeccion, cuadrados_proy, (xv_min, yv_min, xv_max, yv_max) = proyectar_cuadrados_calibracion()
    
    # Dibujar área de calibración
    cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
    dibujar_cuadricula(proyeccion, xv_min, yv_min, xv_max, yv_max)
    
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(2000)
    
    while not calibracion_completada:
        # Leer frame de color
        frame = color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 1)
        
        # Detectar cuadrados
        cuadrados_detectados = detectar_cuadrados(frame_bgr)
        
        # Dibujar cuadrados detectados
        frame_visual = frame_bgr.copy()
        puntos_camara = []
        
        for cuadrado in cuadrados_detectados:
            M = cv2.moments(cuadrado)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                puntos_camara.append([cx, cy])
                cv2.drawContours(frame_visual, [cuadrado], -1, (0, 255, 0), 2)
                cv2.circle(frame_visual, (cx, cy), 5, (0, 0, 255), -1)
        
        # Si detectamos al menos 2 cuadrados, podemos calibrar
        if len(puntos_camara) >= 2:
            # Ordenar puntos (superior izquierda, superior derecha, inferior izquierda, inferior derecha)
            puntos_ordenados = sorted(puntos_camara, key=lambda p: (p[1], p[0]))
            
            if len(puntos_ordenados) >= 2:
                # Calcular área de trabajo en la cámara
                x_coords = [p[0] for p in puntos_ordenados]
                y_coords = [p[1] for p in puntos_ordenados]
                
                xw_min = max(0, min(x_coords) - 20)
                xw_max = min(640, max(x_coords) + 20)
                yw_min = max(0, min(y_coords) - 20)
                yw_max = min(480, max(y_coords) + 20)
                
                # Aplicar escalado para mejorar el área de detección
                factor_escala_x = 1.15
                factor_escala_y = 1.12
                
                xw_centro = (xw_min + xw_max) // 2
                yw_centro = (yw_min + yw_max) // 2
                
                xw_min_escalado = max(0, xw_min - 10)
                xw_max_escalado = min(640, int(xw_centro + (xw_max - xw_centro) * factor_escala_x))
                yw_min_escalado = max(0, int(yw_centro - (yw_centro - yw_min) * factor_escala_y))
                
                # Actualizar variables
                xw_min = xw_min_escalado
                
                # Calcular homografía (transformación de perspectiva)
                # Puntos en la cámara
                pts_camara = np.array([
                    [xw_min, yw_min_escalado],
                    [xw_max_escalado, yw_min_escalado],
                    [xw_max_escalado, yw_max],
                    [xw_min, yw_max]
                ], dtype=np.float32)
                
                # Puntos correspondientes en la proyección
                pts_proyeccion = np.array([
                    [xv_min, yv_min],
                    [xv_max, yv_min],
                    [xv_max, yv_max],
                    [xv_min, yv_max]
                ], dtype=np.float32)
                
                # Calcular matriz de homografía
                homography_matrix = cv2.getPerspectiveTransform(pts_camara, pts_proyeccion)
                
                # Dibujar área calibrada en la proyección
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 3)
                mostrar_mensaje(proyeccion, "Calibracion Exitosa!", (0, 255, 0), font_scale=1.2)
                
                # Dibujar área en la cámara
                cv2.rectangle(frame_visual, (xw_min, yw_min_escalado), 
                            (xw_max_escalado, yw_max), (255, 0, 0), 2)
                
                calibracion_completada = True
                print(f"\n✓ Calibración completada!")
                print(f"  Área cámara: ({xw_min}, {yw_min_escalado}) a ({xw_max_escalado}, {yw_max})")
                print(f"  Área proyección: ({xv_min}, {yv_min}) a ({xv_max}, {yv_max})")
                
                cv2.imshow("Proyeccion", proyeccion)
                cv2.imshow("Camara", frame_visual)
                cv2.waitKey(3000)
                break
        
        cv2.imshow("Camara", frame_visual)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Calibración cancelada por el usuario")
            color_stream.stop()
            depth_stream.stop()
            cv2.destroyAllWindows()
            return None
    
    if not calibracion_completada:
        print("ERROR: No se pudo completar la calibración")
        color_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()
        return None
    
    # ========================================================================
    # PASO 2: CALIBRACIÓN DE PROFUNDIDAD
    # ========================================================================
    print("\n" + "=" * 60)
    print("PASO 2: Calibración de profundidad")
    print("=" * 60)
    print("Mantén la mesa libre de objetos durante la calibración...")
    
    calibrated_area = (xw_min, yw_min_escalado, xw_max_escalado - xw_min, yw_max - yw_min_escalado)
    dmax_map = calcular_mapa_profundidad(device, calibrated_area, xv_min, yv_min, xv_max, yv_max)
    
    # Guardar configuración inmediatamente después de calcular dmax_map
    # Esto evita desincronización si se cancela la fase interactiva de prueba de toques
    print("\nGuardando configuración de calibración...")
    config = {
        "xv_min": int(xv_min),
        "xv_max": int(xv_max),
        "yv_min": int(yv_min),
        "yv_max": int(yv_max),
        "xw_min": int(xw_min),
        "xw_max": int(xw_max_escalado),
        "yw_min": int(yw_min_escalado),
        "yw_max": int(yw_max),
        "homography_matrix": homography_matrix.tolist() if homography_matrix is not None else None,
        "projection_width": PROJECTION_WIDTH,
        "projection_height": PROJECTION_HEIGHT,
        "touch_depth_offset": TOUCH_DEPTH_OFFSET,
        "touch_depth_range": TOUCH_DEPTH_RANGE,
        "calibration_date": datetime.now().isoformat()
    }
    
    os.makedirs("config", exist_ok=True)
    config_path = "config/ultima_configuracion_coordenadas.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuración guardada en: {config_path}")

    # ========================================================================
    # PASO 3: PRUEBA DE TOQUES (VERIFICACIÓN)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PASO 3: Prueba de toques - Verificación de calibración")
    print("=" * 60)
    print("Toca la mesa para verificar que la calibración es correcta")
    print("Presiona 'r' para recalibrar, 's' para guardar y continuar, 'q' para salir")
    
    # Preparar mapas de profundidad para detección de toques
    dmax_map = dmax_map.astype(np.int16) - TOUCH_DEPTH_OFFSET
    dmin_map = dmax_map.astype(np.int16) - TOUCH_DEPTH_RANGE
    
    # Asegurar dimensiones correctas
    roi_height = yw_max - yw_min_escalado
    roi_width = xw_max_escalado - xw_min
    if dmax_map.shape != (roi_height, roi_width):
        print(f"[INFO] Redimensionando mapas de profundidad...")
        dmax_map = cv2.resize(dmax_map.astype(np.float32), (roi_width, roi_height), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int16)
        dmin_map = cv2.resize(dmin_map.astype(np.float32), (roi_width, roi_height), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int16)
    
    # Reiniciar streams para prueba
    try:
        depth_stream.start()
    except:
        pass
    
    # Crear proyección para prueba
    proyeccion_prueba = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(proyeccion_prueba, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 3)
    dibujar_cuadricula(proyeccion_prueba, xv_min, yv_min, xv_max, yv_max)
    mostrar_mensaje(proyeccion_prueba, "Prueba de Toques - Toca la mesa", (255, 255, 0), font_scale=1.0)
    cv2.imshow("Proyeccion", proyeccion_prueba)
    cv2.waitKey(1000)
    
    # Variables para detección de toques
    previous_roi = None
    touch_history = []
    vibration_threshold = 15
    touch_duration_threshold = 3
    max_history_frames = 8
    
    calibracion_aceptada = False
    
    while not calibracion_aceptada:
        # Leer frame de profundidad
        depth_frame = depth_stream.read_frame()
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        
        # Leer frame de color para visualización
        try:
            color_frame = color_stream.read_frame()
            frame_data = color_frame.get_buffer_as_uint8()
            frame_array = np.ndarray((color_frame.height, color_frame.width, 3), 
                                   dtype=np.uint8, buffer=frame_data)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.flip(frame_bgr, 1)
        except:
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Limpiar área calibrada en proyección (mantener borde y cuadrícula)
        proyeccion_prueba = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(proyeccion_prueba, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 3)
        dibujar_cuadricula(proyeccion_prueba, xv_min, yv_min, xv_max, yv_max)
        
        # Extraer ROI de profundidad
        depth_roi = depth_data[yw_min_escalado:yw_max, xw_min:xw_max_escalado]
        frame_roi = frame_bgr[yw_min_escalado:yw_max, xw_min:xw_max_escalado]
        
        # Verificar dimensiones
        if depth_roi.shape != dmax_map.shape:
            print(f"[ERROR] Dimensiones no coinciden: depth_roi={depth_roi.shape}, dmax_map={dmax_map.shape}")
            break
        
        # Filtrar vibraciones
        if previous_roi is not None:
            roi_diff = cv2.absdiff(depth_roi, previous_roi)
            vibration_mask = cv2.threshold(roi_diff, vibration_threshold, 255, cv2.THRESH_BINARY)[1]
            vibration_mask = cv2.medianBlur(vibration_mask, ksize=5)
            depth_roi[vibration_mask > 0] = previous_roi[vibration_mask > 0]
        
        previous_roi = depth_roi.copy()
        
        # Crear máscara de toques
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        
        # Aplicar filtros
        touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
        
        # Historial de toques para estabilidad
        touch_history.append(touch_mask_filtered)
        if len(touch_history) > max_history_frames:
            touch_history.pop(0)
        
        # Acumular máscaras
        accumulated_mask = np.sum(touch_history, axis=0)
        accumulated_mask = np.clip(accumulated_mask, 0, 255).astype(np.uint8)
        _, final_touch_mask = cv2.threshold(accumulated_mask, touch_duration_threshold * 255, 255, cv2.THRESH_BINARY)
        
        # Detectar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_touch_mask, connectivity=8)
        min_size = 50
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # Calcular centroide
                mask_component = (labels == i).astype(np.uint8)
                M = cv2.moments(mask_component)
                
                if M['m00'] > 0:
                    # Centroide estándar
                    cx_standard = M['m10'] / M['m00']
                    cy_standard = M['m01'] / M['m00']
                    
                    # Centroide ponderado por profundidad
                    depth_inverse = (dmax_map - depth_roi).astype(np.float32)
                    depth_inverse[depth_inverse < 0] = 0
                    depth_weighted = mask_component.astype(np.float32) * depth_inverse
                    
                    M_weighted = cv2.moments(depth_weighted)
                    if M_weighted['m00'] > 0:
                        cx_weighted = M_weighted['m10'] / M_weighted['m00']
                        cy_weighted = M_weighted['m01'] / M_weighted['m00']
                        x_touch = int(round(0.75 * cx_weighted + 0.25 * cx_standard))
                        y_touch = int(round(0.75 * cy_weighted + 0.25 * cy_standard))
                    else:
                        x_touch = int(cx_standard)
                        y_touch = int(cy_standard)
                else:
                    centroid = centroids[i]
                    x_touch, y_touch = int(centroid[0]), int(centroid[1])
                
                # Convertir a coordenadas absolutas de cámara
                x_camara = xw_min + x_touch
                y_camara = yw_min_escalado + y_touch
                
                # Transformar usando homografía
                if homography_matrix is not None:
                    point_camara = np.array([[[x_camara, y_camara]]], dtype=np.float32)
                    point_proyeccion = cv2.perspectiveTransform(point_camara, homography_matrix)
                    x_viewport = int(round(point_proyeccion[0][0][0]))
                    y_viewport = int(round(point_proyeccion[0][0][1]))
                else:
                    # Fallback a mapeo lineal
                    sx = float(xv_max - xv_min) / depth_roi.shape[1]
                    sy = float(yv_max - yv_min) / depth_roi.shape[0]
                    x_viewport = int(round(xv_min + (x_touch * sx)))
                    y_viewport = int(round(yv_min + (y_touch * sy)))
                
                # Asegurar coordenadas dentro de límites
                x_viewport = np.clip(x_viewport, 0, PROJECTION_WIDTH - 1)
                y_viewport = np.clip(y_viewport, 0, PROJECTION_HEIGHT - 1)
                
                # Dibujar en cámara
                cv2.circle(frame_roi, (x_touch, y_touch), 8, (255, 0, 0), -1)
                cv2.circle(frame_roi, (x_touch, y_touch), 10, (255, 255, 255), 2)
                
                # Dibujar en proyección (más grande y visible)
                cv2.circle(proyeccion_prueba, (x_viewport, y_viewport), 12, (0, 0, 255), -1)
                cv2.circle(proyeccion_prueba, (x_viewport, y_viewport), 15, (255, 255, 255), 2)
                
                # Mostrar coordenadas
                coord_text = f"({x_viewport}, {y_viewport})"
                text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(proyeccion_prueba, 
                            (x_viewport + 15, y_viewport - 5), 
                            (x_viewport + 15 + text_size[0] + 5, y_viewport + text_size[1] + 5), 
                            (0, 0, 0), -1)
                cv2.putText(proyeccion_prueba, coord_text, (x_viewport + 18, y_viewport + 18),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar instrucciones
        instrucciones = [
            "Toca la mesa para probar",
            "Presiona 's' para guardar",
            "Presiona 'r' para recalibrar",
            "Presiona 'q' para salir"
        ]
        y_text = 30
        for texto in instrucciones:
            cv2.putText(proyeccion_prueba, texto, (10, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += 20
        
        # Mostrar ventanas
        cv2.imshow("Proyeccion", proyeccion_prueba)
        cv2.imshow("Camara", frame_bgr)
        cv2.imshow("ROI", frame_roi)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("\n✓ Calibración aceptada y guardada")
            calibracion_aceptada = True
            break
        elif key == ord('r'):
            print("\n↻ Recalibrando...")
            # Reiniciar calibración
            color_stream.stop()
            depth_stream.stop()
            cv2.destroyAllWindows()
            return calibrar_mesa(device)  # Llamada recursiva para recalibrar
        elif key == ord('q'):
            print("\n✗ Calibración cancelada")
            color_stream.stop()
            depth_stream.stop()
            cv2.destroyAllWindows()
            return None
    
    # Cerrar ventanas de prueba
    cv2.destroyWindow("ROI")
    
    print("\n" + "=" * 60)
    print("CALIBRACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("\nLa configuración está lista para usar en el juego principal.")
    print("Puedes cerrar esta ventana.")
    print("\n" + "=" * 60)
    print("CALIBRACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("\nLa configuración está lista para usar en el juego principal.")
    print("Puedes cerrar esta ventana.")
    
    # Mostrar mensaje final
    proyeccion = np.zeros((PROJECTION_HEIGHT, PROJECTION_WIDTH, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibracion Guardada!", (0, 255, 0), font_scale=2.0)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(3000)
    
    # Limpiar
    color_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()
    
    return config

# ============================================================================
# INICIALIZACIÓN Y MAIN
# ============================================================================

def inicializar_openni2():
    """Inicializa OpenNI2 buscando en múltiples ubicaciones"""
    openni2_paths = []
    
    # Variable de entorno
    env_path = os.environ.get("OPENNI2_PATH")
    if env_path:
        openni2_paths.append(env_path)
        openni2_paths.append(os.path.join(env_path, "Redist"))
    
    # Ubicaciones comunes
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    
    openni2_paths.extend([
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
        os.path.join(program_files, "OpenNI2", "Redist") if program_files else None,
        os.path.join(program_files_x86, "OpenNI2", "Redist") if program_files_x86 else None,
    ])
    
    # Eliminar None y duplicados
    openni2_paths = list(dict.fromkeys([p for p in openni2_paths if p]))
    
    for path in openni2_paths:
        if os.path.exists(path):
            try:
                openni2.initialize(path)
                print(f"✓ OpenNI2 inicializado desde: {path}")
                return True
            except:
                continue
    
    print("=" * 60)
    print("ERROR: No se pudo encontrar OpenNI2 SDK")
    print("=" * 60)
    print("\nInstala OpenNI2 SDK desde: https://structure.io/openni")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE CALIBRACIÓN - REALIDAD AUMENTADA ESPACIAL")
    print("=" * 60)
    
    # Inicializar OpenNI2
    if not inicializar_openni2():
        exit(1)
    
    # Abrir dispositivo
    try:
        print("\nBuscando dispositivo Kinect...")
        device = openni2.Device.open_any()
        device_info = device.get_device_info()
        print(f"✓ Dispositivo encontrado: {device_info.name.decode('utf-8') if device_info.name else 'Kinect'}")
    except Exception as e:
        print(f"\nERROR: No se pudo abrir el dispositivo: {e}")
        print("\nAsegúrate de que:")
        print("1. El Kinect esté conectado y encendido")
        print("2. No haya otros programas usando el Kinect")
        print("3. Los drivers estén instalados correctamente")
        openni2.unload()
        exit(1)
    
    # Ejecutar calibración
    try:
        config = calibrar_mesa(device)
        if config:
            print("\n✓ Calibración completada exitosamente!")
        else:
            print("\n✗ La calibración no se completó")
    except KeyboardInterrupt:
        print("\n\nCalibración cancelada por el usuario")
    except Exception as e:
        print(f"\nERROR durante la calibración: {e}")
        import traceback
        traceback.print_exc()
    finally:
        openni2.unload()
        print("\nSistema cerrado correctamente")

