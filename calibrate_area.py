import cv2
import numpy as np
from tqdm import tqdm
from openni import openni2
import customtkinter as ctk  # Para preguntar al usuario
import json
import os

# Función para mostrar un mensaje en pantalla
def mostrar_mensaje(proyeccion, texto, xv_min, yv_min, xv_max, yv_max):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(texto, font, 2, 4)[0]  # Obtener el tamaño del texto
    text_x = xv_min + (xv_max - xv_min - text_size[0]) // 2  # Centrar horizontalmente
    text_y = yv_min + (yv_max - yv_min + text_size[1]) // 2  # Centrar verticalmente
    cv2.putText(proyeccion, texto, (text_x, text_y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)

# Función para calcular dmax_map con barra de progreso y mensaje
def calculate_dmax(device, calibrated_area, xv_min, yv_min, xv_max, yv_max, num_frames=500):
    depth_stream = device.create_depth_stream()
    depth_stream.start()

    x, y, w, h = calibrated_area
    print(f"{w} * {h} = {w * h} ")

    # Definir un rango de profundidad para optimizar el uso de memoria
    min_depth = 500  # Ajusta según tu aplicación
    max_depth = 4000
    depth_accum = np.zeros((h, w, max_depth - min_depth + 1), dtype=int)

    # Crear una ventana de proyección para mostrar el mensaje
    proyeccion = np.zeros((800, 1280, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibrando...", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1)

    for _ in tqdm(range(num_frames), desc="Numero de frames", unit="frames"):
        frame = depth_stream.read_frame()
        depth_data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[y:y+h, x:x+w]

        # Vectorizado para contar frecuencias
        valid_mask = (depth_roi >= min_depth) & (depth_roi <= max_depth)
        valid_depth = depth_roi[valid_mask] - min_depth
        indices = np.where(valid_mask)
        depth_accum[indices[0], indices[1], valid_depth] += 1

    depth_stream.stop()

    # Generar el mapa dmax basado en la moda de la profundidad
    dmax_map = np.argmax(depth_accum, axis=2) + min_depth

    np.savetxt("config/dmax_map.txt", dmax_map.flatten(), fmt="%d")
    
    # Mostrar mensaje de finalización
    cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0,0,0), -1)
    mostrar_mensaje(proyeccion, "Calibracion Completada", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1000)
    
    return dmax_map

# Función para proyectar cuadrados de calibración
def proyectar_cuadrados(view_width, view_height):
    proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    cuadrado_size = 50
    margen = 250

    # Cuadrado inferior izquierdo
    x_izquierda = margen
    y_izquierda = view_height - cuadrado_size 
    cv2.rectangle(proyeccion, (x_izquierda, y_izquierda),
                  (x_izquierda + cuadrado_size, y_izquierda + cuadrado_size), (255, 255, 255), -1)

    cx_izquierda = x_izquierda + cuadrado_size // 2
    cy_izquierda = y_izquierda + cuadrado_size // 2

    # Cuadrado superior derecho
    y_derecha = margen
    x_derecha = view_width - margen - cuadrado_size
    cv2.rectangle(proyeccion, (x_derecha, y_derecha),
                  (x_derecha + cuadrado_size, y_derecha + cuadrado_size), (255, 255, 255), -1)

    cx_derecha = x_derecha + cuadrado_size // 2
    cy_derecha = y_derecha + cuadrado_size // 2

    return proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, (cx_izquierda, cy_izquierda), (cx_derecha, cy_derecha)

# Detección de cuadrados en la imagen de la cámara (mejorada)
def detectar_cuadrados(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Probar múltiples métodos de umbralización para mayor robustez
    cuadrados_encontrados = []
    
    # Método 1: Umbral adaptativo (mejor para diferentes condiciones de iluminación)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # Método 2: Umbral fijo con diferentes valores
    _, thresh_fixed1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, thresh_fixed2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, thresh_fixed3 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Probar con cada umbral
    for thresh in [thresh_adapt, thresh_fixed1, thresh_fixed2, thresh_fixed3]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Área mínima más flexible (entre 200 y 5000 píxeles)
            if 200 < area < 5000:
                # Calcular la relación de aspecto del rectángulo delimitador
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Un cuadrado debería tener relación de aspecto cercana a 1
                if 0.7 < aspect_ratio < 1.3:
                    # Aproximación más flexible (acepta formas cercanas a cuadrados)
                    epsilon = 0.04 * cv2.arcLength(cnt, True)  # Más flexible que 0.02
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # Aceptar formas con 4 vértices o cercanas (3-5 vértices)
                    if 3 <= len(approx) <= 5:
                        # Calcular qué tan "cuadrada" es la forma
                        # Usar la relación entre área del contorno y área del rectángulo delimitador
                        rect_area = w * h
                        extent = float(area) / rect_area if rect_area > 0 else 0
                        
                        # Un cuadrado debería llenar bien su rectángulo delimitador
                        if extent > 0.7:  # Al menos 70% del rectángulo
                            # Evitar duplicados (verificar si ya existe un cuadrado similar)
                            es_duplicado = False
                            for cuadrado_existente, _ in cuadrados_encontrados:
                                # Calcular distancia entre centros
                                M_existente = cv2.moments(cuadrado_existente)
                                M_nuevo = cv2.moments(cnt)
                                if M_existente['m00'] > 0 and M_nuevo['m00'] > 0:
                                    cx_existente = M_existente['m10'] / M_existente['m00']
                                    cy_existente = M_existente['m01'] / M_existente['m00']
                                    cx_nuevo = M_nuevo['m10'] / M_nuevo['m00']
                                    cy_nuevo = M_nuevo['m01'] / M_nuevo['m00']
                                    
                                    distancia = np.sqrt((cx_existente - cx_nuevo)**2 + (cy_existente - cy_nuevo)**2)
                                    if distancia < 50:  # Si están muy cerca, es probablemente el mismo
                                        es_duplicado = True
                                        break
                            
                            if not es_duplicado:
                                cuadrados_encontrados.append((cnt, area))
    
    # Ordenar por área (más grandes primero) y devolver los 2 más grandes
    cuadrados_encontrados = sorted(cuadrados_encontrados, key=lambda x: x[1], reverse=True)
    
    # Devolver máximo 2 cuadrados (los más grandes)
    return [cuadrado[0] for cuadrado in cuadrados_encontrados[:2]]

# Función para calibrar la mesa y detectar toques
def calibrar_mesa_y_detectar_toques(device):
    view_width = 1280  # Ancho de la proyección (videobeam)
    view_height = 800  # Alto de la proyección (videobeam)

    # Crear una ventana para la proyección
    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Proyeccion", 1920, 0)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Verificar primero si el sensor de color está disponible
    if not device.has_sensor(openni2.SENSOR_COLOR):
        print("=" * 60)
        print("ERROR: El dispositivo no tiene sensor de color disponible")
        print("=" * 60)
        return
    
    # Iniciar los streams de color y profundidad
    color_stream = None
    try:
        color_stream = device.create_color_stream()
        color_stream.start()
        print("✓ Stream de color iniciado correctamente")
    except Exception as e:
        print("=" * 60)
        print("ERROR: No se pudo iniciar el stream de color")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nPOSIBLES SOLUCIONES:")
        print("1. CIERRA COMPLETAMENTE Kinect Studio y cualquier otra aplicación")
        print("   que esté usando el Kinect (verifica en el Administrador de tareas)")
        print("2. Desconecta y vuelve a conectar el cable USB del Kinect")
        print("3. Espera 5 segundos después de cerrar Kinect Studio antes de ejecutar")
        print("4. Si el problema persiste, reinicia tu computadora")
        print("\nPROCESOS COMUNES QUE USAN KINECT:")
        print("- Kinect Studio")
        print("- Kinect Explorer")
        print("- Otros scripts de Python con OpenNI2")
        print("=" * 60)
        return
    
    try:
        depth_stream = device.create_depth_stream()
        depth_stream.start()
        print("✓ Stream de profundidad iniciado correctamente")
    except Exception as e:
        print("=" * 60)
        print("ERROR: No se pudo iniciar el stream de profundidad")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nPOSIBLES SOLUCIONES:")
        print("1. CIERRA COMPLETAMENTE Kinect Studio y cualquier otra aplicación")
        print("   que esté usando el Kinect (verifica en el Administrador de tareas)")
        print("2. Desconecta y vuelve a conectar el cable USB del Kinect")
        print("3. Espera 5 segundos después de cerrar Kinect Studio antes de ejecutar")
        print("=" * 60)
        if color_stream is not None:
            try:
                color_stream.stop()
            except:
                pass
        return

    # Leer el primer frame de la cámara de profundidad antes de usar depth_data
    depth_frame = depth_stream.read_frame()
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_data = cv2.flip(depth_data, 1)

    # Inicializar variables para homografía y calibración
    homography_matrix = None
    xw_min = None
    xw_max_escalado = None
    yw_min_escalado = None
    yw_max = None
    xv_min = None
    xv_max = None
    yv_min = None
    yv_max = None
    calibracion_completada = False

    # Proceso de calibración
    while True:
        proyeccion, x_izquierda, y_izquierda, x_derecha, y_derecha, cuadrado_size, centroide_izquierda, centroide_derecha = proyectar_cuadrados(view_width, view_height)
        cv2.imshow("Proyeccion", proyeccion)
        cv2.waitKey(2000)

        frame = color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 1)
        cuadrados_detectados = detectar_cuadrados(frame_bgr)

        # Crear imagen de depuración para mostrar qué se detecta
        frame_debug = frame_bgr.copy()
        
        # Dibujar todos los cuadrados detectados
        for i, cuadrado in enumerate(cuadrados_detectados):
            cv2.drawContours(frame_debug, [cuadrado], -1, (0, 255, 0), 2)
            M = cv2.moments(cuadrado)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame_debug, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame_debug, f"Cuadrado {i+1}", (cx-30, cy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Mostrar información en la imagen
        info_text = f"Cuadrados detectados: {len(cuadrados_detectados)}/2"
        cv2.putText(frame_debug, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if len(cuadrados_detectados) < 2:
            ayuda_text = "Asegurate de que ambos cuadrados blancos sean visibles"
            cv2.putText(frame_debug, ayuda_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Esperando cuadrados... Detectados: {len(cuadrados_detectados)}/2")

        cv2.imshow("Camara", frame_debug)

        if len(cuadrados_detectados) >= 2:
            puntos_camara = []
            for cuadrado in cuadrados_detectados[:2]:
                M = cv2.moments(cuadrado)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    puntos_camara.append([cx, cy])
                    cv2.drawContours(frame_bgr, [cuadrado], -1, (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

            if len(puntos_camara) == 2:
                xw_min, xw_max = sorted([puntos_camara[1][0], puntos_camara[0][0]]) #vmin, vmax
                yw_min, yw_max = sorted([puntos_camara[0][1], puntos_camara[1][1]]) #umin, umax

                x1, y1 = centroide_izquierda
                x2, y2 = centroide_derecha

                xv_min, xv_max = sorted([x1, x2])
                yv_min, yv_max = sorted([y1, y2])

                xw_min = max(0, min(xw_min, 640))
                xw_max = max(0, min(xw_max, 640))
                yw_min = max(0, min(yw_min, 480))
                yw_max = max(0, min(yw_max, 480))
                print(f"Luego de max: {yw_max}")

                # Escalar las coordenadas de la ROI de profundidad
                factor_escala_x = 1.16
                factor_escala_y = 1.12

                xw_centro = (xw_min + xw_max) // 2
                yw_centro = (yw_min + yw_max) // 2

                # Aplicar escalado
                xw_min = xw_min - 10
                xw_max_escalado = min(640, int(xw_centro + (xw_max - xw_centro) * factor_escala_x))
                yw_min_escalado = max(0, int(yw_centro - (yw_centro - yw_min) * factor_escala_y))

                # Calcular homografía para mapeo preciso de coordenadas
                # Puntos en la cámara (coordenadas de la ROI)
                pts_camara = np.array([
                    [xw_min, yw_min_escalado],           # Esquina superior izquierda
                    [xw_max_escalado, yw_min_escalado],  # Esquina superior derecha
                    [xw_max_escalado, yw_max],           # Esquina inferior derecha
                    [xw_min, yw_max]                     # Esquina inferior izquierda
                ], dtype=np.float32)
                
                # Puntos correspondientes en la proyección
                pts_proyeccion = np.array([
                    [xv_min, yv_min],                     # Esquina superior izquierda
                    [xv_max, yv_min],                     # Esquina superior derecha
                    [xv_max, yv_max],                     # Esquina inferior derecha
                    [xv_min, yv_max]                      # Esquina inferior izquierda
                ], dtype=np.float32)
                
                # Calcular matriz de homografía
                homography_matrix = cv2.getPerspectiveTransform(pts_camara, pts_proyeccion)
                print("[OK] Matriz de homografía calculada para mapeo preciso")

                # Extraer la ROI escalada
                depth_roi_escalado = depth_data[yw_min_escalado:yw_max, xw_min:xw_max_escalado]

                # Dibujar el rectángulo de depuración sobre la proyección para verificar que pasa por los cuadrados
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

                # Mostrar el rectángulo en la proyección
                cv2.imshow("Proyeccion", proyeccion)

                cv2.rectangle(frame_bgr, (xw_min, yw_min_escalado), (xw_max_escalado, yw_max), (255, 0, 0), 2)
                cv2.imshow("Camara", frame_bgr)
                print("Calibración completada.")
                calibracion_completada = True
                cv2.waitKey(5000)
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Verificar que la calibración se completó antes de continuar
    if not calibracion_completada:
        print("=" * 60)
        print("ERROR: La calibración no se completó")
        print("=" * 60)
        print("Asegúrate de que los cuadrados blancos sean visibles en la cámara")
        print("y que estén bien iluminados.")
        print("=" * 60)
        color_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()
        return

    # Calcular dmax_map solo si la calibración se completó
    dmax_map = calculate_dmax(device, (xw_min, yw_min_escalado, xw_max_escalado - xw_min, yw_max - yw_min_escalado),xv_min, yv_min, xv_max, yv_max)
    # Detección de toques
    if xw_min is not None:
        print("Iniciando detección de toques...")

        # Preguntar al usuario si desea proceder con la detección de toques
        user_response = None
        
        def on_yes():
            nonlocal user_response
            user_response = 'Sí'
            app.destroy()
            
        def on_no():
            nonlocal user_response
            user_response = 'No'
            app.destroy()

        ctk.set_appearance_mode("Dark")
        app = ctk.CTk()
        app.title('Verificación de Detección')
        app.geometry("400x200")
        
        label = ctk.CTkLabel(app, text='¿Desea proceder con la detección de toques?', font=("Arial", 16))
        label.pack(pady=20)
        
        button_frame = ctk.CTkFrame(app, fg_color="transparent")
        button_frame.pack(pady=10)
        
        btn_yes = ctk.CTkButton(button_frame, text='Sí', command=on_yes)
        btn_yes.pack(side="left", padx=10)
        
        btn_no = ctk.CTkButton(button_frame, text='No', command=on_no)
        btn_no.pack(side="left", padx=10)
        
        app.mainloop()

        if user_response == 'Sí':
            # Ajustar dmax_map y calcular dmin_map
            # Asegurar que dmax_map sea un array numpy
            if not isinstance(dmax_map, np.ndarray):
                print("[ERROR] dmax_map no es un array válido")
                color_stream.stop()
                depth_stream.stop()
                cv2.destroyAllWindows()
                return
            
            dmax_map = dmax_map.astype(np.int16) - 4
            dmin_map = dmax_map.astype(np.int16) - 10
            
            # Verificar dimensiones
            roi_height = yw_max - yw_min_escalado
            roi_width = xw_max_escalado - xw_min
            if dmax_map.shape != (roi_height, roi_width):
                print(f"[ADVERTENCIA] Dimensiones no coinciden: dmax_map={dmax_map.shape}, ROI=({roi_height}, {roi_width})")
                print("[INFO] Redimensionando dmax_map...")
                dmax_map = cv2.resize(dmax_map.astype(np.float32), (roi_width, roi_height), interpolation=cv2.INTER_NEAREST).astype(np.int16)
                dmin_map = cv2.resize(dmin_map.astype(np.float32), (roi_width, roi_height), interpolation=cv2.INTER_NEAREST).astype(np.int16)

            # Asegurar que el stream de profundidad esté iniciado
            try:
                depth_stream.start()
            except:
                pass  # Ya está iniciado, no hay problema

            print("Iniciando detección de toques...")
            print(f"Parámetros de detección:")
            print(f"  - dmax_map: {dmax_map.min()} - {dmax_map.max()}")
            print(f"  - dmin_map: {dmin_map.min()} - {dmin_map.max()}")
            print(f"  - ROI: xw_min={xw_min}, xw_max={xw_max_escalado}, yw_min={yw_min_escalado}, yw_max={yw_max}")
            previous_roi = None
            touch_history = []
            vibration_threshold = 15  # Umbral para vibraciones
            touch_duration_threshold = 5  # Duración de toque requerida
            max_history_frames = 10  # Máximo de frames en el historial

            # Crear la proyección usando las dimensiones del área de trabajo
            proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
            
            # Dibujar el rectángulo del área calibrada en la proyección
            cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
            
            print("[OK] Bucle de detección iniciado. Toca el área calibrada para probar.")
            print("[INFO] Los puntos rojos aparecerán en la proyección cuando toques la superficie.")

            while True:
                # Leer frame de la cámara de profundidad
                depth_frame = depth_stream.read_frame()
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                depth_data = cv2.flip(depth_data, 1)
                
                # Leer frame de color para visualización
                try:
                    color_frame = color_stream.read_frame()
                    frame_data = color_frame.get_buffer_as_uint8()
                    frame_array = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8, buffer=frame_data)
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    frame_bgr = cv2.flip(frame_bgr, 1)
                except:
                    # Si no se puede leer el frame de color, crear uno vacío
                    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Limpiar el área calibrada en la proyección antes de dibujar nuevos puntos
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 0, 0), -1)
                cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
                
                # Extraer la ROI escalada
                depth_roi = depth_data[yw_min_escalado:yw_max, xw_min:xw_max_escalado]
                frame_roi = frame_bgr[yw_min_escalado:yw_max, xw_min:xw_max_escalado]

                if previous_roi is not None:
                    # Detectar vibraciones y ajustar el ROI
                    roi_diff = cv2.absdiff(depth_roi, previous_roi)
                    vibration_mask = cv2.threshold(roi_diff, vibration_threshold, 255, cv2.THRESH_BINARY)[1]
                    vibration_mask = cv2.medianBlur(vibration_mask, ksize=5)

                    depth_roi[vibration_mask > 0] = previous_roi[vibration_mask > 0]

                previous_roi = depth_roi.copy()

                # Verificar dimensiones antes de crear la máscara
                if depth_roi.shape != dmax_map.shape:
                    print(f"[ERROR] Dimensiones no coinciden: depth_roi={depth_roi.shape}, dmax_map={dmax_map.shape}")
                    break
                
                # Crear la "coraza" entre dmin y dmax
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                
                # Depuración: mostrar estadísticas cada 30 frames
                if len(touch_history) % 30 == 0:
                    pixels_touch = np.sum(touch_mask > 0)
                    print(f"Frame {len(touch_history)}: Píxeles en máscara de toque: {pixels_touch}")

                # Aplicar filtros más suaves para no desplazar el centroide
                # Usar filtro mediano más pequeño para preservar posición
                touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
                
                # Aplicar apertura morfológica para eliminar ruido pequeño (antes de calcular centroide)
                kernel = np.ones((3, 3), np.uint8)
                touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)

                # Identificar componentes conectados
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_filtered, connectivity=8)
                min_size = 100  # Tamaño mínimo para considerar un área como toque válido
                
                # Filtrar componentes pequeños
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_size:
                        touch_mask_filtered[labels == i] = 0

                touch_history.append(touch_mask_filtered)
                
                # Limitar el tamaño del historial para evitar acumulación excesiva
                if len(touch_history) > max_history_frames:
                    touch_history.pop(0)

                # Acumular las máscaras para identificar toques persistentes
                accumulated_mask = np.sum(touch_history, axis=0)
                accumulated_mask = np.clip(accumulated_mask, 0, 255).astype(np.uint8)

                # Considerar solo toques que persisten durante varios cuadros
                _, final_touch_mask = cv2.threshold(accumulated_mask, touch_duration_threshold * 255, 255, cv2.THRESH_BINARY)
                
                # Recalcular componentes conectados en la máscara final
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_touch_mask, connectivity=8)
                
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        # Calcular centroide mejorado usando momentos ponderados por profundidad
                        # Obtener la región del componente
                        mask_component = (labels == i).astype(np.uint8)
                        
                        # Calcular momentos para obtener centroide más preciso
                        M = cv2.moments(mask_component)
                        if M['m00'] > 0:
                            # Centroide estándar
                            cx_standard = M['m10'] / M['m00']
                            cy_standard = M['m01'] / M['m00']
                            
                            # Centroide ponderado por profundidad (más preciso)
                            # Usar la profundidad como peso: áreas más cercanas tienen más peso
                            depth_inverse = (dmax_map - depth_roi).astype(np.float32)  # Invertir: más cercano = mayor valor
                            depth_inverse[depth_inverse < 0] = 0
                            depth_weighted = mask_component.astype(np.float32) * depth_inverse
                            
                            # Calcular momentos ponderados
                            M_weighted = cv2.moments(depth_weighted)
                            if M_weighted['m00'] > 0:
                                cx_weighted = M_weighted['m10'] / M_weighted['m00']
                                cy_weighted = M_weighted['m01'] / M_weighted['m00']
                                # Usar promedio ponderado entre ambos métodos
                                x_touch = int(0.7 * cx_weighted + 0.3 * cx_standard)
                                y_touch = int(0.7 * cy_weighted + 0.3 * cy_standard)
                            else:
                                x_touch = int(cx_standard)
                                y_touch = int(cy_standard)
                        else:
                            # Fallback al centroide básico
                            centroid = centroids[i]
                            x_touch, y_touch = int(centroid[0]), int(centroid[1])

                        # Convertir coordenadas relativas de ROI a coordenadas absolutas de cámara
                        x_camara = xw_min + x_touch
                        y_camara = yw_min_escalado + y_touch

                        # Usar homografía para mapeo preciso (transformación de perspectiva)
                        if homography_matrix is not None:
                            # Transformar punto usando homografía
                            point_camara = np.array([[[x_camara, y_camara]]], dtype=np.float32)
                            point_proyeccion = cv2.perspectiveTransform(point_camara, homography_matrix)
                            x_viewport = int(point_proyeccion[0][0][0])
                            y_viewport = int(point_proyeccion[0][0][1])
                        else:
                            # Fallback a mapeo lineal si no hay homografía
                            sx = float(xv_max - xv_min) / depth_roi.shape[1]
                            sy = float(yv_max - yv_min) / depth_roi.shape[0]
                            x_viewport = int(xv_min + (x_touch * sx))
                            y_viewport = int(yv_min + (y_touch * sy))

                        # Asegurar que las coordenadas estén dentro de los límites del viewport
                        x_viewport = np.clip(x_viewport, 0, view_width - 1)
                        y_viewport = np.clip(y_viewport, 0, view_height - 1)

                        # Dibujar en la ROI para visualización (cámara)
                        cv2.circle(frame_roi, (x_touch, y_touch), 8, (255, 0, 0), -1)
                        cv2.circle(frame_roi, (x_touch, y_touch), 10, (255, 255, 255), 2)

                        # Dibujar punto en la proyección (más grande y visible)
                        cv2.circle(proyeccion, (x_viewport, y_viewport), 10, (0, 0, 255), -1)  # Círculo rojo sólido
                        cv2.circle(proyeccion, (x_viewport, y_viewport), 12, (255, 255, 255), 2)  # Borde blanco

                        # Mostrar coordenadas mapeadas (más visible)
                        coord_text = f"({x_viewport}, {y_viewport})"
                        # Fondo negro para el texto para mejor legibilidad
                        text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(proyeccion, 
                                    (x_viewport + 15, y_viewport - 5), 
                                    (x_viewport + 15 + text_size[0] + 5, y_viewport + text_size[1] + 5), 
                                    (0, 0, 0), -1)
                        cv2.putText(proyeccion, coord_text, (x_viewport + 18, y_viewport + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Mostrar coordenadas en consola también
                        print(f"[TOQUE] Coordenadas: ({x_viewport}, {y_viewport})")

                # Mostrar la proyección y las máscaras
                full_mask = np.zeros_like(depth_data, dtype=np.uint8)
                full_mask[yw_min_escalado:yw_max, xw_min:xw_max_escalado] = final_touch_mask
                cv2.imshow("Proyeccion", proyeccion)
                cv2.imshow("Mascara de Toque", full_mask)
                cv2.imshow("Camara", frame_bgr)
                cv2.imshow("Camara2", frame_roi)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        print(f"Guardando coordenadas, yw_max: {yw_max}")
        coordenadas = {
            "xv_min": xv_min,
            "xv_max": xv_max,
            "yv_min": yv_min,
            "yv_max": yv_max,
            "xw_min": xw_min,
            "xw_max": xw_max_escalado,  # Usamos xw_max_escalado como xw_max
            "yw_min": yw_min_escalado,   # Usamos yw_min_escalado como yw_min
            "yw_max": yw_max,
            "homography_matrix": homography_matrix.tolist() if homography_matrix is not None else None
            }

         # Guardar las coordenadas en un archivo JSON en la carpeta config
        with open("config/ultima_configuracion_coordenadas.json", "w") as file:
            json.dump(coordenadas, file, indent=4)            
        # Detener los streams y destruir las ventanas           
        color_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()

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
        print("\nUbicaciones buscadas:")
        for path in openni2_paths[:10]:
            status = "✓ Existe" if os.path.exists(path) else "✗ No existe"
            print(f"  {status}: {path}")
        if len(openni2_paths) > 10:
            print(f"  ... y {len(openni2_paths) - 10} ubicaciones más")
        if last_exception:
            print(f"\nÚltimo error: {last_exception}")
        print("=" * 60)
        exit(1)
    
    # Intentar abrir el dispositivo
    try:
        print("Buscando dispositivo OpenNI2...")
        device = openni2.Device.open_any()
        device_info = device.get_device_info()
        print(f"✓ Dispositivo encontrado: {device_info.name.decode('utf-8') if device_info.name else 'Desconocido'}")
    except Exception as e:
        print("=" * 60)
        print("ERROR: No se pudo abrir el dispositivo OpenNI2")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPOSIBLES SOLUCIONES:")
        print("1. Verifica que el sensor Kinect esté conectado y encendido")
        print("2. Asegúrate de que ningún otro programa esté usando el dispositivo")
        print("3. Reinicia el sensor Kinect desconectándolo y volviéndolo a conectar")
        print("4. Verifica que los drivers del Kinect estén instalados correctamente")
        print("=" * 60)
        openni2.unload()
        exit(1)
    
    try:
        calibrar_mesa_y_detectar_toques(device)
    except KeyboardInterrupt:
        print("\n\nCalibración cancelada por el usuario.")
    except Exception as e:
        print(f"\nERROR durante la calibración: {e}")
        import traceback
        traceback.print_exc()
    finally:
        openni2.unload()
