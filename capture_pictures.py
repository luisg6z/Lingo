# -*- coding: utf-8 -*-
"""
Script para capturar imágenes del Kinect con capacidad de cambiar entre diferentes vistas.
Permite visualizar la cámara en modo raw o con rectángulos de calibración superpuestos.
La vista de rectángulos se muestra en la segunda pantalla como fondo.
"""

import cv2
import numpy as np
from openni import openni2
import os
import sys
import io
import json
from datetime import datetime

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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


def initialize_openni2():
    """Inicializa OpenNI2 buscando en múltiples ubicaciones comunes."""
    openni2_paths = []
    
    # 1. Verificar variable de entorno
    env_path = os.environ.get("OPENNI2_PATH")
    if env_path:
        openni2_paths.append(env_path)
        openni2_paths.append(os.path.join(env_path, "Redist"))
    
    # 2. Agregar rutas comunes
    openni2_paths.extend([
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
    ])
    
    # 3. Agregar ubicaciones usando variables de entorno
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    
    if program_files:
        openni2_paths.append(os.path.join(program_files, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files, "OpenNI2"))
    
    if program_files_x86:
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2"))
    
    # 4. Otras rutas comunes
    openni2_paths.extend([
        "C:/Program Files (x86)/OpenNI2/Redist",
        "C:/Program Files (x86)/OpenNI2",
        "C:/OpenNI2/Redist",
        "C:/OpenNI2",
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
                return True
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
        if last_exception:
            print(f"\nÚltimo error: {last_exception}")
        print("=" * 60)
        return False
    
    return True


def load_calibration():
    """Carga las coordenadas de calibración desde el archivo JSON."""
    config_path = "config/ultima_configuracion_coordenadas.json"
    
    if not os.path.exists(config_path):
        print(f"⚠ Advertencia: No se encontró el archivo de calibración: {config_path}")
        print("  El modo de rectángulos no estará disponible.")
        return None
    
    try:
        with open(config_path, "r") as file:
            coordenadas = json.load(file)
        print(f"✓ Calibración cargada desde: {config_path}")
        return coordenadas
    except Exception as e:
        print(f"⚠ Error al cargar calibración: {e}")
        return None


def create_output_directory():
    """Crea el directorio para guardar las imágenes si no existe."""
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_next_image_number(output_dir):
    """Obtiene el siguiente número de imagen disponible."""
    existing_images = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not existing_images:
        return 1
    
    # Extraer números de los nombres de archivo
    numbers = []
    for img in existing_images:
        try:
            # Buscar números en el nombre del archivo
            name_without_ext = os.path.splitext(img)[0]
            # Intentar extraer el número
            num_str = ''.join(filter(str.isdigit, name_without_ext))
            if num_str:
                numbers.append(int(num_str))
        except:
            continue
    
    return max(numbers) + 1 if numbers else 1


def create_rectangles_view_screen(coordenadas, escenarios_seleccionados=None):
    """
    Crea la pantalla con rectángulos de escenarios similar a mostrar_vista_rectangulos_escenarios.
    
    Args:
        coordenadas: Diccionario con las coordenadas de calibración
        escenarios_seleccionados: Lista de escenarios a mostrar (si None, muestra todos los disponibles)
    
    Returns:
        numpy array con la pantalla de rectángulos
    """
    if coordenadas is None:
        return None
    
    # Extraer coordenadas
    xw_min = coordenadas.get("xw_min", 0)
    xw_max = coordenadas.get("xw_max", 640)
    yw_min = coordenadas.get("yw_min", 0)
    yw_max = coordenadas.get("yw_max", 480)
    xv_min = coordenadas.get("xv_min", 0)
    xv_max = coordenadas.get("xv_max", 1280)
    yv_min = coordenadas.get("yv_min", 0)
    yv_max = coordenadas.get("yv_max", 800)
    
    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800
    
    # Mapeo de escenarios a imágenes
    escenario_images = {
        "Escenario 1": "images/EscenarioGranja.png",
        "Escenario 2": "images/EscenarioCalle.png",
        "Escenario 3": "images/Escenariorestaurante.png",
        "Escenario 4": "images/EscenarioRopa.png",
        "Escenario 5": "images/EscenarioColegio.png"
    }
    
    # Si no se especifican escenarios, usar todos los disponibles
    if escenarios_seleccionados is None:
        escenarios_seleccionados = list(escenario_images.keys())
    
    num_escenarios = len(escenarios_seleccionados)
    
    # Cargar imágenes de los escenarios seleccionados
    loaded_escenario_images = {}
    for escenario in escenarios_seleccionados:
        if escenario in escenario_images:
            path = escenario_images[escenario]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    loaded_escenario_images[escenario] = img
    
    # Crear fondo
    rectangulos_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que el menú)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        rectangulos_screen[y, :] = [b, g, r]
    
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
    
    # Dimensiones de los rectángulos
    base_rect_width = 450
    base_rect_height = 480
    rect_spacing = 70
    
    # Margen de seguridad desde los bordes del área visible
    margin_x = 30
    margin_y = 30
    
    # Calcular ancho disponible dentro del área visible
    available_width = visible_area_width - 2 * margin_x
    available_height = visible_area_height - 2 * margin_y
    
    # Ajustar el tamaño de las cards para que quepan en el área visible
    rect_width = base_rect_width
    rect_height = base_rect_height
    
    if rect_height > available_height:
        rect_height = available_height
    
    # Calcular el ancho total necesario
    total_width_needed = num_escenarios * rect_width + (num_escenarios - 1) * rect_spacing
    
    # Si no caben, ajustar el spacing
    if total_width_needed > available_width:
        if num_escenarios > 1:
            max_spacing = (available_width - num_escenarios * rect_width) / (num_escenarios - 1)
            if max_spacing >= 30:
                rect_spacing = int(max_spacing)
            else:
                max_rect_width = (available_width - (num_escenarios - 1) * 30) / num_escenarios
                rect_width = int(max_rect_width)
                rect_spacing = 30
        else:
            if rect_width > available_width:
                rect_width = available_width
    
    # Recalcular el ancho total
    total_width = num_escenarios * rect_width + (num_escenarios - 1) * rect_spacing
    
    # Calcular start_x centrado
    start_x = xv_min + margin_x + (available_width - total_width) // 2
    if start_x < xv_min + margin_x:
        start_x = xv_min + margin_x
    
    # Verificar que el último rectángulo no choque con el borde derecho
    last_rect_end = start_x + (num_escenarios - 1) * (rect_width + rect_spacing) + rect_width
    if last_rect_end > xv_max - margin_x:
        start_x = xv_max - margin_x - (num_escenarios - 1) * (rect_width + rect_spacing) - rect_width
        start_x = max(xv_min + margin_x, start_x)
    
    # Calcular start_y
    start_y_base = 250
    start_y = max(start_y_base, yv_min + margin_y + (available_height - rect_height) // 2)
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
                
                img_area_height = h
                img_area_width = w
                
                # Calcular el factor de escala
                scale_w = img_area_width / img_w
                scale_h = img_area_height / img_h
                scale = max(scale_w, scale_h)
                
                # Redimensionar la imagen
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Calcular posición para centrar la imagen
                img_x = x + (w - new_w) // 2
                img_y = y + (h - new_h) // 2
                
                # Recortar si es necesario
                if new_w > w or new_h > h:
                    crop_x = max(0, (new_w - w) // 2)
                    crop_y = max(0, (new_h - h) // 2)
                    crop_w = min(w, new_w)
                    crop_h = min(h, new_h)
                    
                    resized_img = resized_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    img_x = x
                    img_y = y
                    new_w = crop_w
                    new_h = crop_h
                
                # Aplicar transparencia del 90%
                opacity = 0.9
                
                if resized_img.shape[2] == 4:
                    # Si la imagen tiene canal alfa
                    b, g, r, a_original = cv2.split(resized_img)
                    a_original = a_original.astype(np.float32) / 255.0
                    a_combined = a_original * opacity
                    
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - a_combined) +
                            resized_img[:, :, c] * a_combined
                        ).astype(np.uint8)
                else:
                    # Si no tiene canal alfa
                    for c in range(3):
                        screen[img_y:img_y+new_h, img_x:img_x+new_w, c] = (
                            screen[img_y:img_y+new_h, img_x:img_x+new_w, c] * (1 - opacity) +
                            resized_img[:, :, c] * opacity
                        ).astype(np.uint8)
            
            # Dibujar borde del rectángulo
            border_color = (150, 150, 150)  # Gris medio
            border_thickness = 4
            cv2.rectangle(screen, (x, y), (x + w, y + h), border_color, border_thickness)
    
    # Dibujar los rectángulos
    draw_rectangulos(rectangulos_screen, rectangulos_positions)
    
    return rectangulos_screen


def draw_instructions(frame, image_count, view_mode):
    """Dibuja instrucciones en el frame."""
    instructions = [
        "INSTRUCCIONES:",
        "Presiona 's' o ESPACIO para capturar imagen",
        "Presiona '1' para vista RAW",
        "Presiona '2' para vista con RECTANGULOS",
        "Presiona 'q' para salir",
        f"Imagenes capturadas: {image_count}",
        f"Vista actual: {view_mode}"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        # Fondo negro semitransparente para mejor legibilidad
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(frame, (5, y_offset + i * 25 - text_height - 2), 
                     (5 + text_width + 5, y_offset + i * 25 + 2), (0, 0, 0), -1)
        
        cv2.putText(frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


def main():
    """Función principal para mostrar la cámara y capturar imágenes."""
    print("=" * 60)
    print("CAPTURA DE IMAGENES CON CAMBIO DE VISTAS")
    print("=" * 60)
    
    # Inicializar OpenNI2
    if not initialize_openni2():
        return
    
    # Intentar abrir el dispositivo
    try:
        print("\nBuscando dispositivo OpenNI2...")
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
        return
    
    # Verificar si el sensor de color está disponible
    if not device.has_sensor(openni2.SENSOR_COLOR):
        print("=" * 60)
        print("ERROR: El dispositivo no tiene sensor de color disponible")
        print("=" * 60)
        openni2.unload()
        return
    
    # Iniciar el stream de color
    try:
        color_stream = device.create_color_stream()
        
        # Configurar parámetros de la cámara antes de iniciar
        configurar_camara_kinect(color_stream)
        
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
        print("=" * 60)
        openni2.unload()
        return
    
    # Cargar calibración
    coordenadas = load_calibration()
    
    # Crear directorio de salida
    output_dir = create_output_directory()
    print(f"\n✓ Las imágenes se guardarán en: {os.path.abspath(output_dir)}")
    
    # Obtener el siguiente número de imagen
    image_number = get_next_image_number(output_dir)
    image_count = image_number - 1
    
    # Modo de vista inicial
    view_mode = "RAW"  # "RAW" o "RECTANGLES"
    
    # Escenarios disponibles (puedes modificar esta lista)
    escenarios_disponibles = ["Escenario 1", "Escenario 2", "Escenario 3", "Escenario 4", "Escenario 5"]
    
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("  - Presiona 's' o ESPACIO para capturar una imagen")
    print("  - Presiona '1' para vista RAW (sin overlays)")
    print("  - Presiona '2' para vista con RECTANGULOS (mostrado en segunda pantalla)")
    print("  - Presiona 'q' para salir")
    print("=" * 60)
    print("\nMostrando feed de la cámara...")
    
    # Crear ventana principal (primera pantalla) para la cámara
    window_name = "Kinect Camera - Captura con Vistas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Crear ventana para la segunda pantalla (rectángulos)
    videobeam_window_name = "Vista Rectangulos - Segunda Pantalla"
    videobeam_window_created = False
    rectangulos_screen = None
    
    try:
        while True:
            # Leer frame de color
            try:
                frame = color_stream.read_frame()
                frame_data = frame.get_buffer_as_uint8()
                frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.flip(frame_bgr, 1)  # Voltear horizontalmente
            except Exception as e:
                print(f"Error al leer frame: {e}")
                break
            
            # Mostrar frame de la cámara (siempre en RAW en la primera pantalla)
            display_frame = frame_bgr.copy()
            draw_instructions(display_frame, image_count, view_mode)
            cv2.imshow(window_name, display_frame)
            
            # Si está en modo RECTANGLES, mostrar en la segunda pantalla
            if view_mode == "RECTANGLES" and coordenadas is not None:
                # Crear o actualizar la pantalla de rectángulos
                rectangulos_screen = create_rectangles_view_screen(coordenadas, escenarios_disponibles)
                
                if rectangulos_screen is not None:
                    # Crear ventana en segunda pantalla si no existe
                    if not videobeam_window_created:
                        cv2.namedWindow(videobeam_window_name, cv2.WINDOW_NORMAL)
                        cv2.moveWindow(videobeam_window_name, 1920, 0)
                        cv2.waitKey(50)
                        cv2.setWindowProperty(videobeam_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.waitKey(50)
                        videobeam_window_created = True
                    
                    # Escalar y mostrar en la segunda pantalla
                    rectangulos_screen_scaled = scale_to_videobeam(rectangulos_screen)
                    cv2.imshow(videobeam_window_name, rectangulos_screen_scaled)
                    
                    # Asegurar que esté en pantalla completa
                    try:
                        cv2.moveWindow(videobeam_window_name, 1920, 0)
                        cv2.setWindowProperty(videobeam_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    except:
                        pass
                else:
                    # Si no se pudo crear, cerrar la ventana de segunda pantalla
                    if videobeam_window_created:
                        try:
                            cv2.destroyWindow(videobeam_window_name)
                        except:
                            pass
                        videobeam_window_created = False
            else:
                # Si no está en modo RECTANGLES, cerrar la ventana de segunda pantalla
                if videobeam_window_created:
                    try:
                        cv2.destroyWindow(videobeam_window_name)
                    except:
                        pass
                    videobeam_window_created = False
            
            # Leer tecla presionada
            key = cv2.waitKey(1) & 0xFF
            
            # Cambiar a vista RAW
            if key == ord('1'):
                view_mode = "RAW"
                print(f"✓ Vista cambiada a: {view_mode}")
                if videobeam_window_created:
                    try:
                        cv2.destroyWindow(videobeam_window_name)
                    except:
                        pass
                    videobeam_window_created = False
            
            # Cambiar a vista con rectángulos
            elif key == ord('2'):
                if coordenadas is not None:
                    view_mode = "RECTANGLES"
                    print(f"✓ Vista cambiada a: {view_mode}")
                    print("  La vista de rectángulos se muestra en la segunda pantalla")
                else:
                    print("⚠ No se puede cambiar a vista RECTANGLES: calibración no disponible")
            
            # Capturar imagen con 's' o espacio
            elif key == ord('s') or key == ord(' '):
                # Generar nombre de archivo con timestamp y modo de vista
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                view_suffix = view_mode.lower()
                filename = f"image_{image_number:04d}_{view_suffix}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Guardar la imagen de la cámara (siempre RAW, sin overlays)
                cv2.imwrite(filepath, frame_bgr)
                print(f"✓ Imagen guardada: {filename} (vista: {view_mode})")
                
                image_number += 1
                image_count += 1
                
                # Mostrar confirmación en el frame
                cv2.putText(display_frame, "IMAGEN GUARDADA!", (200, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(500)  # Mostrar mensaje por 500ms
            
            # Salir con 'q'
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nCaptura cancelada por el usuario.")
    except Exception as e:
        print(f"\nERROR durante la captura: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpiar recursos
        try:
            color_stream.stop()
        except:
            pass
        # Cerrar todas las ventanas
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
        if videobeam_window_created:
            try:
                cv2.destroyWindow(videobeam_window_name)
            except:
                pass
        cv2.destroyAllWindows()
        openni2.unload()
        print(f"\n✓ Total de imágenes capturadas en esta sesión: {image_count}")
        print(f"✓ Próxima imagen será: image_{image_number:04d}_...")
        print("\nPuedes usar estas imágenes para análisis o entrenamiento de modelos.")


if __name__ == "__main__":
    main()

