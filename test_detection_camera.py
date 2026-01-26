# -*- coding: utf-8 -*-
"""
Script para capturar imágenes del Kinect para entrenar un modelo de detección.
Muestra el feed de la cámara y permite tomar fotos presionando 's' o espacio.
Las imágenes se guardan en la carpeta 'dataset/images' para su posterior etiquetado.
"""

import cv2
import numpy as np
from openni import openni2
import os
import sys
import io
from datetime import datetime

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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


def create_output_directory():
    """Crea el directorio para guardar las imágenes si no existe."""
    output_dir = "dataset/images"
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


def draw_instructions(frame, image_count):
    """Dibuja instrucciones en el frame."""
    instructions = [
        "INSTRUCCIONES:",
        "Presiona 's' o ESPACIO para capturar imagen",
        "Presiona 'q' para salir",
        f"Imagenes capturadas: {image_count}"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, y_offset + i * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    """Función principal para mostrar la cámara y capturar imágenes."""
    print("=" * 60)
    print("CAPTURA DE IMAGENES PARA ENTRENAMIENTO DE MODELO")
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
    
    # Crear directorio de salida
    output_dir = create_output_directory()
    print(f"\n✓ Las imágenes se guardarán en: {os.path.abspath(output_dir)}")
    
    # Obtener el siguiente número de imagen
    image_number = get_next_image_number(output_dir)
    image_count = image_number - 1
    
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("  - Presiona 's' o ESPACIO para capturar una imagen")
    print("  - Presiona 'q' para salir")
    print("=" * 60)
    print("\nMostrando feed de la cámara...")
    
    # Crear ventana
    cv2.namedWindow("Kinect Camera - Captura de Imagenes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kinect Camera - Captura de Imagenes", 1280, 720)
    
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
            
            # Crear una copia para mostrar instrucciones
            display_frame = frame_bgr.copy()
            draw_instructions(display_frame, image_count)
            
            # Mostrar el frame
            cv2.imshow("Kinect Camera - Captura de Imagenes", display_frame)
            
            # Leer tecla presionada
            key = cv2.waitKey(1) & 0xFF
            
            # Capturar imagen con 's' o espacio
            if key == ord('s') or key == ord(' '):
                # Generar nombre de archivo con timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{image_number:04d}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Guardar imagen
                cv2.imwrite(filepath, frame_bgr)
                print(f"✓ Imagen guardada: {filename}")
                
                image_number += 1
                image_count += 1
                
                # Mostrar confirmación en el frame
                cv2.putText(display_frame, "IMAGEN GUARDADA!", (200, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Kinect Camera - Captura de Imagenes", display_frame)
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
        cv2.destroyAllWindows()
        openni2.unload()
        print(f"\n✓ Total de imágenes capturadas en esta sesión: {image_count}")
        print(f"✓ Próxima imagen será: image_{image_number:04d}_...")
        print("\nPuedes usar estas imágenes para etiquetar y entrenar tu modelo.")


if __name__ == "__main__":
    main()

