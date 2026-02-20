# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
import random
import json
import threading
import pygame
import speech_recognition as sr
import pyaudio
from sentence_transformers import SentenceTransformer
import pyttsx3

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Advertencia: PIL/Pillow no está disponible. Los caracteres acentuados pueden no mostrarse correctamente.")


def put_text_safe(img, text, position, font_face, font_scale, color, thickness, line_type=cv2.LINE_AA):
    """
    Función auxiliar para renderizar texto con caracteres acentuados de forma segura.
    Si PIL está disponible y el texto contiene caracteres especiales, usa PIL.
    De lo contrario, usa OpenCV directamente.
    Modifica la imagen in-place.
    """
    # Verificar si el texto contiene caracteres acentuados o especiales
    has_special_chars = any(ord(c) > 127 for c in text)
    
    if PIL_AVAILABLE and has_special_chars:
        try:
            # Convertir imagen OpenCV a PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Intentar usar una fuente que soporte UTF-8 y que sea similar a FONT_HERSHEY_DUPLEX
            # En Windows, usar Arial o similar
            try:
                if os.name == 'nt':  # Windows
                    font_path = "C:/Windows/Fonts/arial.ttf"
                    if not os.path.exists(font_path):
                        font_path = "C:/Windows/Fonts/calibri.ttf"
                    if os.path.exists(font_path):
                        # Calcular tamaño de fuente basado en font_scale para que coincida con OpenCV
                        # FONT_HERSHEY_DUPLEX con scale 0.9 renderiza visualmente como ~22-24px
                        # Aumentar el tamaño para que coincida mejor visualmente
                        font_size = max(int(22 * font_scale), 16)  # Aumentado para coincidir mejor
                        font = ImageFont.truetype(font_path, font_size)
                    else:
                        font = ImageFont.load_default()
                else:  # Linux/Mac
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Obtener el tamaño real del texto con PIL para centrado correcto
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width_pil = bbox[2] - bbox[0]
            
            # Obtener tamaño con OpenCV para comparar y ajustar centrado
            text_size_cv, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_width_cv = text_size_cv[0]
            
            # Ajustar posición x para centrado correcto basado en el ancho real de PIL
            x, y = position
            # Calcular la diferencia y ajustar para mantener el centrado
            width_diff = text_width_pil - text_width_cv
            x = x - width_diff // 2
            
            # Renderizar texto
            # Convertir color BGR a RGB
            color_rgb = (color[2], color[1], color[0])
            draw.text((x, y), text, fill=color_rgb, font=font)
            
            # Convertir de vuelta a OpenCV y copiar a la imagen original
            img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img[:] = img_result[:]
        except Exception as e:
            print(f"Error al renderizar texto con PIL: {e}, usando OpenCV")
            # Fallback a OpenCV
            cv2.putText(img, text, position, font_face, font_scale, color, thickness, line_type)
    else:
        # Usar OpenCV directamente si no hay caracteres especiales o PIL no está disponible
        cv2.putText(img, text, position, font_face, font_scale, color, thickness, line_type)


def draw_close_card(screen, view_width, view_height, elevated=False):
    """
    Dibuja una card redonda con X en el centro, estilo infantil, roja con X blanca.
    Similar a la función en niveles_clasificacion.py
    
    Args:
        screen: Pantalla donde dibujar
        view_width: Ancho de la vista
        view_height: Alto de la vista
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
    
    return card_center, card_radius


def draw_logo(screen, view_width, view_height):
    """
    Dibuja el logo en la parte superior de la pantalla.
    Similar a la función en main.py
    """
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


def juego_absurdos_reconocimiento_voz(device, coordenadas, dmax_map, dmin_map, model=None):
    """
    Juego de reconocimiento de voz para absurdos lógicos.
    Muestra una imagen aleatoria y el usuario debe describir qué está mal usando voz.
    """
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
        return cv2.resize(image, (VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Cargar configuración de absurdos
    json_path = "absurdos-visuales/absurdos_logicos/config/absurdos.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        absurdos_list = datos.get("absurdos", [])
        if not absurdos_list:
            print("Error: No se encontraron absurdos en el archivo JSON")
            return
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {json_path} no es un JSON válido")
        return
    
    # Crear ventana una sola vez (fuera del bucle)
    window_name = "Juego de Reconocimiento de Voz"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1920, 0)
    
    # Variable para controlar si es la primera imagen (solo TTS en la primera)
    primera_imagen = True
    
    # Bucle principal del juego - continuar hasta que el usuario presione "Volver al Menú"
    while True:
        # Reconfigurar ventana en cada iteración para asegurar tamaño correcto
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        
        # Seleccionar un absurdo aleatorio
        absurdo_actual = random.choice(absurdos_list)
        imagen_nombre = absurdo_actual.get("imagen", "")
        
        # Construir ruta de imagen
        assets_path = "absurdos-visuales/absurdos_logicos/assets"
        ruta_imagen = os.path.join(assets_path, imagen_nombre)
        
        # Función para normalizar nombres de archivo (remover extensiones, guiones, guiones bajos, espacios, convertir a minúsculas)
        def normalizar_nombre(nombre):
            # Remover extensión
            nombre_sin_ext = os.path.splitext(nombre)[0]
            # Normalizar: convertir a minúsculas, reemplazar guiones y guiones bajos con nada, remover espacios
            nombre_normalizado = nombre_sin_ext.lower().replace("-", "").replace("_", "").replace(" ", "")
            return nombre_normalizado
        
        # Intentar encontrar la imagen con matching flexible
        if not os.path.exists(ruta_imagen):
            # Intentar con guiones en lugar de guiones bajos
            imagen_nombre_alt = imagen_nombre.replace("_", "-")
            ruta_imagen = os.path.join(assets_path, imagen_nombre_alt)
        
        # Si aún no existe, buscar archivos similares usando normalización
        if not os.path.exists(ruta_imagen):
            imagen_normalizada = normalizar_nombre(imagen_nombre)
            mejor_coincidencia = None
            mejor_puntuacion = 0
            
            # Obtener lista de archivos en el directorio
            if os.path.exists(assets_path):
                archivos_disponibles = os.listdir(assets_path)
                for archivo in archivos_disponibles:
                    # Solo procesar archivos de imagen
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        archivo_normalizado = normalizar_nombre(archivo)
                        
                        # Si los nombres normalizados son iguales
                        if imagen_normalizada == archivo_normalizado:
                            mejor_coincidencia = archivo
                            mejor_puntuacion = 100
                            break
                        
                        # Si uno contiene al otro (coincidencia parcial)
                        if imagen_normalizada in archivo_normalizado or archivo_normalizado in imagen_normalizada:
                            # Calcular puntuación basada en la longitud de la coincidencia
                            len_menor = min(len(imagen_normalizada), len(archivo_normalizado))
                            len_mayor = max(len(imagen_normalizada), len(archivo_normalizado))
                            puntuacion = (len_menor / len_mayor) * 100
                            if puntuacion > mejor_puntuacion:
                                mejor_coincidencia = archivo
                                mejor_puntuacion = puntuacion
                        
                        # Calcular similitud por caracteres comunes (método de Jaccard)
                        caracteres_imagen = set(imagen_normalizada)
                        caracteres_archivo = set(archivo_normalizado)
                        caracteres_comunes = caracteres_imagen & caracteres_archivo
                        caracteres_totales = caracteres_imagen | caracteres_archivo
                        
                        if len(caracteres_totales) > 0:
                            similitud_jaccard = len(caracteres_comunes) / len(caracteres_totales) * 100
                            # También considerar la longitud de la subsecuencia común más larga
                            # Extraer palabras clave comunes (palabras de 3+ caracteres)
                            palabras_imagen = [imagen_normalizada[i:i+3] for i in range(len(imagen_normalizada)-2)]
                            palabras_archivo = [archivo_normalizado[i:i+3] for i in range(len(archivo_normalizado)-2)]
                            palabras_comunes = set(palabras_imagen) & set(palabras_archivo)
                            
                            if len(palabras_comunes) > 0:
                                puntuacion_palabras = len(palabras_comunes) / max(len(palabras_imagen), len(palabras_archivo)) * 100
                                puntuacion_final = (similitud_jaccard * 0.6 + puntuacion_palabras * 0.4)
                            else:
                                puntuacion_final = similitud_jaccard
                            
                            if puntuacion_final > mejor_puntuacion and puntuacion_final > 40:  # Al menos 40% de similitud
                                mejor_coincidencia = archivo
                                mejor_puntuacion = puntuacion_final
            
            if mejor_coincidencia:
                ruta_imagen = os.path.join(assets_path, mejor_coincidencia)
                print(f"Imagen encontrada por coincidencia: {mejor_coincidencia} (buscando: {imagen_nombre})")
        
        if not os.path.exists(ruta_imagen):
            print(f"Error: No se encontró la imagen {imagen_nombre} en {assets_path}")
            print(f"Archivos disponibles: {', '.join([f for f in os.listdir(assets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])}")
            continue  # Continuar con el siguiente absurdo en el bucle while True
        
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
            continue  # Continuar con el siguiente absurdo
        
        # Crear pantalla del videobeam
        videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        
        # Crear degradado de colores (mismo estilo que otros juegos)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            videobeam_screen[y, :] = [b, g, r]
        
        # Dibujar logo en la parte superior
        draw_logo(videobeam_screen, view_width, view_height)
        
        # Dibujar botón X (cerrar) en la esquina superior derecha
        close_card_center, close_card_radius = draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
        
        # Redimensionar imagen para que quepa en la pantalla (más pequeña y arriba del botón)
        max_img_width = int(view_width * 0.5)  # Reducido de 0.7 a 0.5
        max_img_height = int(view_height * 0.4)  # Reducido de 0.6 a 0.4
        
        img_height, img_width = imagen.shape[:2]
        aspect_ratio = img_width / img_height
        
        if aspect_ratio > (max_img_width / max_img_height):
            new_width = max_img_width
            new_height = int(max_img_width / aspect_ratio)
        else:
            new_height = max_img_height
            new_width = int(max_img_height * aspect_ratio)
        
        imagen_resized = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calcular posición del botón primero (para posicionar la imagen arriba de él)
        circle_radius = 60
        circle_center_x = view_width // 2
        espacio_para_franja = 100  # Espacio necesario para la franja "Escuchando..." (50 arriba + 50 abajo)
        # Posición del botón: cerca de la parte inferior pero dejando espacio para la franja
        circle_center_y = view_height - espacio_para_franja - circle_radius
        
        # Posicionar la imagen arriba del botón con espacio
        espacio_entre_imagen_y_boton = 40  # Espacio entre la imagen y el botón
        img_y = circle_center_y - circle_radius - espacio_entre_imagen_y_boton - new_height
        # Asegurar que la imagen no se salga por arriba (dejar espacio para el logo)
        logo_height = 120
        logo_y = 30
        espacio_desde_logo = 20  # Espacio mínimo entre logo e imagen
        min_img_y = logo_y + logo_height + espacio_desde_logo
        img_y = max(img_y, min_img_y)  # Asegurar que esté debajo del logo
        
        # Centrar horizontalmente
        img_x = (view_width - new_width) // 2
        
        # Dibujar imagen (manejar transparencia si existe)
        if len(imagen_resized.shape) == 3 and imagen_resized.shape[2] == 4:
            # Imagen con canal alfa
            alpha = imagen_resized[:, :, 3] / 255.0
            img_bgr = imagen_resized[:, :, :3]
            for c in range(3):
                videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                    alpha * img_bgr[:, :, c] + (1 - alpha) * videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width, c]
                )
        else:
            videobeam_screen[img_y:img_y+new_height, img_x:img_x+new_width] = imagen_resized[:, :, :3]
        
        # Extraer coordenadas para touch detection (necesarias para detectar el botón)
        xw_min = coordenadas["xw_min"]
        xw_max = coordenadas["xw_max"]
        yw_min = coordenadas["yw_min"]
        yw_max = coordenadas["yw_max"]
        xv_min = coordenadas["xv_min"]
        xv_max = coordenadas["xv_max"]
        yv_min = coordenadas["yv_min"]
        yv_max = coordenadas["yv_max"]
        
        # Dibujar card circular en la parte inferior (debajo de la imagen) - SIEMPRE visible
        # La posición del círculo ya fue calculada arriba para posicionar la imagen correctamente
        
        # Dibujar círculo con sombra
        cv2.circle(videobeam_screen, (circle_center_x + 3, circle_center_y + 3), circle_radius, (0, 0, 0), -1)  # Sombra
        cv2.circle(videobeam_screen, (circle_center_x, circle_center_y), circle_radius, (0, 200, 0), -1)  # Verde
        cv2.circle(videobeam_screen, (circle_center_x, circle_center_y), circle_radius, (255, 255, 255), 3)  # Borde blanco
        
        # Agregar texto "Hablar" o icono de micrófono
        button_text = "Hablar"
        font_button = cv2.FONT_HERSHEY_DUPLEX
        font_scale_button = 0.7
        thickness_button = 2
        text_size_button, _ = cv2.getTextSize(button_text, font_button, font_scale_button, thickness_button)
        text_x_button = circle_center_x - text_size_button[0] // 2
        text_y_button = circle_center_y + text_size_button[1] // 2
        
        # Texto con sombra
        cv2.putText(videobeam_screen, button_text, (text_x_button + 1, text_y_button + 1), 
                   font_button, font_scale_button, (0, 0, 0), thickness_button + 1)
        cv2.putText(videobeam_screen, button_text, (text_x_button, text_y_button), 
                   font_button, font_scale_button, (255, 255, 255), thickness_button)
        
        # Mostrar pantalla con el botón circular (siempre visible)
        videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
        cv2.imshow(window_name, videobeam_screen_scaled)
        cv2.waitKey(50)
        
        # Inicializar TTS solo si es la primera imagen
        tts_finished = threading.Event()
        boton_habilitado = False  # El botón solo se habilita después del TTS
        
        if primera_imagen:
            try:
                engine_tts = pyttsx3.init()
                engine_tts.setProperty('rate', 150)  # Velocidad del habla
                # Intentar configurar voz en español
                voices = engine_tts.getProperty('voices')
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                        engine_tts.setProperty('voice', voice.id)
                        break
            except Exception as e:
                print(f"Advertencia: No se pudo inicializar TTS: {e}")
                engine_tts = None
            
            # Decir instrucción en un hilo separado (solo en la primera imagen)
            def decir_instruccion():
                if engine_tts:
                    try:
                        engine_tts.say("Ahora habla, di qué está mal")
                        engine_tts.runAndWait()
                    except Exception as e:
                        print(f"Error al decir instrucción: {e}")
                    finally:
                        tts_finished.set()
                else:
                    tts_finished.set()
            
            tts_thread = threading.Thread(target=decir_instruccion, daemon=True)
            tts_thread.start()
            
            # Esperar a que termine de hablar el TTS, actualizando la pantalla mientras tanto
            while not tts_finished.is_set():
                videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                cv2.imshow(window_name, videobeam_screen_scaled)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                cv2.waitKey(10)
                tts_finished.wait(timeout=0.1)  # Verificar cada 100ms
            
            # Esperar un poco más para asegurar que el audio se haya limpiado
            time.sleep(0.5)
            
            # Marcar que ya no es la primera imagen y habilitar el botón
            primera_imagen = False
            boton_habilitado = True
            
            # Actualizar pantalla para mostrar que el botón está habilitado
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            cv2.imshow(window_name, videobeam_screen_scaled)
            cv2.waitKey(10)
        else:
            # Si no es la primera imagen, el botón está habilitado inmediatamente
            boton_habilitado = True
        
        # Inicializar reconocimiento de voz (pero no escuchar todavía)
        try:
            recognizer = sr.Recognizer()
            # Configurar reconocedor para mejor precisión
            recognizer.energy_threshold = 300  # Umbral de energía inicial
            recognizer.dynamic_energy_threshold = True  # Ajustar dinámicamente
            recognizer.pause_threshold = 1.5  # Más tiempo de silencio después del habla para capturar mejor el final
            recognizer.operation_timeout = None  # Sin timeout en operaciones
            
            # Listar micrófonos disponibles para debugging
            print("Micrófonos disponibles:")
            for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {i}: {mic_name}")
            
            # Intentar usar el micrófono por defecto, o el primero disponible
            try:
                microphone = sr.Microphone()
            except Exception as e:
                print(f"Error al inicializar micrófono por defecto: {e}")
                # Intentar con el primer micrófono disponible
                microphone = sr.Microphone(pyaudio.get_device_count() - 1)
            
            print(f"Usando micrófono: {microphone}")
        except OSError as e:
            if "PyAudio" in str(e) or "pyaudio" in str(e).lower():
                print("Error: PyAudio no está instalado. Instalando...")
                print("Por favor ejecuta: uv pip install pyaudio")
                print("O si eso no funciona, instala desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            else:
                print(f"Error al inicializar reconocimiento de voz: {e}")
            cv2.destroyWindow(window_name)
            continue
        except Exception as e:
            print(f"Error al inicializar reconocimiento de voz: {e}")
            print("Asegúrate de que PyAudio esté instalado: uv pip install pyaudio")
            cv2.destroyWindow(window_name)
            continue
        
        # Función para detectar si se tocó el botón circular
        def detectar_boton_circular_tocado(cx_roi, cy_roi):
            # cx_roi y cy_roi son coordenadas relativas a depth_roi
            # Convertir coordenadas de ROI a viewport
            sx = float(xv_max - xv_min) / (xw_max - xw_min)
            sy = float(yv_max - yv_min) / (yw_max - yw_min)
            x_viewport = int(xv_min + (cx_roi * sx))
            y_viewport = int(yv_min + (cy_roi * sy))
            
            # Calcular distancia desde el centro del círculo
            distance = np.sqrt((x_viewport - circle_center_x)**2 + (y_viewport - circle_center_y)**2)
            
            # Si la distancia es menor que el radio, se tocó el botón
            if distance <= circle_radius:
                return True
            return False
        
        # Definir text_y para evitar errores (usado más adelante)
        text_y = view_height - 50
        
        # Cargar modelo ANTES del bucle de intentos (solo una vez)
        modelo_cargado_correctamente = True
        if model is None:
            try:
                # Usar modelo multilingüe que soporta español
                print("Cargando modelo de sentence-transformers...")
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ Modelo de sentence-transformers cargado correctamente")
            except Exception as e:
                print(f"Error al cargar modelo de sentence-transformers: {e}")
                print("Intentando cargar modelo nuevamente...")
                try:
                    # Reintentar con manejo de errores de red
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            print("✓ Modelo de sentence-transformers cargado correctamente")
                            break
                        except Exception as e2:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 2  # Esperar 2, 4, 6 segundos
                                print(f"Intento {attempt + 1} fallido. Reintentando en {wait_time} segundos...")
                                time.sleep(wait_time)
                            else:
                                print(f"Error al descargar/cargar modelo después de {max_retries} intentos: {e2}")
                                print("El juego continuará pero no podrá evaluar respuestas correctamente.")
                                cv2.destroyWindow(window_name)
                                modelo_cargado_correctamente = False
                except Exception as e3:
                    print(f"Error crítico al cargar modelo: {e3}")
                    cv2.destroyWindow(window_name)
                    modelo_cargado_correctamente = False
        else:
            print("✓ Usando modelo de sentence-transformers pre-cargado")
        
        # Si no se pudo cargar el modelo, continuar con el siguiente absurdo
        if not modelo_cargado_correctamente:
            continue
        
        # Cargar respuestas correctas ANTES del bucle (solo una vez por imagen)
        respuestas_json_path = "absurdos-visuales/absurdos_logicos/config/respuestas_correctas.json"
        respuestas_correctas = []
        try:
            with open(respuestas_json_path, "r", encoding="utf-8") as f:
                datos_respuestas = json.load(f)
                # Buscar respuestas para el absurdo actual
                absurdo_id = absurdo_actual.get("id")
                imagen_actual = absurdo_actual.get("imagen", "")
                for respuesta_item in datos_respuestas.get("respuestas", []):
                    if respuesta_item.get("id") == absurdo_id or respuesta_item.get("imagen") == imagen_actual:
                        respuestas_correctas = respuesta_item.get("respuestas_correctas", [])
                        break
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {respuestas_json_path}")
            respuestas_correctas = []
        except Exception as e:
            print(f"Error al cargar respuestas correctas: {e}")
            respuestas_correctas = []
        
        # Bucle de intentos para esta imagen (máximo 3 intentos)
        intentos_restantes = 3
        respuesta_correcta = False
        
        while intentos_restantes > 0 and not respuesta_correcta:
            # Iniciar streams para detección de toques
            rgb_stream = device.create_color_stream()
            depth_stream = device.create_depth_stream()
            rgb_stream.start()
            depth_stream.start()
            
            # Esperar a que se toque el botón circular
            boton_presionado = False
            button_pressed_flag = False
            audio = None
            
            print("Esperando a que presiones el botón circular para empezar a escuchar...")
            
            print("Listo. Presiona el botón para empezar a escuchar...")
            
            while True:
                frame = rgb_stream.read_frame()
                depth_frame = depth_stream.read_frame()
                
                if frame is None or depth_frame is None:
                    continue
                
                rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                depth_data = cv2.flip(depth_data, 1)
                depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
                
                # Crear máscara de toques
                touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Verificar si hay toques en el botón (solo si está habilitado)
                hay_toque_en_boton = False
                hay_toque_en_close = False
                current_screen = videobeam_screen.copy()
                
                # Primero verificar si hay toque en el botón X antes de dibujarlo
                # (para mostrar el efecto de elevación)
                close_card_touched = False
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            
                            # Convertir coordenadas de ROI a viewport para detectar toque en X
                            sx = float(xv_max - xv_min) / (xw_max - xw_min)
                            sy = float(yv_max - yv_min) / (yw_max - yw_min)
                            x_viewport_touch = int(xv_min + (cx * sx))
                            y_viewport_touch = int(yv_min + (cy * sy))
                            
                            # Calcular posición del botón X para verificar toque
                            base_card_radius = 50
                            card_margin_x = 180
                            card_margin_y = 80
                            close_card_center_temp = (view_width - card_margin_x - base_card_radius, 
                                                     card_margin_y + base_card_radius)
                            
                            # Verificar si se tocó el botón X (cerrar)
                            distance_to_close = np.sqrt((x_viewport_touch - close_card_center_temp[0])**2 + 
                                                       (y_viewport_touch - close_card_center_temp[1])**2)
                            if distance_to_close <= base_card_radius:
                                close_card_touched = True
                                break
                
                # Dibujar botón X (cerrar) en la esquina superior derecha con efecto de elevación si se toca
                close_card_center, close_card_radius = draw_close_card(current_screen, view_width, view_height, elevated=close_card_touched)
                
                # SIEMPRE dibujar el botón en la pantalla (habilitado o no)
                if boton_habilitado:
                    # Botón habilitado - color verde normal
                    cv2.circle(current_screen, (circle_center_x + 3, circle_center_y + 3), circle_radius, (0, 0, 0), -1)  # Sombra
                    cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (0, 200, 0), -1)  # Verde
                    cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (255, 255, 255), 3)  # Borde blanco
                else:
                    # Botón no habilitado - color gris
                    cv2.circle(current_screen, (circle_center_x + 3, circle_center_y + 3), circle_radius, (0, 0, 0), -1)  # Sombra
                    cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (100, 100, 100), -1)  # Gris
                    cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (255, 255, 255), 3)  # Borde blanco
                
                cv2.putText(current_screen, button_text, (text_x_button + 1, text_y_button + 1), 
                           font_button, font_scale_button, (0, 0, 0), thickness_button + 1)
                cv2.putText(current_screen, button_text, (text_x_button, text_y_button), 
                           font_button, font_scale_button, (255, 255, 255), thickness_button)
                
                # Procesar toques (botón circular y botón X)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            
                            # Convertir coordenadas de ROI a viewport para detectar toque en X
                            sx = float(xv_max - xv_min) / (xw_max - xw_min)
                            sy = float(yv_max - yv_min) / (yw_max - yw_min)
                            x_viewport_touch = int(xv_min + (cx * sx))
                            y_viewport_touch = int(yv_min + (cy * sy))
                            
                            # Verificar si se tocó el botón X (cerrar)
                            # Usar la posición base del botón (sin elevación) para la detección
                            base_card_radius = 50
                            card_margin_x = 180
                            card_margin_y = 80
                            close_card_center_base = (view_width - card_margin_x - base_card_radius, 
                                                     card_margin_y + base_card_radius)
                            
                            distance_to_close = np.sqrt((x_viewport_touch - close_card_center_base[0])**2 + 
                                                       (y_viewport_touch - close_card_center_base[1])**2)
                            if distance_to_close <= base_card_radius:
                                hay_toque_en_close = True
                                # Mostrar efecto de elevación y luego salir
                                print("Botón X (cerrar) presionado. Volviendo al menú principal...")
                                
                                # Redibujar con efecto de elevación (usar current_screen que ya tiene todos los elementos)
                                current_screen_elevated = current_screen.copy()
                                # Redibujar solo el botón X con efecto de elevación
                                draw_close_card(current_screen_elevated, view_width, view_height, elevated=True)
                                
                                # Mostrar el efecto de elevación brevemente
                                videobeam_screen_scaled_elevated = scale_to_videobeam(current_screen_elevated)
                                cv2.imshow(window_name, videobeam_screen_scaled_elevated)
                                cv2.waitKey(200)  # Mostrar el efecto por 200ms
                                
                                # Detener streams
                                try:
                                    rgb_stream.stop()
                                    depth_stream.stop()
                                except:
                                    pass
                                # Cerrar ventana y salir
                                try:
                                    cv2.destroyWindow(window_name)
                                except:
                                    pass
                                return  # Volver al menú principal
                            
                            # Verificar si se tocó el botón circular (solo si está habilitado)
                            if boton_habilitado and detectar_boton_circular_tocado(cx, cy):
                                hay_toque_en_boton = True
                                
                                if not button_pressed_flag:
                                        # Botón recién presionado - activar micrófono
                                        button_pressed_flag = True
                                        print("Botón presionado. Activando micrófono...")
                                        
                                        # Cambiar color del botón cuando se presiona (verde más oscuro)
                                        cv2.circle(current_screen, (circle_center_x + 3, circle_center_y + 3), circle_radius, (0, 0, 0), -1)  # Sombra
                                        cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (0, 150, 0), -1)  # Verde más oscuro
                                        cv2.circle(current_screen, (circle_center_x, circle_center_y), circle_radius, (255, 255, 255), 3)  # Borde blanco
                                        cv2.putText(current_screen, button_text, (text_x_button + 1, text_y_button + 1), 
                                                   font_button, font_scale_button, (0, 0, 0), thickness_button + 1)
                                        cv2.putText(current_screen, button_text, (text_x_button, text_y_button), 
                                                   font_button, font_scale_button, (255, 255, 255), thickness_button)
                                        
                                        # Agregar franja "Escuchando..." debajo del botón
                                        mensaje_escuchando = "Escuchando..."
                                        font_escuchando = cv2.FONT_HERSHEY_DUPLEX
                                        font_scale_escuchando = 0.9
                                        thickness_escuchando = 2
                                        text_size_escuchando, _ = cv2.getTextSize(mensaje_escuchando, font_escuchando, font_scale_escuchando, thickness_escuchando)
                                        text_x_escuchando = (view_width - text_size_escuchando[0]) // 2
                                        text_y_escuchando = circle_center_y + circle_radius + 50  # Debajo del botón
                                        
                                        # Color azul oscuro para indicar que está escuchando
                                        color_fondo_escuchando = (200, 100, 0)  # Azul oscuro (en BGR)
                                        
                                        # Dibujar franja de fondo con color azul (siempre en la parte inferior)
                                        franja_y_inicio = text_y_escuchando - 25
                                        franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
                                        cv2.rectangle(current_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                                     color_fondo_escuchando, -1)  # Fondo azul
                                        cv2.rectangle(current_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                                     (255, 255, 255), 2)  # Borde blanco
                                        
                                        # Texto siempre en blanco (usar función segura para caracteres acentuados)
                                        put_text_safe(current_screen, mensaje_escuchando, (text_x_escuchando + 2, text_y_escuchando + 2), 
                                                   font_escuchando, font_scale_escuchando, (0, 0, 0), thickness_escuchando + 1)  # Sombra negra
                                        put_text_safe(current_screen, mensaje_escuchando, (text_x_escuchando, text_y_escuchando), 
                                                   font_escuchando, font_scale_escuchando, (255, 255, 255), thickness_escuchando)  # Texto blanco
                                        
                                        # Mostrar pantalla actualizada
                                        videobeam_screen_scaled = scale_to_videobeam(current_screen)
                                        cv2.imshow(window_name, videobeam_screen_scaled)
                                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                        cv2.waitKey(200)
                                        
                                        # Detener streams antes de escuchar
                                        rgb_stream.stop()
                                        depth_stream.stop()
                                        
                                        # Ahora escuchar audio (no necesita mantener presionado)
                                        boton_presionado = True
                                        break
                
                # Si el botón fue presionado, salir del bucle de detección y empezar a escuchar
                if boton_presionado:
                    break
                
                # Mostrar pantalla (usar current_screen que tiene el estado actual del botón)
                videobeam_screen_scaled = scale_to_videobeam(current_screen)
                cv2.imshow(window_name, videobeam_screen_scaled)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                cv2.waitKey(10)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rgb_stream.stop()
                    depth_stream.stop()
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass
                    return
            
            # Si el botón fue presionado, ahora sí escuchar audio
            audio = None
            
            if boton_presionado:
                # Ajustar para ruido ambiente rápidamente antes de escuchar
                print("Ajustando para ruido ambiente antes de escuchar...")
                try:
                    with microphone as source:
                        print("Calibrando micrófono (1 segundo)...")
                        recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Calibración
                        print(f"✓ Umbral de energía ajustado: {recognizer.energy_threshold}")
                except Exception as e:
                    print(f"⚠ Error al ajustar ruido ambiente: {e}")
                    # Continuar de todas formas
                
                # Escuchar audio con detección de silencio
                print("=" * 50)
                print("INICIANDO CAPTURA DE AUDIO")
                print("=" * 50)
                print("Esperando audio (timeout: 20 segundos, límite de frase: 25 segundos)...")
                print("Habla ahora...")
                
                try:
                    with microphone as source:
                        # Escuchar hasta que detecte que terminó de hablar
                        audio = recognizer.listen(source, timeout=20, phrase_time_limit=25)
                        if audio:
                            duracion = len(audio.frame_data) / audio.sample_rate
                            print(f"✓ Audio capturado exitosamente!")
                            print(f"  Duración: {duracion:.2f} segundos")
                            print(f"  Tamaño de datos: {len(audio.frame_data)} bytes")
                            print(f"  Sample rate: {audio.sample_rate} Hz")
                        else:
                            print("✗ No se capturó audio (audio es None)")
                            audio = None
                except sr.WaitTimeoutError:
                    print("✗ Tiempo de espera agotado. No se detectó ningún audio.")
                    print("  Verifica que:")
                    print("  - El micrófono esté funcionando")
                    print("  - El micrófono no esté silenciado")
                    print("  - Estés hablando lo suficientemente fuerte")
                    audio = None
                except Exception as e:
                    print(f"✗ Error al escuchar el micrófono: {e}")
                    print(f"  Tipo de error: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    audio = None
            else:
                # Si no se presionó el botón, continuar con el siguiente intento
                print("Botón no presionado. Intentando de nuevo...")
                intentos_restantes -= 1
                continue
        
            if not boton_presionado or audio is None:
                # Si no se presionó el botón o no se capturó audio, continuar con el siguiente intento
                print("No se capturó audio. Intentando de nuevo...")
                intentos_restantes -= 1
                continue
        
            # Reconocer el audio
            if audio is not None:
                print("Procesando audio con Google Speech Recognition...")
                try:
                    # Usar configuración mejorada para mejor precisión
                    texto_reconocido = recognizer.recognize_google(
                        audio, 
                        language="es-ES",
                        show_all=False  # Solo devolver el mejor resultado
                    )
                    print(f"Texto reconocido: {texto_reconocido}")
                except sr.UnknownValueError:
                    print("No se pudo entender el audio. El audio podría estar muy bajo o no contener palabras claras.")
                    texto_reconocido = None
                except sr.RequestError as e:
                    print(f"Error al conectar con el servicio de reconocimiento: {e}")
                    print("Verifica tu conexión a internet.")
                    texto_reconocido = None
                except Exception as e:
                    print(f"Error inesperado en reconocimiento: {e}")
                    import traceback
                    traceback.print_exc()
                    texto_reconocido = None
            else:
                print("No se capturó audio. No se puede transcribir.")
                texto_reconocido = None
            
            # Evaluar respuesta usando sentence-transformers (DENTRO del bucle de intentos)
            es_correcta = False
            mensaje_resultado = ""
            SIMILARITY_THRESHOLD = 0.7
            
            if texto_reconocido and model:
                if respuestas_correctas:
                    try:
                        # Crear embeddings para el texto del usuario
                        texto_usuario_embedding = model.encode([texto_reconocido])[0]  # Obtener el vector único
                        
                        # Crear embeddings para todas las respuestas correctas
                        respuestas_embeddings = model.encode(respuestas_correctas)
                        
                        # Calcular similitud coseno entre el texto del usuario y cada respuesta correcta
                        # Usar numpy para calcular cosine similarity
                        def cosine_similarity_numpy(vec1, vec2_array):
                            """Calcula la similitud coseno entre un vector y un array de vectores usando numpy"""
                            # vec1: vector 1D, vec2_array: array 2D de vectores
                            dot_products = np.dot(vec2_array, vec1)
                            norm1 = np.linalg.norm(vec1)
                            norms2 = np.linalg.norm(vec2_array, axis=1)
                            return dot_products / (norm1 * norms2)
                        
                        similarities = cosine_similarity_numpy(texto_usuario_embedding, respuestas_embeddings)
                        
                        # Encontrar la similitud máxima
                        max_similarity = float(np.max(similarities))
                        
                        print(f"Similitud máxima encontrada: {max_similarity:.3f}")
                        
                        # Si la similitud máxima supera el umbral, la respuesta es correcta
                        if max_similarity >= SIMILARITY_THRESHOLD:
                            es_correcta = True
                            mensaje_resultado = "¡Correcto!"
                        else:
                            es_correcta = False
                            mensaje_resultado = "Inténtalo de nuevo"
                    except Exception as e:
                        print(f"Error al evaluar respuesta con sentence-transformers: {e}")
                        es_correcta = False
                        mensaje_resultado = "Error al evaluar"
                else:
                    print("Advertencia: No se encontraron respuestas correctas para comparar")
                    es_correcta = False
                    mensaje_resultado = "Inténtalo de nuevo"
            else:
                if not texto_reconocido:
                    mensaje_resultado = "No se detectó audio"
                else:
                    mensaje_resultado = "Error al cargar modelo"
            
            # Mostrar resultado (DENTRO del bucle de intentos)
            # TODOS los mensajes de feedback se muestran en el área inferior (misma posición que "Escuchando...")
            feedback_y_position = circle_center_y + circle_radius + 50  # Misma posición que "Escuchando..."
            
            # Determinar color de fondo según el tipo de feedback
            if es_correcta:
                # Verde para feedback positivo/correcto
                color_fondo = (0, 200, 0)  # Verde oscuro para el fondo
            elif "Error" in mensaje_resultado or "Error al cargar" in mensaje_resultado:
                # Naranja para errores del sistema
                color_fondo = (0, 140, 255)  # Naranja oscuro para el fondo
            elif "No se detectó" in mensaje_resultado:
                # Amarillo para advertencias
                color_fondo = (0, 200, 255)  # Amarillo oscuro para el fondo
            else:
                # Rojo para respuestas incorrectas
                color_fondo = (0, 0, 200)  # Rojo oscuro para el fondo
            
            font_resultado = cv2.FONT_HERSHEY_DUPLEX
            font_scale_resultado = 0.9
            thickness_resultado = 2
            
            text_size_resultado, _ = cv2.getTextSize(mensaje_resultado, font_resultado, font_scale_resultado, thickness_resultado)
            text_x_resultado = (view_width - text_size_resultado[0]) // 2
            text_y_resultado = feedback_y_position
            
            # Dibujar franja de fondo con color según el tipo de feedback (siempre en la parte inferior)
            franja_y_inicio = text_y_resultado - 25
            franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         color_fondo, -1)  # Fondo con color según feedback
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         (255, 255, 255), 2)  # Borde blanco
            
            # Texto siempre en blanco (usar función segura para caracteres acentuados)
            put_text_safe(videobeam_screen, mensaje_resultado, (text_x_resultado + 2, text_y_resultado + 2), 
                       font_resultado, font_scale_resultado, (0, 0, 0), thickness_resultado + 1)  # Sombra negra
            put_text_safe(videobeam_screen, mensaje_resultado, (text_x_resultado, text_y_resultado), 
                       font_resultado, font_scale_resultado, (255, 255, 255), thickness_resultado)  # Texto blanco
            
            # Dibujar botón X (cerrar) en la esquina superior derecha
            draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
            
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            cv2.imshow(window_name, videobeam_screen_scaled)
            # Asegurar que la ventana esté configurada correctamente
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
            cv2.waitKey(10)
            
            # Reproducir sonido de retroalimentación
            pygame.mixer.init()
            try:
                if es_correcta:
                    sonido_path = "sounds/correct.mp3"
                    if not os.path.exists(sonido_path):
                        sonido_path = "sounds/correcto.wav"
                else:
                    sonido_path = "sounds/incorrect.mp3"
                    if not os.path.exists(sonido_path):
                        sonido_path = "sounds/incorrecto.mp3"
                
                if os.path.exists(sonido_path):
                    try:
                        sonido = pygame.mixer.Sound(sonido_path)
                        sonido.play()
                        time.sleep(1.5)  # Esperar a que termine el sonido
                    except Exception as e:
                        print(f"Error al reproducir sonido: {e}")
            except Exception as e:
                print(f"Error al inicializar mixer: {e}")
            
            # Si la respuesta es correcta, mostrar pantalla de éxito
            if es_correcta:
                respuesta_correcta = True
                # Detener streams antes de mostrar pantalla de éxito
                try:
                    rgb_stream.stop()
                    depth_stream.stop()
                except:
                    pass
                
                # Extraer coordenadas para touch detection
                xw_min = coordenadas["xw_min"]
                xw_max = coordenadas["xw_max"]
                yw_min = coordenadas["yw_min"]
                yw_max = coordenadas["yw_max"]
                xv_min = coordenadas["xv_min"]
                xv_max = coordenadas["xv_max"]
                yv_min = coordenadas["yv_min"]
                yv_max = coordenadas["yv_max"]
                
                # Crear pantalla de éxito con opacidad negra
                success_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                
                # Crear degradado de fondo
                for y in range(view_height):
                    ratio = y / view_height
                    r = int(255 * (0.3 + 0.4 * ratio))
                    g = int(200 * (0.5 + 0.3 * ratio))
                    b = int(255 * (0.8 - 0.3 * ratio))
                    success_screen[y, :] = [b, g, r]
                
                # Dibujar imagen original con opacidad negra (0.3 para oscurecer más)
                opacity = 0.3
                if len(imagen_resized.shape) == 3 and imagen_resized.shape[2] == 4:
                    # Imagen con canal alfa
                    alpha_original = imagen_resized[:, :, 3] / 255.0
                    img_bgr = imagen_resized[:, :, :3]
                    for c in range(3):
                        success_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                            (1 - opacity) * success_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] +
                            opacity * (alpha_original * img_bgr[:, :, c] + (1 - alpha_original) * success_screen[img_y:img_y+new_height, img_x:img_x+new_width, c])
                        )
                else:
                    for c in range(3):
                        success_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] = (
                            (1 - opacity) * success_screen[img_y:img_y+new_height, img_x:img_x+new_width, c] +
                            opacity * imagen_resized[:, :, c]
                        )
                
                # Cargar y mostrar imagen LingoBien
                lingo_bien_path = "images/LingoBien.png"
                if os.path.exists(lingo_bien_path):
                    lingo_bien_img = cv2.imread(lingo_bien_path, cv2.IMREAD_UNCHANGED)
                    if lingo_bien_img is not None:
                        # Redimensionar LingoBien para que quepa bien
                        lb_height, lb_width = lingo_bien_img.shape[:2]
                        max_lb_width = int(view_width * 0.6)
                        max_lb_height = int(view_height * 0.4)
                        
                        lb_aspect = lb_width / lb_height
                        if lb_aspect > (max_lb_width / max_lb_height):
                            lb_new_width = max_lb_width
                            lb_new_height = int(max_lb_width / lb_aspect)
                        else:
                            lb_new_height = max_lb_height
                            lb_new_width = int(max_lb_height * lb_aspect)
                        
                        lingo_bien_resized = cv2.resize(lingo_bien_img, (lb_new_width, lb_new_height), interpolation=cv2.INTER_AREA)
                        
                        # Posicionar LingoBien centrado (bajado un poco)
                        lb_x = (view_width - lb_new_width) // 2
                        lb_y = 120
                        
                        # Dibujar LingoBien (manejar transparencia)
                        if len(lingo_bien_resized.shape) == 3 and lingo_bien_resized.shape[2] == 4:
                            alpha_lb = lingo_bien_resized[:, :, 3] / 255.0
                            img_lb_bgr = lingo_bien_resized[:, :, :3]
                            for c in range(3):
                                success_screen[lb_y:lb_y+lb_new_height, lb_x:lb_x+lb_new_width, c] = (
                                    alpha_lb * img_lb_bgr[:, :, c] + (1 - alpha_lb) * success_screen[lb_y:lb_y+lb_new_height, lb_x:lb_x+lb_new_width, c]
                                )
                        else:
                            success_screen[lb_y:lb_y+lb_new_height, lb_x:lb_x+lb_new_width] = lingo_bien_resized[:, :, :3]
                
                # Dibujar dos botones: "Salir" (izquierda) y "Siguiente" (derecha)
                button_width = 250
                button_height = 80
                button_spacing = 30
                total_buttons_width = (button_width * 2) + button_spacing
                center_x = view_width // 2
                button_y = view_height - 180  # Subidos más arriba
                
                # Botón "Salir" (izquierda) - Rojo como en el juego de clasificación
                button_salir_x = center_x - button_width - button_spacing // 2
                button_salir_text = "Salir"
                
                # Dibujar botón "Salir" (rojo con borde blanco, mismo estilo que clasificación)
                cv2.rectangle(success_screen, (button_salir_x, button_y), 
                             (button_salir_x + button_width, button_y + button_height), (0, 0, 200), -1)  # Rojo
                cv2.rectangle(success_screen, (button_salir_x, button_y), 
                             (button_salir_x + button_width, button_y + button_height), (255, 255, 255), 3)  # Borde blanco
                
                font_button = cv2.FONT_HERSHEY_DUPLEX
                font_scale_button = 0.8
                thickness_button = 2
                text_size_salir, _ = cv2.getTextSize(button_salir_text, font_button, font_scale_button, thickness_button)
                text_x_salir = button_salir_x + (button_width - text_size_salir[0]) // 2
                text_y_button = button_y + (button_height + text_size_salir[1]) // 2
                
                cv2.putText(success_screen, button_salir_text, (text_x_salir, text_y_button), 
                           font_button, font_scale_button, (255, 255, 255), thickness_button)
                
                # Botón "Siguiente" (derecha) - Verde como en el juego de clasificación
                button_siguiente_x = center_x + button_spacing // 2
                button_siguiente_text = "Siguiente"
                
                # Dibujar botón "Siguiente" (verde con borde blanco, mismo estilo que clasificación)
                cv2.rectangle(success_screen, (button_siguiente_x, button_y), 
                             (button_siguiente_x + button_width, button_y + button_height), (0, 200, 0), -1)  # Verde
                cv2.rectangle(success_screen, (button_siguiente_x, button_y), 
                             (button_siguiente_x + button_width, button_y + button_height), (255, 255, 255), 3)  # Borde blanco
                
                text_size_siguiente, _ = cv2.getTextSize(button_siguiente_text, font_button, font_scale_button, thickness_button)
                text_x_siguiente = button_siguiente_x + (button_width - text_size_siguiente[0]) // 2
                
                cv2.putText(success_screen, button_siguiente_text, (text_x_siguiente, text_y_button), 
                           font_button, font_scale_button, (255, 255, 255), thickness_button)
                
                # Mostrar pantalla de éxito
                success_screen_scaled = scale_to_videobeam(success_screen)
                cv2.imshow(window_name, success_screen_scaled)
                cv2.waitKey(10)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                
                # Iniciar streams para detección de toques
                rgb_stream = device.create_color_stream()
                depth_stream = device.create_depth_stream()
                rgb_stream.start()
                depth_stream.start()
                
                # Función para detectar toque en los botones
                def detectar_boton_tocado(cx_roi, cy_roi):
                    # cx_roi y cy_roi son coordenadas relativas a depth_roi
                    # Convertir coordenadas de ROI a viewport
                    sx = float(xv_max - xv_min) / (xw_max - xw_min)
                    sy = float(yv_max - yv_min) / (yw_max - yw_min)
                    x_viewport = int(xv_min + (cx_roi * sx))
                    y_viewport = int(yv_min + (cy_roi * sy))
                    
                    # Verificar botón "Salir" (izquierda)
                    if button_salir_x <= x_viewport <= button_salir_x + button_width and button_y <= y_viewport <= button_y + button_height:
                        return "menu"
                    # Verificar botón "Siguiente" (derecha)
                    elif button_siguiente_x <= x_viewport <= button_siguiente_x + button_width and button_y <= y_viewport <= button_y + button_height:
                        return "siguiente"
                    return None
                
                # Bucle de detección de toques
                button_pressed = False
                accion_seleccionada = None  # "siguiente", "menu", o None
                salir_bucle = False
                try:
                    while not salir_bucle:
                        frame = rgb_stream.read_frame()
                        depth_frame = depth_stream.read_frame()
                        
                        if frame is None or depth_frame is None:
                            continue
                        
                        rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
                        depth_data = cv2.flip(depth_data, 1)
                        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
                        
                        # Crear máscara de toques
                        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
                        kernel = np.ones((3, 3), np.uint8)
                        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
                        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Verificar si hay toques en esta iteración
                        hay_toque = False
                        boton_actual_tocado = None
                        
                        # Procesar toques
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 50:
                                M = cv2.moments(contour)
                                if M['m00'] != 0:
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    
                                    # Verificar si se tocó algún botón
                                    boton_tocado = detectar_boton_tocado(cx, cy)
                                    if boton_tocado:
                                        hay_toque = True
                                        boton_actual_tocado = boton_tocado
                                        
                                        # Solo procesar si no se había presionado antes
                                        if not button_pressed:
                                            button_pressed = True
                                            
                                            # Animación de botón presionado
                                            temp_screen = success_screen.copy()
                                            
                                            if boton_tocado == "siguiente":
                                                print("Botón 'Siguiente' presionado")
                                                cv2.rectangle(temp_screen, (button_siguiente_x, button_y), 
                                                             (button_siguiente_x + button_width, button_y + button_height), (0, 150, 0), -1)
                                                cv2.rectangle(temp_screen, (button_siguiente_x, button_y), 
                                                             (button_siguiente_x + button_width, button_y + button_height), (255, 255, 255), 3)
                                                cv2.putText(temp_screen, button_siguiente_text, (text_x_siguiente, text_y_button), 
                                                           font_button, font_scale_button, (255, 255, 255), thickness_button)
                                            elif boton_tocado == "menu":
                                                print("Botón 'Salir' presionado")
                                                cv2.rectangle(temp_screen, (button_salir_x, button_y), 
                                                             (button_salir_x + button_width, button_y + button_height), (0, 0, 150), -1)
                                                cv2.rectangle(temp_screen, (button_salir_x, button_y), 
                                                             (button_salir_x + button_width, button_y + button_height), (255, 255, 255), 3)
                                                cv2.putText(temp_screen, button_salir_text, (text_x_salir, text_y_button), 
                                                           font_button, font_scale_button, (255, 255, 255), thickness_button)
                                            
                                            temp_screen_scaled = scale_to_videobeam(temp_screen)
                                            cv2.imshow(window_name, temp_screen_scaled)
                                            cv2.waitKey(200)
                                            
                                            # Detener streams y salir del bucle
                                            rgb_stream.stop()
                                            depth_stream.stop()
                                            
                                            if boton_tocado == "menu":
                                                # Volver al menú principal
                                                accion_seleccionada = "menu"
                                                # Cerrar solo la ventana del juego (no todas las ventanas)
                                                try:
                                                    cv2.destroyWindow(window_name)
                                                except:
                                                    pass
                                                salir_bucle = True
                                                break
                                            elif boton_tocado == "siguiente":
                                                # Continuar con siguiente absurdo
                                                accion_seleccionada = "siguiente"
                                                cv2.destroyWindow(window_name)
                                                salir_bucle = True
                                                break
                        
                        # Resetear flag si no hay toque o si el toque no está en ningún botón
                        if button_pressed and (not hay_toque or boton_actual_tocado is None):
                            button_pressed = False
                        
                        # Mostrar pantalla
                        success_screen_scaled = scale_to_videobeam(success_screen)
                        cv2.imshow(window_name, success_screen_scaled)
                        # Asegurar que la ventana esté configurada correctamente en cada frame
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                        cv2.waitKey(10)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            rgb_stream.stop()
                            depth_stream.stop()
                            try:
                                cv2.destroyWindow(window_name)
                            except:
                                pass
                            return
                finally:
                    # Asegurar que los streams se detengan
                    try:
                        rgb_stream.stop()
                        depth_stream.stop()
                    except:
                        pass
                
                # Verificar qué acción se seleccionó (después del bloque try-finally, dentro del if es_correcta)
                if accion_seleccionada == "menu":
                    # Cerrar solo la ventana del juego (no todas las ventanas)
                    try:
                        cv2.destroyWindow(window_name)
                    except:
                        pass
                    return  # Volver al menú principal - salir completamente de la función
                elif accion_seleccionada == "siguiente":
                    break  # Salir del bucle de intentos y continuar con siguiente absurdo
            else:
                # Si la respuesta es incorrecta, manejar intentos
                intentos_restantes -= 1
                
                # Detener streams antes de mostrar mensaje
                try:
                    rgb_stream.stop()
                    depth_stream.stop()
                except:
                    pass
                
                # Mostrar mensaje de intentos restantes (en la misma posición que "Escuchando...")
                if intentos_restantes > 0:
                    mensaje_intentos = f"Inténtalo de nuevo. Intentos restantes: {intentos_restantes}"
                    font_intentos = cv2.FONT_HERSHEY_DUPLEX
                    font_scale_intentos = 0.9
                    thickness_intentos = 2
                    text_size_intentos, _ = cv2.getTextSize(mensaje_intentos, font_intentos, font_scale_intentos, thickness_intentos)
                    text_x_intentos = (view_width - text_size_intentos[0]) // 2
                    text_y_intentos = circle_center_y + circle_radius + 50  # Misma posición que "Escuchando..."
                    
                    # Color naranja oscuro para el fondo de mensajes de reintento
                    color_fondo_intentos = (0, 140, 255)  # Naranja oscuro
                    
                    # Dibujar franja de fondo con color naranja (siempre en la parte inferior)
                    franja_y_inicio = text_y_intentos - 25
                    franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
                    cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                 color_fondo_intentos, -1)  # Fondo naranja
                    cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                                 (255, 255, 255), 2)  # Borde blanco
                    
                    # Texto siempre en blanco (usar función segura para caracteres acentuados)
                    put_text_safe(videobeam_screen, mensaje_intentos, (text_x_intentos + 2, text_y_intentos + 2), 
                               font_intentos, font_scale_intentos, (0, 0, 0), thickness_intentos + 1)  # Sombra negra
                    put_text_safe(videobeam_screen, mensaje_intentos, (text_x_intentos, text_y_intentos), 
                               font_intentos, font_scale_intentos, (255, 255, 255), thickness_intentos)  # Texto blanco
                    
                    # Dibujar botón X (cerrar) en la esquina superior derecha
                    draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
                    
                    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                    cv2.imshow(window_name, videobeam_screen_scaled)
                    cv2.waitKey(10)
                    
                    # Esperar un momento antes de permitir otro intento
                    time.sleep(2)
                    
                    # Continuar con el siguiente intento (volver al inicio del bucle para la misma imagen)
                    # Reiniciar botón habilitado para el siguiente intento
                    boton_habilitado = True
            continue
        else:
            # Se agotaron los intentos, mostrar mensaje y continuar con siguiente absurdo
            mensaje_final = "Se agotaron los intentos"
            font_final = cv2.FONT_HERSHEY_DUPLEX
            font_scale_final = 0.9
            thickness_final = 2
            text_size_final, _ = cv2.getTextSize(mensaje_final, font_final, font_scale_final, thickness_final)
            text_x_final = (view_width - text_size_final[0]) // 2
            # Mover al área inferior (misma posición que "Escuchando...")
            text_y_final = circle_center_y + circle_radius + 50
            
            # Color rojo oscuro para el fondo de mensaje de intentos agotados
            color_fondo_final = (0, 0, 200)  # Rojo oscuro
            
            # Dibujar franja de fondo con color rojo oscuro (siempre en la parte inferior)
            franja_y_inicio = text_y_final - 25
            franja_y_fin = view_height - 1  # Llegar hasta el borde inferior
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         color_fondo_final, -1)  # Fondo rojo oscuro
            cv2.rectangle(videobeam_screen, (0, franja_y_inicio), (view_width, franja_y_fin), 
                         (255, 255, 255), 2)  # Borde blanco
            
            # Texto siempre en blanco (usar función segura para caracteres acentuados)
            put_text_safe(videobeam_screen, mensaje_final, (text_x_final + 2, text_y_final + 2), 
                       font_final, font_scale_final, (0, 0, 0), thickness_final + 1)  # Sombra negra
            put_text_safe(videobeam_screen, mensaje_final, (text_x_final, text_y_final), 
                       font_final, font_scale_final, (255, 255, 255), thickness_final)  # Texto blanco
            
            # Dibujar botón X (cerrar) en la esquina superior derecha
            draw_close_card(videobeam_screen, view_width, view_height, elevated=False)
            
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            cv2.imshow(window_name, videobeam_screen_scaled)
            cv2.waitKey(10)
            
            time.sleep(3)
            # Continuar con el siguiente absurdo (salir del bucle de intentos)
            break

