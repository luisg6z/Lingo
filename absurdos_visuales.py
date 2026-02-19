import cv2
import numpy as np
import os
import time
import random
import json
import threading
import pygame
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
import pyttsx3


def juego_absurdos_reconocimiento_voz(device, coordenadas, dmax_map, dmin_map):
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
    json_path = "absurdos_logicos/config/absurdos.json"
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
    
    # Seleccionar un absurdo aleatorio
    absurdo_actual = random.choice(absurdos_list)
    imagen_nombre = absurdo_actual.get("imagen", "")
    
    # Construir ruta de imagen
    assets_path = "absurdos_logicos/assets"
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
        return
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    # Crear pantalla del videobeam
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores (mismo estilo que otros juegos)
    for y in range(view_height):
        ratio = y / view_height
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        videobeam_screen[y, :] = [b, g, r]
    
    # Redimensionar imagen para que quepa en la pantalla (dejar espacio para texto)
    max_img_width = int(view_width * 0.7)
    max_img_height = int(view_height * 0.6)
    
    img_height, img_width = imagen.shape[:2]
    aspect_ratio = img_width / img_height
    
    if aspect_ratio > (max_img_width / max_img_height):
        new_width = max_img_width
        new_height = int(max_img_width / aspect_ratio)
    else:
        new_height = max_img_height
        new_width = int(max_img_height * aspect_ratio)
    
    imagen_resized = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Centrar la imagen en la pantalla
    img_x = (view_width - new_width) // 2
    img_y = (view_height - new_height) // 2 - 50  # Un poco más arriba para dejar espacio para texto
    
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
    
    # Mostrar mensaje inicial
    mensaje = "Ahora habla, di que esta mal"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(mensaje, font, font_scale, thickness)
    text_x = (view_width - text_size[0]) // 2
    text_y = view_height - 50
    
    # Sombra del texto
    cv2.putText(videobeam_screen, mensaje, (text_x + 2, text_y + 2), 
               font, font_scale, (0, 0, 0), thickness + 1)
    # Texto principal
    cv2.putText(videobeam_screen, mensaje, (text_x, text_y), 
               font, font_scale, (255, 255, 255), thickness)
    
    # Mostrar en pantalla
    window_name = "Juego de Reconocimiento de Voz"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1920, 0)
    cv2.waitKey(50)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
    
    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
    cv2.imshow(window_name, videobeam_screen_scaled)
    cv2.waitKey(50)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(window_name, VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
    
    # Inicializar TTS
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
    
    # Decir instrucción en un hilo separado
    tts_finished = threading.Event()
    def decir_instruccion():
        if engine_tts:
            try:
                engine_tts.say("Ahora habla, di qué está mal")
                engine_tts.runAndWait()
            except Exception as e:
                print(f"Error al decir instrucción: {e}")
            finally:
                tts_finished.set()
    
    tts_thread = threading.Thread(target=decir_instruccion, daemon=True)
    tts_thread.start()
    
    # Esperar a que termine de hablar el TTS (máximo 5 segundos)
    tts_finished.wait(timeout=5.0)
    # Esperar un poco más para asegurar que el audio se haya limpiado
    time.sleep(0.5)
    
    # Inicializar reconocimiento de voz
    try:
        recognizer = sr.Recognizer()
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
            microphone = sr.Microphone(device_index=0)
        
        print(f"Usando micrófono: {microphone}")
    except OSError as e:
        if "PyAudio" in str(e) or "pyaudio" in str(e).lower():
            print("Error: PyAudio no está instalado. Instalando...")
            print("Por favor ejecuta: uv pip install pyaudio")
            print("O si eso no funciona, instala desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        else:
            print(f"Error al inicializar reconocimiento de voz: {e}")
        cv2.destroyWindow(window_name)
        return
    except Exception as e:
        print(f"Error al inicializar reconocimiento de voz: {e}")
        print("Asegúrate de que PyAudio esté instalado: uv pip install pyaudio")
        cv2.destroyWindow(window_name)
        return
    
    # Ajustar para ruido ambiente
    print("Ajustando para ruido ambiente...")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print(f"Umbral de energía ajustado: {recognizer.energy_threshold}")
    except Exception as e:
        print(f"Error al ajustar ruido ambiente: {e}")
        # Continuar de todas formas con valores por defecto
    
    # Actualizar mensaje en pantalla
    mensaje_escuchando = "Escuchando..."
    text_size_esc, _ = cv2.getTextSize(mensaje_escuchando, font, font_scale, thickness)
    text_x_esc = (view_width - text_size_esc[0]) // 2
    
    # Limpiar área de texto anterior y dibujar nuevo mensaje
    cv2.rectangle(videobeam_screen, (0, text_y - 30), (view_width, text_y + 30), (int(255 * 0.3), int(200 * 0.5), int(255 * 0.8)), -1)
    cv2.putText(videobeam_screen, mensaje_escuchando, (text_x_esc + 2, text_y + 2), 
               font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(videobeam_screen, mensaje_escuchando, (text_x_esc, text_y), 
               font, font_scale, (0, 255, 255), thickness)
    
    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
    cv2.imshow(window_name, videobeam_screen_scaled)
    cv2.waitKey(10)
    
    # Escuchar audio con detección de silencio
    texto_reconocido = None
    audio = None
    print("Iniciando captura de audio...")
    try:
        with microphone as source:
            print("Esperando audio (timeout: 15 segundos, límite de frase: 20 segundos)...")
            # Aumentar timeout y límite de frase para dar más tiempo
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)
            print(f"Audio capturado. Duración aproximada: {len(audio.frame_data) / audio.sample_rate:.2f} segundos")
    except sr.WaitTimeoutError:
        print("Tiempo de espera agotado. No se detectó ningún audio.")
        print("Verifica que el micrófono esté funcionando y que no esté silenciado.")
        texto_reconocido = None
        audio = None
    except Exception as e:
        print(f"Error al escuchar el micrófono: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        texto_reconocido = None
        audio = None
    
    # Reconocer el audio
    if audio is not None:
        print("Procesando audio con Google Speech Recognition...")
        try:
            texto_reconocido = recognizer.recognize_google(audio, language="es-ES")
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
    
    # Cargar modelo de sentence-transformers
    model = None
    try:
        # Usar modelo multilingüe que soporta español
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modelo de sentence-transformers cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo de sentence-transformers: {e}")
        print("El modelo se descargará automáticamente en el primer uso")
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e2:
            print(f"Error al descargar/cargar modelo: {e2}")
            cv2.destroyWindow(window_name)
            return
    
    # Cargar respuestas correctas desde JSON
    respuestas_json_path = "absurdos_logicos/config/respuestas_correctas.json"
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
    
    # Evaluar respuesta usando sentence-transformers
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
    
    # Mostrar resultado
    color_resultado = (0, 255, 0) if es_correcta else (0, 0, 255)  # Verde o rojo
    font_resultado = cv2.FONT_HERSHEY_DUPLEX
    font_scale_resultado = 1.5
    thickness_resultado = 3
    
    text_size_resultado, _ = cv2.getTextSize(mensaje_resultado, font_resultado, font_scale_resultado, thickness_resultado)
    text_x_resultado = (view_width - text_size_resultado[0]) // 2
    text_y_resultado = text_y - 80
    
    # Limpiar área y dibujar resultado
    cv2.rectangle(videobeam_screen, (text_x_resultado - 20, text_y_resultado - 40), 
                 (text_x_resultado + text_size_resultado[0] + 20, text_y_resultado + 20), 
                 (int(255 * 0.3), int(200 * 0.5), int(255 * 0.8)), -1)
    
    # Sombra del texto
    cv2.putText(videobeam_screen, mensaje_resultado, (text_x_resultado + 3, text_y_resultado + 3), 
               font_resultado, font_scale_resultado, (0, 0, 0), thickness_resultado + 2)
    # Texto principal
    cv2.putText(videobeam_screen, mensaje_resultado, (text_x_resultado, text_y_resultado), 
               font_resultado, font_scale_resultado, color_resultado, thickness_resultado)
    
    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
    cv2.imshow(window_name, videobeam_screen_scaled)
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
                time.sleep(2)  # Esperar a que termine el sonido
            except Exception as e:
                print(f"Error al reproducir sonido: {e}")
    except Exception as e:
        print(f"Error al inicializar mixer: {e}")
    
    # Esperar antes de cerrar
    time.sleep(3)
    
    # Cerrar ventana
    cv2.destroyWindow(window_name)

