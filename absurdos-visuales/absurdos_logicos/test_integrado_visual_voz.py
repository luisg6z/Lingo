"""
Script de prueba integrado: JSON + Pygame + pyttsx3 + SpeechRecognition.
Este archivo es solo para pruebas y no modifica main.py.
"""

import pygame
import sys
import os
import json
import threading
import time

# Verificar e importar pyttsx3
try:
    import pyttsx3
except ImportError:
    print("Error: pyttsx3 no está instalado. Instálalo con: pip install pyttsx3")
    sys.exit(1)

# Verificar e importar speech_recognition
try:
    import speech_recognition as sr
except ImportError:
    print("Error: speech_recognition no está instalado. Instálalo con: pip install SpeechRecognition")
    sys.exit(1)

# Ruta del archivo JSON
ruta_json = os.path.join("config", "absurdos.json")

# Verificar si existe el archivo JSON
if not os.path.exists(ruta_json):
    print(f"Error: No se encontró el archivo {ruta_json}")
    sys.exit(1)

# Leer el archivo JSON
try:
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
except json.JSONDecodeError:
    print(f"Error: El archivo {ruta_json} no es un JSON válido")
    sys.exit(1)
except Exception as e:
    print(f"Error al leer el archivo {ruta_json}: {e}")
    sys.exit(1)

# Obtener la lista de absurdos
if "absurdos" not in datos:
    print("Error: No se encontró la clave 'absurdos' en el JSON")
    sys.exit(1)

if not datos["absurdos"] or len(datos["absurdos"]) == 0:
    print("Error: La lista 'absurdos' está vacía")
    sys.exit(1)

# Variables globales para el absurdo actual
indice_absurdo_actual = 0
lista_absurdos = datos["absurdos"]
absurdo_actual = lista_absurdos[0]

# Verificar que el absurdo actual tenga los campos necesarios
if "imagen" not in absurdo_actual:
    print("Error: No se encontró la clave 'imagen' en el absurdo actual")
    sys.exit(1)

if "descripcion" not in absurdo_actual:
    print("Error: No se encontró la clave 'descripcion' en el absurdo actual")
    sys.exit(1)

nombre_imagen = absurdo_actual["imagen"]
descripcion = absurdo_actual["descripcion"]

# Construir la ruta completa de la imagen
ruta_imagen = os.path.join("assets", "images", nombre_imagen)

# Verificar si existe la imagen
if not os.path.exists(ruta_imagen):
    print(f"Error: No se encontró la imagen en {ruta_imagen}")
    sys.exit(1)

# Inicializar Pygame
pygame.init()

# Inicializar pygame.mixer para reproducir sonidos
pygame.mixer.init()

# Configuración de la ventana
ANCHO = 800
ALTO = 600

# Tamaño fijo para todas las imágenes (configurable)
IMG_W = 450
IMG_H = 450

# Crear la ventana
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Absurdos Lógicos - Prueba Integrada")

# Inicializar sistema de puntaje
puntaje = 0
PUNTOS_POR_ACIERTO = 10

# Inicializar fuente para mostrar el puntaje
try:
    fuente_puntaje = pygame.font.Font(None, 36)  # Tamaño mediano
except:
    fuente_puntaje = pygame.font.SysFont("arial", 36)

# Cargar la imagen y escalarla al tamaño fijo
try:
    imagen_original = pygame.image.load(ruta_imagen)
    # Escalar la imagen al tamaño fijo manteniendo la calidad
    imagen = pygame.transform.smoothscale(imagen_original, (IMG_W, IMG_H))
except pygame.error as e:
    print(f"Error al cargar la imagen {ruta_imagen}: {e}")
    pygame.quit()
    sys.exit(1)

# Calcular la posición para centrar la imagen (ya tiene tamaño fijo)
pos_x = (ANCHO - IMG_W) // 2
pos_y = (ALTO - IMG_H) // 2

# Constante para el tiempo de escucha (configurable entre 10 y 15 segundos)
TIEMPO_ESCUCHA = 12

# Constante para el máximo de intentos cuando falta explicación
# MAX_INTENTOS = 2 permite hasta intento_actual = 1 (2 intentos totales)
# Para permitir más intentos, aumentar este valor
MAX_INTENTOS = 2

# Instrucción para el niño
INSTRUCCION = "Ahora dime qué está mal"

# Variables globales para la imagen actual (se actualizarán cuando cambie el absurdo)
imagen_actual = imagen
pos_x_actual = pos_x
pos_y_actual = pos_y

# Función para cargar un absurdo por índice
def cargar_absurdo(indice):
    """
    Carga un absurdo específico por su índice y actualiza las variables globales.
    """
    global absurdo_actual, nombre_imagen, descripcion, ruta_imagen
    global imagen_actual, pos_x_actual, pos_y_actual
    
    if indice >= len(lista_absurdos):
        print(f"No hay más absurdos. Se completaron {len(lista_absurdos)} absurdos.")
        return False
    
    absurdo_actual = lista_absurdos[indice]
    
    # Verificar campos necesarios
    if "imagen" not in absurdo_actual or "descripcion" not in absurdo_actual:
        print(f"Error: El absurdo {indice + 1} no tiene los campos necesarios")
        return False
    
    nombre_imagen = absurdo_actual["imagen"]
    descripcion = absurdo_actual["descripcion"]
    
    # Construir ruta de imagen
    ruta_imagen = os.path.join("assets", "images", nombre_imagen)
    
    # Verificar si existe la imagen
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró la imagen en {ruta_imagen}")
        return False
    
    # Cargar la nueva imagen y escalarla al tamaño fijo
    try:
        imagen_original = pygame.image.load(ruta_imagen)
        # Escalar la imagen al tamaño fijo manteniendo la calidad
        imagen_actual = pygame.transform.smoothscale(imagen_original, (IMG_W, IMG_H))
        # Calcular posición centrada (tamaño fijo)
        pos_x_actual = (ANCHO - IMG_W) // 2
        pos_y_actual = (ALTO - IMG_H) // 2
        print(f"\n=== CARGADO ABSURDO {indice + 1}: {descripcion} ===")
        return True
    except pygame.error as e:
        print(f"Error al cargar la imagen {ruta_imagen}: {e}")
        return False

# Cargar sonidos de retroalimentación
ruta_sonido_correcto = os.path.join("assets", "sounds", "correcto.wav")
ruta_sonido_incorrecto = os.path.join("assets", "sounds", "incorrecto.wav")

sonido_correcto = None
sonido_incorrecto = None

# Cargar sonido correcto
if os.path.exists(ruta_sonido_correcto):
    try:
        sonido_correcto = pygame.mixer.Sound(ruta_sonido_correcto)
    except pygame.error as e:
        print(f"Advertencia: No se pudo cargar el sonido correcto: {e}")
else:
    print(f"Advertencia: No se encontró el archivo de sonido: {ruta_sonido_correcto}")

# Cargar sonido incorrecto
if os.path.exists(ruta_sonido_incorrecto):
    try:
        sonido_incorrecto = pygame.mixer.Sound(ruta_sonido_incorrecto)
    except pygame.error as e:
        print(f"Advertencia: No se pudo cargar el sonido incorrecto: {e}")
else:
    print(f"Advertencia: No se encontró el archivo de sonido: {ruta_sonido_incorrecto}")

# Variable para controlar el bucle principal
ejecutando = True

# Variable para controlar si se debe mostrar la pantalla final
juego_completado = False

# Lock para evitar que múltiples hilos hablen al mismo tiempo
tts_lock = threading.Lock()

# Función para hablar textos en orden con lock
def hablar_en_orden(textos):
    """
    Inicializa pyttsx3, dice cada texto en orden, y cierra el engine.
    Usa un lock para evitar que múltiples hilos hablen simultáneamente.
    
    Args:
        textos: Lista de strings a decir en orden, o un solo string
    """
    # Si es un solo string, convertirlo a lista
    if isinstance(textos, str):
        textos = [textos]
    
    with tts_lock:
        try:
            # Inicializar engine dentro de la función (nuevo para cada llamada)
            engine = pyttsx3.init()
            
            # Decir cada texto en orden
            for i, texto in enumerate(textos):
                # print(f"TTS: diciendo texto {i+1}/{len(textos)}: {texto}")  # Debug comentado
                engine.say(texto)
            
            # Ejecutar todos los textos en orden
            engine.runAndWait()
            # print("TTS: todos los textos reproducidos correctamente")  # Debug comentado
            
            # El engine se cierra automáticamente al salir del contexto
        except Exception as e:
            print(f"Error en TTS: {e}")

# Función auxiliar para normalizar texto (eliminar tildes)
def normalizar_texto(texto):
    """
    Normaliza el texto a minúsculas y elimina tildes para tolerar variaciones.
    """
    if not texto:
        return ""
    
    texto_lower = texto.lower()
    
    # Eliminar tildes
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u'
    }
    texto_normalizado = texto_lower
    for tilde, sin_tilde in reemplazos.items():
        texto_normalizado = texto_normalizado.replace(tilde, sin_tilde)
    
    return texto_normalizado

# Función para mostrar el puntaje en la esquina inferior derecha
def mostrar_puntaje():
    """
    Renderiza el texto del puntaje en la esquina inferior derecha de la ventana.
    """
    global puntaje
    texto_puntaje = f"Puntaje: {puntaje}"
    superficie_texto = fuente_puntaje.render(texto_puntaje, True, (0, 0, 0))  # Color negro
    # Calcular posición en esquina inferior derecha con margen
    margen = 10
    pos_x_texto = ANCHO - superficie_texto.get_width() - margen
    pos_y_texto = ALTO - superficie_texto.get_height() - margen
    ventana.blit(superficie_texto, (pos_x_texto, pos_y_texto))

# Función para mostrar la pantalla final
def mostrar_pantalla_final():
    """
    Muestra la pantalla final con el mensaje de finalización y el puntaje total.
    También dice el mensaje con pyttsx3.
    """
    global puntaje, ejecutando
    
    # Limpiar la ventana con color blanco
    ventana.fill((255, 255, 255))
    
    # Crear fuente más grande para los mensajes de la pantalla final
    try:
        fuente_titulo = pygame.font.Font(None, 48)
        fuente_subtitulo = pygame.font.Font(None, 36)
    except:
        fuente_titulo = pygame.font.SysFont("arial", 48)
        fuente_subtitulo = pygame.font.SysFont("arial", 36)
    
    # Texto 1: "¡Muy bien! Terminamos el juego."
    texto1 = "¡Muy bien! Terminamos el juego."
    superficie_texto1 = fuente_titulo.render(texto1, True, (0, 0, 0))
    pos_x1 = (ANCHO - superficie_texto1.get_width()) // 2
    pos_y1 = ALTO // 2 - 60
    
    # Texto 2: "Puntaje total: X"
    texto2 = f"Puntaje total: {puntaje}"
    superficie_texto2 = fuente_subtitulo.render(texto2, True, (0, 0, 0))
    pos_x2 = (ANCHO - superficie_texto2.get_width()) // 2
    pos_y2 = ALTO // 2 + 20
    
    # Dibujar los textos centrados
    ventana.blit(superficie_texto1, (pos_x1, pos_y1))
    ventana.blit(superficie_texto2, (pos_x2, pos_y2))
    
    # Actualizar la pantalla
    pygame.display.flip()
    
    # Decir el mensaje con pyttsx3 en un hilo para no congelar
    def decir_mensaje_final():
        try:
            mensaje_voz = f"Terminamos el juego. Tu puntaje total fue {puntaje} puntos."
            print(f"Diciendo mensaje final: {mensaje_voz}")
            hablar_en_orden(mensaje_voz)  # Usa hablar_en_orden con lock
        except Exception as e:
            print(f"Error al decir mensaje final: {e}")
    
    # Decir el mensaje en un hilo
    hilo_final = threading.Thread(target=decir_mensaje_final, daemon=True)
    hilo_final.start()
    
    # Esperar 2-3 segundos (esperar a que termine de hablar + tiempo adicional)
    time.sleep(3.0)
    
    # Cerrar el juego
    ejecutando = False

# Función para evaluar si la respuesta es correcta (versión estricta y pedagógica)
def evaluar_respuesta(texto_respuesta, absurdo):
    """
    Evalúa si la respuesta del niño es correcta con reglas estrictas y pedagógicas.
    El sistema solo marca CORRECTO cuando el niño rechaza el absurdo y explica por qué.
    
    Devuelve una tupla: (es_correcta: bool, motivo: str)
    Motivos posibles: "correcto", "incorrecto", "falta_explicacion"
    """
    # Regla 1: Validar que el texto no esté vacío y tenga al menos 2 palabras
    if not texto_respuesta or texto_respuesta.strip() == "":
        return (False, "incorrecto")
    
    palabras = texto_respuesta.strip().split()
    if len(palabras) < 2:
        return (False, "incorrecto")
    
    # Normalizar el texto (minúsculas y sin tildes)
    texto_normalizado = normalizar_texto(texto_respuesta)
    
    # Obtener información del absurdo para evaluar
    propiedad_incorrecta = normalizar_texto(absurdo.get("propiedad_incorrecta", ""))
    propiedades_correctas = [normalizar_texto(p) for p in absurdo.get("propiedades_correctas", [])]
    objeto = normalizar_texto(absurdo.get("objeto", ""))
    
    # Detecciones básicas
    
    # rechazo: si contiene palabras o frases de rechazo
    palabras_rechazo = ["no", "no puede", "no es", "no existe", "esta mal", "está mal", 
                        "incorrecto", "imposible"]
    rechazo = any(palabra in texto_normalizado for palabra in palabras_rechazo)
    
    # aceptacion: si contiene palabras de aceptación
    palabras_aceptacion = ["si", "sí", "puede", "esta bien", "está bien", 
                           "es logico", "es lógico", "correcto", "es verdad"]
    aceptacion = any(palabra in texto_normalizado for palabra in palabras_aceptacion)
    
    # menciona_absurdo: si menciona el color absurdo
    menciona_absurdo = propiedad_incorrecta in texto_normalizado if propiedad_incorrecta else False
    
    # niega_absurdo: si aparecen patrones como "no azul", "no es azul", "no puede ser azul"
    niega_absurdo = False
    if propiedad_incorrecta:
        patrones_negacion = [
            f"no {propiedad_incorrecta}",
            f"no es {propiedad_incorrecta}",
            f"no puede ser {propiedad_incorrecta}",
            f"{propiedad_incorrecta} no"
        ]
        niega_absurdo = any(patron in texto_normalizado for patron in patrones_negacion)
    
    # menciona_color_correcto: si menciona algún color correcto (con tolerancia a variaciones)
    menciona_color_correcto = False
    for prop_correcta in propiedades_correctas:
        if prop_correcta in texto_normalizado:
            menciona_color_correcto = True
            break
        # Verificar variaciones de género (ej: "amarilla" vs "amarillo")
        if prop_correcta.endswith('a') and len(prop_correcta) > 1:
            variacion = prop_correcta[:-1] + 'o'  # "amarilla" -> "amarillo"
            if variacion in texto_normalizado:
                menciona_color_correcto = True
                break
        elif prop_correcta.endswith('o') and len(prop_correcta) > 1:
            variacion = prop_correcta[:-1] + 'a'  # "amarillo" -> "amarilla"
            if variacion in texto_normalizado:
                menciona_color_correcto = True
                break
    
    # justificacion: verdadero si ocurre al menos UNA de estas condiciones
    tiene_porque = "porque" in texto_normalizado
    frases_regla = []
    if objeto:
        frases_regla = [
            f"las {objeto}s",
            f"la {objeto}",
            f"una {objeto}",
            f"un {objeto}",
            "normalmente",
            "tiene que ser",
            "debe ser"
        ]
    tiene_frase_regla = any(frase in texto_normalizado for frase in frases_regla)
    justificacion = tiene_porque or menciona_color_correcto or tiene_frase_regla
    
    # Reglas de decisión (en este orden)
    
    # Regla 2: Si hay aceptacion y NO hay rechazo → INCORRECTO
    if aceptacion and not rechazo:
        return (False, "incorrecto")
    
    # Regla 3: Si menciona_absurdo y NO niega_absurdo → INCORRECTO
    if menciona_absurdo and not niega_absurdo:
        return (False, "incorrecto")
    
    # Regla 4: Para que sea CORRECTO, deben cumplirse AMBAS:
    #   (rechazo o niega_absurdo) Y justificacion
    tiene_rechazo_o_niega = rechazo or niega_absurdo
    
    if tiene_rechazo_o_niega and justificacion:
        return (True, "correcto")
    
    # Regla 5: Si hay rechazo pero NO hay justificación → falta_explicacion
    if tiene_rechazo_o_niega and not justificacion:
        return (False, "falta_explicacion")
    
    # Regla 6: En cualquier otro caso → INCORRECTO
    return (False, "incorrecto")

# Función para dar retroalimentación (sonido + voz)
def dar_retroalimentacion(es_correcta, motivo, reintentar_escucha=False, intento_actual=0):
    """
    Reproduce el sonido y dice el mensaje de retroalimentación según el motivo.
    Se ejecuta en un hilo para no congelar la ventana.
    Para "falta_explicacion", maneja los reintentos hasta MAX_INTENTOS.
    """
    try:
        # Determinar el mensaje según el motivo
        if motivo == "correcto":
            # Sumar puntos por acierto
            global puntaje
            puntaje += PUNTOS_POR_ACIERTO
            print(f"¡Puntos ganados! Puntaje actual: {puntaje}")
            
            # Reproducir sonido correcto
            if sonido_correcto:
                sonido_correcto.play()
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
            mensaje = "¡Muy bien! Eso no es lógico."
            # Decir el mensaje con pyttsx3 usando hablar_en_orden (con lock)
            print(f"Diciendo retroalimentación: {mensaje}")
            hablar_en_orden(mensaje)
            
            # Avanzar al siguiente absurdo
            global indice_absurdo_actual, escucha_iniciada, hilo_tts
            indice_absurdo_actual += 1
            
            # Si hay más absurdos, cargar el siguiente
            if cargar_absurdo(indice_absurdo_actual):
                # Esperar un momento antes de iniciar el siguiente absurdo
                time.sleep(1.0)
                # Reiniciar el flag de escucha para el nuevo absurdo
                escucha_iniciada = False
                # Iniciar el proceso para el siguiente absurdo
                # print(f"TTS: iniciando hilo para absurdo {indice_absurdo_actual + 1}")  # Debug comentado
                hilo_tts = threading.Thread(target=decir_descripcion_e_instruccion, daemon=False)
                hilo_tts.start()
                # print(f"TTS: hilo iniciado, estado: {hilo_tts.is_alive()}")  # Debug comentado
                print(f"Iniciado nuevo absurdo {indice_absurdo_actual + 1}. Esperando a que termine de hablar...")
            else:
                # No hay más absurdos, activar pantalla final
                global juego_completado
                juego_completado = True
                print("¡Felicidades! Has completado todos los absurdos.")
            return
            
        elif motivo == "falta_explicacion":
            # Reproducir sonido incorrecto
            if sonido_incorrecto:
                sonido_incorrecto.play()
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
            
            # Si ya se alcanzó el máximo de intentos, dar explicación genérica
            if intento_actual >= MAX_INTENTOS:
                mensaje = "Te explico: en la vida real, eso no ocurre así."
            else:
                mensaje = "Está bien que digas que está mal, pero dime por qué."
            
            # Decir el mensaje con pyttsx3 usando hablar_en_orden (con lock)
            print(f"Diciendo retroalimentación: {mensaje}")
            hablar_en_orden(mensaje)
            
            # Si no se alcanzó el máximo de intentos, volver a escuchar
            print(f"Debug retroalimentacion: intento_actual={intento_actual}, MAX_INTENTOS={MAX_INTENTOS}, reintentar_escucha={reintentar_escucha}")
            print(f"Debug: Condición reintentar: intento_actual < MAX_INTENTOS = {intento_actual < MAX_INTENTOS}")
            if intento_actual < MAX_INTENTOS and reintentar_escucha:
                # Esperar un momento antes de volver a escuchar
                print("Esperando 0.5 segundos antes de iniciar el reintento...")
                time.sleep(0.5)
                # Volver a iniciar el proceso COMPLETO de escucha con el siguiente intento
                # Esto incluye: ajustar ruido → escuchar → procesar → transcribir → evaluar
                siguiente_intento = intento_actual + 1
                print(f"\n=== INICIANDO REINTENTO {siguiente_intento + 1} ===")
                print(f"Se repetirá TODO el proceso: ajustar ruido → escuchar → procesar → transcribir → evaluar")
                print(f"Llamando a escuchar_respuesta con intento_actual={siguiente_intento}")
                
                # Iniciar el hilo (NO daemon para asegurar que se ejecute completamente)
                hilo_escucha = threading.Thread(target=escuchar_respuesta, args=(siguiente_intento,), daemon=False)
                hilo_escucha.start()
                print(f"Hilo de escucha iniciado. Estado: {hilo_escucha.is_alive()}")
                
                # Dar un momento para que el hilo se inicie correctamente
                time.sleep(0.2)
                print(f"Estado del hilo después de iniciar: {hilo_escucha.is_alive()}")
            else:
                print(f"Debug: NO se reinicia porque intento_actual ({intento_actual}) >= MAX_INTENTOS ({MAX_INTENTOS}) o reintentar_escucha es False")
            return
            
        else:  # motivo == "incorrecto"
            # Reproducir sonido incorrecto
            if sonido_incorrecto:
                sonido_incorrecto.play()
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
            mensaje = "Intenta otra vez."
            # Decir el mensaje con pyttsx3 usando hablar_en_orden (con lock)
            print(f"Diciendo retroalimentación: {mensaje}")
            hablar_en_orden(mensaje)
            
            # Si se debe reintentar (solo para el primer intento de "incorrecto")
            if reintentar_escucha and intento_actual == 0:
                # Esperar un momento antes de volver a escuchar
                print("Esperando 0.5 segundos antes de iniciar el segundo intento...")
                time.sleep(0.5)
                # Volver a iniciar el proceso COMPLETO de escucha con el siguiente intento
                siguiente_intento = intento_actual + 1
                print(f"\n=== INICIANDO SEGUNDO INTENTO (motivo: incorrecto) ===")
                print(f"Se repetirá TODO el proceso: ajustar ruido → escuchar → procesar → transcribir → evaluar")
                print(f"Llamando a escuchar_respuesta con intento_actual={siguiente_intento}")
                
                # Iniciar el hilo (NO daemon para asegurar que se ejecute completamente)
                hilo_escucha = threading.Thread(target=escuchar_respuesta, args=(siguiente_intento,), daemon=False)
                hilo_escucha.start()
                print(f"Hilo de escucha iniciado. Estado: {hilo_escucha.is_alive()}")
                
                # Dar un momento para que el hilo se inicie correctamente
                time.sleep(0.2)
                print(f"Estado del hilo después de iniciar: {hilo_escucha.is_alive()}")
            else:
                print("Motivo 'incorrecto': NO se reinicia el proceso. Solo feedback una vez.")
            return
        
    except Exception as e:
        print(f"Error al dar retroalimentación: {e}")

# Función para decir la descripción e instrucción en un hilo separado
def decir_descripcion_e_instruccion():
    """
    Función que se ejecuta en un hilo para decir la descripción e instrucción sin congelar Pygame.
    Usa el absurdo actual para obtener la descripción correcta.
    """
    global absurdo_actual, indice_absurdo_actual
    
    try:
        # Obtener la descripción del absurdo actual
        descripcion_actual = absurdo_actual.get("descripcion", "")
        # print(f"TTS: diciendo descripción del absurdo {indice_absurdo_actual + 1}")  # Debug comentado
        # print(f"TTS: descripción = '{descripcion_actual}'")  # Debug comentado
        
        # Preparar los textos a decir en orden
        textos = [
            descripcion_actual,
            INSTRUCCION
        ]
        
        # print("TTS: diciendo instrucción")  # Debug comentado
        
        # Usar la función hablar_en_orden que maneja el lock y el engine
        hablar_en_orden(textos)
        
        # print("TTS: descripción e instrucción completadas")  # Debug comentado
    except Exception as e:
        print(f"Error al reproducir descripción e instrucción: {e}")

# Función para escuchar la respuesta en un hilo separado
def escuchar_respuesta(intento_actual=0):
    """
    Función que se ejecuta en un hilo para escuchar la respuesta sin congelar Pygame.
    Acepta un parámetro intento_actual para manejar reintentos en caso de "falta_explicacion".
    En cada llamada, ejecuta el proceso completo: ajustar ruido → escuchar → procesar → transcribir → evaluar.
    """
    print(f"\n{'='*50}")
    print(f"FUNCIÓN escuchar_respuesta EJECUTÁNDOSE - Intento {intento_actual + 1}")
    print(f"{'='*50}")
    
    texto_reconocido = None
    
    # Indicar si es un reintento
    if intento_actual > 0:
        print(f"\n--- REINTENTO {intento_actual + 1} ---")
        print("Repitiendo el proceso completo: ajustar ruido → escuchar → procesar → transcribir → evaluar")
    
    try:
        # Paso 1: Inicializar reconocedor y micrófono
        print("Paso 1: Inicializando reconocedor y micrófono...")
        recognizer = sr.Recognizer()
        
        # Configurar el reconocedor para mejor precisión
        recognizer.energy_threshold = 300  # Umbral de energía más bajo para mejor detección
        recognizer.dynamic_energy_threshold = True  # Ajuste dinámico del umbral
        recognizer.pause_threshold = 0.8  # Pausa más corta para detectar fin de frase
        recognizer.phrase_threshold = 0.3  # Umbral de frase más bajo
        
        microfono = sr.Microphone()
        print("Reconocedor y micrófono inicializados correctamente.")
        
        # Paso 2: Ajustar ruido ambiente (siempre se repite en cada intento)
        print("Paso 2: Ajustando ruido ambiente... Por favor, mantén silencio por un momento.")
        try:
            with microfono as source:
                # Aumentar el tiempo de ajuste para mejor calibración
                recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Ruido ambiente ajustado. Ahora puedes hablar claramente...")
        except Exception as e:
            print(f"Error al ajustar ruido ambiente: {e}")
            raise
        
        # Paso 3: Escuchar audio
        print(f"Paso 3: Escuchando... ({TIEMPO_ESCUCHA} segundos)")
        print("Habla claramente y pausa cuando termines...")
        try:
            with microfono as source:
                # Usar listen con mejor configuración para captura más precisa
                audio = recognizer.listen(
                    source, 
                    timeout=TIEMPO_ESCUCHA, 
                    phrase_time_limit=TIEMPO_ESCUCHA
                )
            print("Audio capturado correctamente.")
        except Exception as e:
            print(f"Error al escuchar audio: {e}")
            raise
        
        # Paso 4: Procesar y transcribir audio
        print("Paso 4: Procesando y transcribiendo audio...")
        try:
            # Usar reconocimiento de Google con mejor configuración
            # show_all=False para obtener solo el mejor resultado
            texto_reconocido = recognizer.recognize_google(
                audio, 
                language="es-ES",
                show_all=False
            )
            if texto_reconocido:
                texto_reconocido = texto_reconocido.strip()  # Eliminar espacios al inicio/final
                print(f"Respuesta del niño: {texto_reconocido}")
            else:
                print("Advertencia: No se transcribió ningún texto")
                texto_reconocido = ""
        except sr.UnknownValueError:
            print("No se pudo entender el audio. Por favor, habla más claro.")
            texto_reconocido = ""
        except Exception as e:
            print(f"Error al procesar/transcribir audio: {e}")
            raise
        
        # Paso 5: Evaluar la respuesta
        print("Evaluando respuesta...")
        es_correcta, motivo = evaluar_respuesta(texto_reconocido, absurdo_actual)
        print(f"Evaluación: {'Correcta' if es_correcta else 'Incorrecta'} (motivo: {motivo}, intento: {intento_actual + 1})")
        
        # Determinar si se debe reintentar
        # Para "falta_explicacion": permitir hasta MAX_INTENTOS reintentos
        # Para "incorrecto": permitir solo 1 reintento (solo si intento_actual = 0)
        # Para "correcto": nunca reintentar
        if motivo == "falta_explicacion" and intento_actual < MAX_INTENTOS:
            reintentar = True
        elif motivo == "incorrecto" and intento_actual == 0:
            # Permitir solo 1 reintento para "incorrecto" (solo en el primer intento)
            reintentar = True
        else:
            reintentar = False
        print(f"Debug: motivo={motivo}, intento_actual={intento_actual}, MAX_INTENTOS={MAX_INTENTOS}, reintentar={reintentar}")
        
        # Dar retroalimentación en un hilo separado
        hilo_feedback = threading.Thread(target=dar_retroalimentacion, args=(es_correcta, motivo, reintentar, intento_actual), daemon=True)
        hilo_feedback.start()
        
    except sr.WaitTimeoutError:
        print("Tiempo de espera agotado. No se detectó ningún audio.")
        # Si no se detectó audio, dar retroalimentación de incorrecto (NO reintentar)
        hilo_feedback = threading.Thread(target=dar_retroalimentacion, args=(False, "incorrecto", False, intento_actual), daemon=True)
        hilo_feedback.start()
    except sr.UnknownValueError:
        print("No entendí, intenta otra vez")
        # Si no se entendió, dar retroalimentación de incorrecto (NO reintentar)
        hilo_feedback = threading.Thread(target=dar_retroalimentacion, args=(False, "incorrecto", False, intento_actual), daemon=True)
        hilo_feedback.start()
    except sr.RequestError as e:
        print(f"Error al conectar con el servicio de reconocimiento: {e}")
        print("Verifica tu conexión a internet.")
    except Exception as e:
        print(f"Error al escuchar: {e}")

# Variable global para el hilo TTS (se actualizará cuando cambie el absurdo)
hilo_tts = None

# Iniciar el hilo para decir la descripción e instrucción del primer absurdo
# print(f"TTS: iniciando hilo para absurdo {indice_absurdo_actual + 1}")  # Debug comentado
hilo_tts = threading.Thread(target=decir_descripcion_e_instruccion, daemon=False)
hilo_tts.start()
# print(f"TTS: hilo iniciado, estado: {hilo_tts.is_alive()}")  # Debug comentado

# Variable para controlar si ya se inició la escucha
escucha_iniciada = False

# Bucle principal de Pygame
while ejecutando:
    # Manejar eventos
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False
        elif evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_ESCAPE:
                ejecutando = False
    
    # Verificar si se debe mostrar la pantalla final
    if juego_completado:
        mostrar_pantalla_final()
        break  # Salir del bucle después de mostrar la pantalla final
    
    # Iniciar la escucha solo después de que termine de hablar (descripción + instrucción)
    if not escucha_iniciada and hilo_tts is not None and not hilo_tts.is_alive():
        # Esperar un momento adicional para asegurar que el TTS terminó completamente
        time.sleep(0.5)
        # print("STT: comenzando escucha")  # Debug comentado
        hilo_stt = threading.Thread(target=escuchar_respuesta, args=(0,), daemon=False)
        hilo_stt.start()
        escucha_iniciada = True
    
    # Limpiar la ventana con color blanco
    ventana.fill((255, 255, 255))
    
    # Dibujar la imagen centrada (usa imagen_actual que se actualiza cuando cambia el absurdo)
    ventana.blit(imagen_actual, (pos_x_actual, pos_y_actual))
    
    # Mostrar el puntaje en la esquina inferior derecha
    mostrar_puntaje()
    
    # Actualizar la pantalla
    pygame.display.flip()

# Cerrar Pygame
pygame.quit()
sys.exit()

