"""
Script de prueba para reconocimiento de voz usando speech_recognition.
Este archivo es solo para pruebas y no modifica main.py.
"""

import sys

# Verificar si speech_recognition está instalado
try:
    import speech_recognition as sr
except ImportError:
    print("Error: speech_recognition no está instalado. Instálalo con: pip install SpeechRecognition")
    sys.exit(1)

# Inicializar el reconocedor
recognizer = sr.Recognizer()

# Configurar el micrófono
try:
    microfono = sr.Microphone()
except Exception as e:
    print(f"Error al acceder al micrófono: {e}")
    sys.exit(1)

print("Ajustando ruido ambiente... Por favor, mantén silencio por un momento.")

# Ajustar para el ruido ambiente
# Esto ayuda a mejorar la precisión del reconocimiento
with microfono as source:
    try:
        # Escuchar el ruido ambiente durante 1 segundo para ajustarse
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ruido ambiente ajustado. Ahora puedes hablar...")
    except Exception as e:
        print(f"Error al ajustar el ruido ambiente: {e}")
        sys.exit(1)

print("Escuchando... (5-7 segundos)")

# Escuchar el audio del micrófono
try:
    with microfono as source:
        # Escuchar durante 7 segundos (puedes ajustar este valor)
        audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)
except sr.WaitTimeoutError:
    print("Tiempo de espera agotado. No se detectó ningún audio.")
    sys.exit(1)
except Exception as e:
    print(f"Error al escuchar el micrófono: {e}")
    sys.exit(1)

print("Procesando audio...")

# Intentar reconocer el audio usando Google Speech Recognition
try:
    # Usar recognize_google para convertir voz a texto
    texto = recognizer.recognize_google(audio, language="es-ES")
    print(f"Texto reconocido: {texto}")
except sr.UnknownValueError:
    # No se pudo entender el audio
    print("No entendí, intenta otra vez")
except sr.RequestError as e:
    # Error al hacer la solicitud a la API de Google
    print(f"Error al conectar con el servicio de reconocimiento: {e}")
    print("Verifica tu conexión a internet.")
except Exception as e:
    print(f"Error inesperado: {e}")

