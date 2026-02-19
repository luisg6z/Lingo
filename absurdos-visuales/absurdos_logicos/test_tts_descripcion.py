"""
Script de prueba para leer descripción del JSON y decirla en voz alta con pyttsx3.
Este archivo es solo para pruebas y no modifica main.py.
"""

import json
import os
import sys

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

# Obtener el primer elemento de la lista "absurdos"
if "absurdos" not in datos:
    print("Error: No se encontró la clave 'absurdos' en el JSON")
    sys.exit(1)

if not datos["absurdos"] or len(datos["absurdos"]) == 0:
    print("Error: La lista 'absurdos' está vacía")
    sys.exit(1)

primer_absurdo = datos["absurdos"][0]

# Obtener la descripción del JSON
if "descripcion" not in primer_absurdo:
    print("Error: No se encontró la clave 'descripcion' en el primer absurdo")
    sys.exit(1)

descripcion = primer_absurdo["descripcion"]

# Imprimir el texto que se va a decir
print(f"Diciendo: {descripcion}")

# Inicializar pyttsx3
try:
    import pyttsx3
    engine = pyttsx3.init()
except ImportError:
    print("Error: pyttsx3 no está instalado. Instálalo con: pip install pyttsx3")
    sys.exit(1)
except Exception as e:
    print(f"Error al inicializar pyttsx3: {e}")
    sys.exit(1)

# Configurar propiedades de la voz (opcional)
# Puedes ajustar la velocidad y el volumen si lo deseas
# engine.setProperty('rate', 150)  # Velocidad de habla
# engine.setProperty('volume', 1.0)  # Volumen (0.0 a 1.0)

# Decir la descripción en voz alta
try:
    engine.say(descripcion)
    engine.runAndWait()
except Exception as e:
    print(f"Error al reproducir el texto: {e}")
    sys.exit(1)

print("Texto reproducido correctamente.")

