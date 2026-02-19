"""
Script de prueba para leer JSON y mostrar imagen en Pygame.
Este archivo es solo para pruebas y no modifica main.py.
"""

import pygame
import sys
import os
import json

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

# Obtener el nombre de la imagen del JSON
if "imagen" not in primer_absurdo:
    print("Error: No se encontró la clave 'imagen' en el primer absurdo")
    sys.exit(1)

nombre_imagen = primer_absurdo["imagen"]

# Construir la ruta completa de la imagen
ruta_imagen = os.path.join("assets", "images", nombre_imagen)

# Verificar si existe la imagen
if not os.path.exists(ruta_imagen):
    print(f"Error: No se encontró la imagen en {ruta_imagen}")
    sys.exit(1)

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
ANCHO = 800
ALTO = 600

# Crear la ventana
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Prueba JSON + Imagen - Absurdos Lógicos")

# Cargar la imagen
try:
    imagen = pygame.image.load(ruta_imagen)
except pygame.error as e:
    print(f"Error al cargar la imagen {ruta_imagen}: {e}")
    pygame.quit()
    sys.exit(1)

# Obtener las dimensiones de la imagen
ancho_imagen, alto_imagen = imagen.get_size()

# Calcular la posición para centrar la imagen
pos_x = (ANCHO - ancho_imagen) // 2
pos_y = (ALTO - alto_imagen) // 2

# Bucle principal
ejecutando = True
while ejecutando:
    # Manejar eventos
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False
    
    # Limpiar la ventana con color blanco
    ventana.fill((255, 255, 255))
    
    # Dibujar la imagen centrada
    ventana.blit(imagen, (pos_x, pos_y))
    
    # Actualizar la pantalla
    pygame.display.flip()

# Cerrar Pygame
pygame.quit()
sys.exit()

