"""
Script de prueba para mostrar una imagen en Pygame.
Este archivo es solo para pruebas y no modifica main.py.
"""

import pygame
import sys
import os

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
ANCHO = 800
ALTO = 600

# Crear la ventana
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Prueba de Imagen - Absurdos Lógicos")

# Cargar la imagen
ruta_imagen = os.path.join("assets", "images", "manzana_azul.png")
imagen = pygame.image.load(ruta_imagen)

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

