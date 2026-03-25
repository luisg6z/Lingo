"""
Main menu feature for MagicboARd
"""
import cv2
import numpy as np
import pygame
import time
import os
import json
from collections import defaultdict
from openni import openni2

# Import core utilities
import sys
# Get project root (3 levels up from this file: src/features/menu -> src -> project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from src.core.calibration import load_touch_depth_maps
from src.core.ui_utils import scale_to_videobeam, draw_logo
from src.components import draw_rectangular_button

# Import clasificacion feature
from src.features.clasificacion import levels as niveles_clasificacion
from src.features.clasificacion.levels import mostrar_seleccion_niveles_clasificacion

# Import juego-historia feature (folder has hyphen, need to use importlib)
import importlib.util
historia_selection_path = os.path.join(project_root, "src", "features", "juego-historia", "selection.py")
historia_spec = importlib.util.spec_from_file_location("historia_selection", historia_selection_path)
historia_selection_module = importlib.util.module_from_spec(historia_spec)
sys.modules["historia_selection"] = historia_selection_module
historia_spec.loader.exec_module(historia_selection_module)
mostrar_seleccion_escenarios_historia = historia_selection_module.mostrar_seleccion_escenarios_historia

# Cargar mostrar_seleccion_historias desde juego-historia/views (vista de selección de historias con sujetos y botón Siguiente)
_mostrar_seleccion_historias_func = None
def _get_mostrar_seleccion_historias():
    global _mostrar_seleccion_historias_func
    if _mostrar_seleccion_historias_func is None:
        views_init_path = os.path.join(project_root, "src", "features", "juego-historia", "views", "__init__.py")
        views_spec = importlib.util.spec_from_file_location("juego_historia_views", views_init_path)
        views_mod = importlib.util.module_from_spec(views_spec)
        sys.modules["juego_historia_views"] = views_mod
        views_spec.loader.exec_module(views_mod)
        _mostrar_seleccion_historias_func = views_mod.mostrar_seleccion_historias
    return _mostrar_seleccion_historias_func

# Import absurdos-visuales feature (folder has hyphen, need to use importlib)
import importlib.util
absurdos_game_path = os.path.join(project_root, "src", "features", "absurdos-visuales", "game.py")
spec = importlib.util.spec_from_file_location("absurdos_visuales_game", absurdos_game_path)
absurdos_visuales_module = importlib.util.module_from_spec(spec)
sys.modules["absurdos_visuales_game"] = absurdos_visuales_module
spec.loader.exec_module(absurdos_visuales_module)
juego_absurdos_reconocimiento_voz = absurdos_visuales_module.juego_absurdos_reconocimiento_voz

def mostrar_menu_juegos(device, sentence_transformer_model=None):
    # Inicializar Pygame para reproducir sonidos (opcional)
    pygame.init()
    pygame.mixer.init()

    # 1. Cargar Configuraciones
    try:
        config_path = os.path.join(project_root, "src", "config", "ultima_configuracion_coordenadas.json")
        with open(config_path, "r") as file:
            coordenadas = json.load(file)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'ultima_configuracion_coordenadas.json'.")
        return
    except json.JSONDecodeError:
        print("Error: El archivo JSON tiene un formato inválido.")
        return

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Cargar dmax/dmin para detección de toques (banda según perfil menú + JSON opcional)
    dmax_map, dmin_map = load_touch_depth_maps(coordenadas, band_profile="menu")
    dmax_map_available = dmax_map is not None
    if not dmax_map_available:
        print("\n[ADVERTENCIA] No se pudo cargar dmax_map. El menú se mostrará pero la detección de toques no funcionará.")
        print("Ejecuta 'python src/core/calibrate_area.py' o 'python calibrate_area_mejorado.py' para calibrar.\n")
        dmax_map = None
        dmin_map = None

    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800
    
    # Resolución del videobeam (segunda pantalla)
    VIDEOBEAM_WIDTH = 1920
    VIDEOBEAM_HEIGHT = 1080
    

    
    # Crear fondo colorido con degradado para niños
    videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    
    # Crear degradado de colores pastel (azul a morado a rosa)
    for y in range(view_height):
        ratio = y / view_height
        # Degradado de azul claro a morado a rosa
        r = int(255 * (0.3 + 0.4 * ratio))
        g = int(200 * (0.5 + 0.3 * ratio))
        b = int(255 * (0.8 - 0.3 * ratio))
        videobeam_screen[y, :] = [b, g, r]  # BGR
    
    # Dibujar el logo usando la función auxiliar
    logo_loaded = draw_logo(videobeam_screen, view_width, view_height)
    
    # Cargar imágenes de bocina
    bocina_image = None
    bocina_mute_image = None
    if os.path.exists(os.path.join(project_root, "images", "Bocina.png")):
        bocina_image = cv2.imread(os.path.join(project_root, "images", "Bocina.png"), cv2.IMREAD_UNCHANGED)
        if bocina_image is not None:
            print("✓ Imagen de bocina cargada: images/Bocina.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina: images/Bocina.png")
    else:
        print("⚠ No se encontró la imagen: images/Bocina.png")
    
    if os.path.exists(os.path.join(project_root, "images", "BocinaMute.png")):
        bocina_mute_image = cv2.imread(os.path.join(project_root, "images", "BocinaMute.png"), cv2.IMREAD_UNCHANGED)
        if bocina_mute_image is not None:
            print("✓ Imagen de bocina mute cargada: images/BocinaMute.png")
        else:
            print("⚠ No se pudo cargar la imagen de bocina mute: images/BocinaMute.png")
    else:
        print("⚠ No se encontró la imagen: images/BocinaMute.png")
    
    # Estado de la card de bocina (muteada o no)
    # Usar las variables globales del módulo niveles_clasificacion para mantener el estado del audio entre vistas
    # Cargar y configurar el audio de fondo usando mixer.music (solo si no está cargado)
    if not niveles_clasificacion._background_music_loaded:
        archivo = os.path.join(project_root, "relax-meditate-gentle-peaceful-291162.mp3")
        if os.path.exists(archivo):
            try:
                pygame.mixer.music.load(archivo)
                # El módulo music suele responder mejor a volúmenes bajos
                pygame.mixer.music.set_volume(0.05)  # 5% de volumen
                niveles_clasificacion._background_music_loaded = True
                print("✓ Audio de fondo cargado: relax-meditate-gentle-peaceful-291162.mp3 (volumen al 5%)")
                # Iniciar el audio automáticamente si no está muteado
                if not niveles_clasificacion._bocina_muted_global:
                    pygame.mixer.music.play(-1)  # -1 significa bucle infinito
                    print("✓ Audio de fondo iniciado automáticamente")
            except Exception as e:
                print(f"⚠ No se pudo cargar el audio de fondo: {e}")
        else:
            print("⚠ No se encontró el archivo de audio: relax-meditate-gentle-peaceful-291162.mp3")
    
    # Usar el estado global del audio
    bocina_muted = niveles_clasificacion._bocina_muted_global
    
    # Función para dibujar card cuadrada con icono de bocina (estilo infantil) - amarilla con bocina
    def draw_close_card_main(screen, elevated=False, muted=False):
        """
        Dibuja una card cuadrada con icono de bocina en el centro, estilo infantil, amarilla con bocina
        
        Args:
            screen: Pantalla donde dibujar
            elevated: Si True, la card se dibuja elevada (efecto de levantarse)
            muted: Si True, muestra la imagen de bocina muteada (BocinaMute.png), si False muestra Bocina.png
        """
        # Posición base del lado derecho (parte inferior, misma posición X que tenía la card amarilla)
        base_card_size = 100  # Tamaño del cuadrado (2 * radio de 50 para mantener el mismo tamaño)
        card_margin_x = 180  # Mismo margen X que tenía la card amarilla
        card_margin_y = 50  # Margen desde el borde inferior (misma posición Y que tenía la card amarilla)
        
        # Sin efecto de elevación - siempre el mismo tamaño y posición
        card_size = base_card_size
        # Calcular posición del cuadrado (esquina superior izquierda)
        card_x = view_width - card_margin_x - card_size
        card_y = view_height - card_margin_y - card_size
        
        # Dibujar la imagen completa como fondo de la card (usar imagen mute si está muteada)
        # NOTA: Se eliminó la sombra negra como se solicitó
        current_bocina_image = bocina_mute_image if muted and bocina_mute_image is not None else bocina_image
        if current_bocina_image is not None:
            # Redimensionar la imagen para que llene completamente la card
            bocina_resized = cv2.resize(current_bocina_image, (card_size, card_size), interpolation=cv2.INTER_AREA)
            
            # Asegurar que esté dentro de los límites
            if card_x >= 0 and card_y >= 0 and card_x + card_size <= screen.shape[1] and card_y + card_size <= screen.shape[0]:
                # Si la imagen tiene canal alfa (transparencia)
                if len(bocina_resized.shape) == 3 and bocina_resized.shape[2] == 4:
                    # Extraer canal alfa
                    alpha = bocina_resized[:, :, 3] / 255.0
                    # Convertir BGR de la imagen
                    img_bgr = bocina_resized[:, :, :3]
                    # Mezclar con el fondo
                    for c in range(3):
                        screen[card_y:card_y+card_size, card_x:card_x+card_size, c] = (
                            alpha * img_bgr[:, :, c] + 
                            (1 - alpha) * screen[card_y:card_y+card_size, card_x:card_x+card_size, c]
                        )
                else:
                    # Sin canal alfa, copiar directamente
                    screen[card_y:card_y+card_size, card_x:card_x+card_size] = bocina_resized[:, :, :3]
    
    # Variables para la card de cerrar (necesarias para la detección)
    # Usar un área rectangular para la detección, similar a las otras cards
    close_card_size_main = 100  # Tamaño del cuadrado
    close_card_margin_x_main = 180
    close_card_margin_y_main = 50  # Margen desde el borde inferior
    close_card_x_main = view_width - close_card_margin_x_main - close_card_size_main
    close_card_y_main = view_height - close_card_margin_y_main - close_card_size_main  # Posición desde abajo
    
    # Crear un área rectangular de detección (más grande que el cuadrado para facilitar el toque)
    close_card_detection_size_main = int(close_card_size_main * 1.2)  # Área más grande para facilitar el toque
    close_card_detection_x_main = close_card_x_main - int(close_card_size_main * 0.1)
    close_card_detection_y_main = close_card_y_main - int(close_card_size_main * 0.1)
    close_card_detection_w_main = close_card_detection_size_main
    close_card_detection_h_main = close_card_detection_size_main
    
    # Función para detectar si se tocó la card de cerrar (usando área rectangular como las otras cards)
    def detectar_close_card_touch_main(x_touch, y_touch):
        """Detecta si el toque está dentro del área de la card de cerrar (usando área rectangular)"""
        return (close_card_detection_x_main <= x_touch <= close_card_detection_x_main + close_card_detection_w_main and
                close_card_detection_y_main <= y_touch <= close_card_detection_y_main + close_card_detection_h_main)

    # 2. Definir los 3 juegos ficticios (sin acentos en los nombres)
    juegos = [
        {
            "nombre": "Juego de Clasificacion",
            "color": (100, 200, 255),  # Azul claro (BGR)
            "descripcion": "Clasifica objetos",
            "icono": "📦",
            "imagen": os.path.join(project_root, "images", "LogoClasificacion.png")  # Ruta de la imagen del logo
        },
        {
            "nombre": "Absurdos Logicos",
            "color": (100, 255, 150),  # Verde claro (BGR)
            "descripcion": "Encuentra lo absurdo",
            "icono": "🤔",
            "imagen": os.path.join(project_root, "images", "LogoAbsurdosVisuales.png")  # Ruta de la imagen del logo
        },
        {
            "nombre": "Historias",
            "color": (255, 150, 200),  # Rosa claro (BGR)
            "descripcion": "Crea historias",
            "icono": "📚",
            "imagen": os.path.join(project_root, "images", "LogoHistoria.png")  # Ruta de la imagen del logo
        }
    ]
    
    # Cargar imágenes de los juegos si existen
    juego_images = {}
    for juego in juegos:
        if "imagen" in juego and os.path.exists(juego["imagen"]):
            img = cv2.imread(juego["imagen"], cv2.IMREAD_UNCHANGED)
            if img is not None:
                juego_images[juego["nombre"]] = img
                print(f"✓ Imagen cargada para {juego['nombre']}: {juego['imagen']}")
            else:
                print(f"⚠ No se pudo cargar la imagen para {juego['nombre']}: {juego['imagen']}")
    
    # 3. Calcular posiciones de las cards (centradas verticalmente)
    card_width = 320
    card_height = 400
    card_spacing = 50
    total_cards_width = len(juegos) * card_width + (len(juegos) - 1) * card_spacing
    start_x = (view_width - total_cards_width) // 2
    start_y = 200  # Posición vertical para las cards
    
    game_positions = {}
    for idx, juego in enumerate(juegos):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        game_positions[juego["nombre"]] = {
            'x': x,
            'y': y,
            'width': card_width,
            'height': card_height,
            'color': juego["color"],
            'descripcion': juego["descripcion"],
            'icono': juego["icono"],
            'imagen': juego_images.get(juego["nombre"])  # Imagen cargada si existe
        }
    
    # 4. Función para dibujar las cards de juegos
    def draw_game_cards(screen, game_positions, elevated_card=None):
        """
        Dibuja las cards de juegos en la pantalla.
        
        Args:
            screen: La imagen donde dibujar
            game_positions: Diccionario con las posiciones de las cards
            elevated_card: Nombre de la card que debe estar elevada (None si ninguna)
        """
        for nombre, pos in game_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            color = pos['color']
            
            # Efecto de elevación si esta card está seleccionada
            elevation_offset = 0
            scale_factor = 1.0
            shadow_offset_base = 8
            
            if elevated_card == nombre:
                elevation_offset = -20  # Mover hacia arriba
                scale_factor = 1.05  # Aumentar tamaño ligeramente
                shadow_offset_base = 15  # Sombra más grande para mayor profundidad
                # Hacer el color más brillante cuando está elevada
                color = tuple(min(255, int(c * 1.15)) for c in color)
            
            # Aplicar escala
            w_scaled = int(w * scale_factor)
            h_scaled = int(h * scale_factor)
            x_scaled = x - (w_scaled - w) // 2  # Centrar el escalado
            y_scaled = y + elevation_offset - (h_scaled - h) // 2
            
            # Asegurar que no se salga de los límites
            x_scaled = max(0, min(x_scaled, screen.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, screen.shape[0] - h_scaled))
            
            # Dibujar sombra (más grande si está elevada)
            shadow_offset = int(shadow_offset_base * scale_factor)
            shadow_color = (50, 50, 50)
            cv2.rectangle(screen, (x_scaled + shadow_offset, y_scaled + shadow_offset), 
                        (x_scaled + w_scaled + shadow_offset, y_scaled + h_scaled + shadow_offset), 
                        shadow_color, -1)
            
            # Verificar si esta card tiene imagen para usar como fondo completo
            if 'imagen' in pos and pos['imagen'] is not None:
                # Usar la imagen como fondo completo de la card
                img = pos['imagen'].copy()
                
                # Redimensionar la imagen para que llene completamente la card
                img_resized = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                
                # Verificar límites antes de dibujar
                if x_scaled >= 0 and y_scaled >= 0 and x_scaled + w_scaled <= screen.shape[1] and y_scaled + h_scaled <= screen.shape[0]:
                    # Si la imagen tiene canal alfa (transparencia)
                    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                        # Extraer canal alfa
                        alpha = img_resized[:, :, 3] / 255.0
                        # Convertir BGR de la imagen
                        img_bgr = img_resized[:, :, :3]
                        # Mezclar con el fondo
                        for c in range(3):
                            screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c] = (
                                alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled, c]
                            )
                    else:
                        # Sin canal alfa, copiar directamente
                        screen[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = img_resized[:, :, :3]
                
                # No dibujar nombre ni descripción si la imagen ya los incluye
                # (Opcional: puedes comentar estas líneas si quieres que NO se muestren el nombre y descripción)
                continue  # Saltar el resto del dibujado para esta card si tiene imagen
            else:
                # Comportamiento original para cards sin imagen
                # Dibujar card con bordes redondeados (simulado con rectángulos)
                # Fondo de la card
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
                
                # Borde de la card (más oscuro, más grueso si está elevada)
                border_color = tuple(max(0, c - 30) for c in color)
                border_thickness = 7 if elevated_card == nombre else 5
                cv2.rectangle(screen, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), border_color, border_thickness)
                
                # Dibujar efecto de brillo en la parte superior (más brillante si está elevada)
                highlight_boost = 60 if elevated_card == nombre else 40
                highlight_color = tuple(min(255, c + highlight_boost) for c in color)
                highlight_margin = int(10 * scale_factor)
                cv2.rectangle(screen, (x_scaled + highlight_margin, y_scaled + highlight_margin), 
                            (x_scaled + w_scaled - highlight_margin, y_scaled + int(50 * scale_factor) + highlight_margin), 
                            highlight_color, -1)
                
                # Ajustar posiciones de texto según el escalado
                icon_x_offset = (w_scaled - w) // 2
                icon_y_offset = (h_scaled - h) // 2
                
                # Dibujar icono emoji (comportamiento original)
                icono = pos['icono']
                font_icon = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_icon = 3.0 * scale_factor  # Escalar el icono también
                thickness_icon = int(3 * scale_factor)
                icon_size, _ = cv2.getTextSize(icono, font_icon, font_scale_icon, thickness_icon)
                icon_x = x_scaled + (w_scaled - icon_size[0]) // 2
                icon_y = y_scaled + int(100 * scale_factor) + icon_y_offset
                cv2.putText(screen, icono, (icon_x, icon_y), font_icon, font_scale_icon, (255, 255, 255), thickness_icon)
            
            # Dibujar nombre del juego (nombre es la clave del diccionario)
            # Ajustar el texto para que quepa dentro de la card
            nombre_texto = nombre
            font_nombre = cv2.FONT_HERSHEY_DUPLEX
            font_scale_nombre = 0.9
            thickness_nombre = 2
            
            # Calcular el ancho disponible (con margen de 20 píxeles a cada lado)
            available_width = w - 40
            
            # Verificar si el texto cabe, si no, reducir el tamaño de fuente
            nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            while nombre_size[0] > available_width and font_scale_nombre > 0.5:
                font_scale_nombre -= 0.1
                nombre_size, _ = cv2.getTextSize(nombre_texto, font_nombre, font_scale_nombre, thickness_nombre)
            
            # Si aún no cabe, dividir en múltiples líneas
            if nombre_size[0] > available_width:
                # Dividir el texto en palabras y crear líneas
                palabras = nombre_texto.split()
                lineas = []
                linea_actual = ""
                for palabra in palabras:
                    test_linea = linea_actual + (" " if linea_actual else "") + palabra
                    test_size, _ = cv2.getTextSize(test_linea, font_nombre, font_scale_nombre, thickness_nombre)
                    if test_size[0] <= available_width:
                        linea_actual = test_linea
                    else:
                        if linea_actual:
                            lineas.append(linea_actual)
                        linea_actual = palabra
                if linea_actual:
                    lineas.append(linea_actual)
                
                # Dibujar cada línea centrada
                line_height = nombre_size[1] + 5
                start_y = y_scaled + int(180 * scale_factor) + icon_y_offset - (len(lineas) - 1) * line_height // 2
                for i, linea in enumerate(lineas):
                    linea_size, _ = cv2.getTextSize(linea, font_nombre, font_scale_nombre, thickness_nombre)
                    linea_x = x_scaled + (w_scaled - linea_size[0]) // 2
                    linea_y = start_y + i * line_height
                    # Sombra del texto
                    cv2.putText(screen, linea, (linea_x + 2, linea_y + 2), 
                               font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                    # Texto principal
                    cv2.putText(screen, linea, (linea_x, linea_y), 
                               font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            else:
                # El texto cabe en una línea, dibujarlo normalmente
                nombre_x = x_scaled + (w_scaled - nombre_size[0]) // 2
                nombre_y = y_scaled + int(180 * scale_factor) + icon_y_offset
                # Sombra del texto
                cv2.putText(screen, nombre_texto, (nombre_x + 2, nombre_y + 2), 
                           font_nombre, font_scale_nombre, (0, 0, 0), thickness_nombre + 1)
                # Texto principal
                cv2.putText(screen, nombre_texto, (nombre_x, nombre_y), 
                           font_nombre, font_scale_nombre, (255, 255, 255), thickness_nombre)
            
            # Dibujar descripción (ajustar si es muy larga)
            desc_texto = pos['descripcion']
            font_desc = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_desc = 0.7 * scale_factor
            thickness_desc = int(2 * scale_factor)
            
            # Calcular el ancho disponible (con margen de 20 píxeles a cada lado)
            available_width_desc = int(w_scaled - 40)
            
            # Verificar si el texto cabe, si no, reducir el tamaño de fuente
            desc_size, _ = cv2.getTextSize(desc_texto, font_desc, font_scale_desc, thickness_desc)
            while desc_size[0] > available_width_desc and font_scale_desc > 0.4:
                font_scale_desc -= 0.05
                desc_size, _ = cv2.getTextSize(desc_texto, font_desc, font_scale_desc, thickness_desc)
            
            desc_x = x_scaled + (w_scaled - desc_size[0]) // 2
            desc_y = y_scaled + int(220 * scale_factor) + icon_y_offset
            cv2.putText(screen, desc_texto, (desc_x, desc_y), 
                       font_desc, font_scale_desc, (255, 255, 255), thickness_desc)
    
    # 5. Crear y configurar la ventana ANTES de dibujar (igual que calibrate_area.py)
    # Crear una ventana para la proyección
    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Menú de Juegos", 1920, 0)
    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Dibujar las cards en la pantalla
    draw_game_cards(videobeam_screen, game_positions)
    # Dibujar card de bocina en la parte inferior derecha
    draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)

    # Escalar a la resolución del videobeam antes de mostrar
    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
    cv2.imshow("Menú de Juegos", videobeam_screen_scaled)

    # 6. Función para detectar el juego seleccionado
    def detectar_juego_seleccionado(x_touch, y_touch, game_positions):
        """
        Dado un punto de toque (x_touch, y_touch), determina qué juego ha sido seleccionado.
        """
        for nombre, pos in game_positions.items():
            x, y = pos['x'], pos['y']
            w, h = pos['width'], pos['height']
            if x <= x_touch <= x + w and y <= y_touch <= y + h:
                return nombre
        return None
    
    # 7. La función de selección de niveles ahora está en vista_niveles_clasificacion.py
    
    # 7b. Funciones placeholder para otros juegos
    def mostrar_mensaje_juego(nombre_juego):
        """Muestra un mensaje cuando se selecciona un juego (placeholder)"""
        mensaje_screen = videobeam_screen.copy()
        
        # Fondo semi-transparente
        overlay = mensaje_screen.copy()
        cv2.rectangle(overlay, (0, 0), (view_width, view_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, mensaje_screen, 0.3, 0, mensaje_screen)
        
        # Mensaje principal
        texto_principal = f"¡{nombre_juego}!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.0
        thickness = 4
        text_size, _ = cv2.getTextSize(texto_principal, font, font_scale, thickness)
        text_x = (view_width - text_size[0]) // 2
        text_y = view_height // 2 - 50
        
        # Sombra del texto
        cv2.putText(mensaje_screen, texto_principal, (text_x + 3, text_y + 3), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        # Texto principal
        cv2.putText(mensaje_screen, texto_principal, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Mensaje secundario
        texto_secundario = "Este juego estará disponible pronto"
        font_sec = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_sec = 1.0
        thickness_sec = 2
        text_size_sec, _ = cv2.getTextSize(texto_secundario, font_sec, font_scale_sec, thickness_sec)
        text_x_sec = (view_width - text_size_sec[0]) // 2
        text_y_sec = view_height // 2 + 50
        
        cv2.putText(mensaje_screen, texto_secundario, (text_x_sec, text_y_sec), 
                   font_sec, font_scale_sec, (255, 255, 255), thickness_sec)
        
        # Escalar a la resolución del videobeam antes de mostrar
        mensaje_screen_scaled = scale_to_videobeam(mensaje_screen)
        cv2.imshow("Menú de Juegos", mensaje_screen_scaled)
        cv2.waitKey(50)
        cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        cv2.waitKey(2000)  # Mostrar por 2 segundos

    # 8. Iniciar streams de cámara para detección de toques (solo si dmax_map está disponible)
    if dmax_map_available:
        rgb_stream = device.create_color_stream()
        depth_stream = device.create_depth_stream()
        rgb_stream.start()
        depth_stream.start()

        juego_seleccionado_flag = False  # Bandera para evitar múltiples selecciones
        frame_count = 0
        initialization_delay = 10  # Reducido a 10 frames para respuesta más rápida
        
        # Sistema de debounce temporal para evitar falsos positivos
        from collections import defaultdict
        touch_history = defaultdict(list)  # Historial de toques por posición
        min_touch_frames = 2  # Requiere que el toque persista por al menos 2 frames consecutivos
        touch_persistence_threshold = 0.8  # 80% de los frames deben tener el toque
        min_touch_area = 100  # Área mínima para filtrar ruido pequeño pero permitir toques reales
        max_touch_area = 50000  # Área máxima para evitar detecciones de objetos grandes
        last_valid_touch_time = time.time()
        touch_cooldown = 0.15  # Cooldown de 150ms entre toques válidos
        history_cleanup_interval = 30  # Limpiar historial cada 30 frames
        max_history_age = 1.0  # Eliminar entradas del historial más antiguas de 1 segundo
        inactivity_threshold = 5.0  # Si no hay toques válidos en 5 segundos, ser más estricto

        # 9. Bucle principal de detección de toques
        while True:
            juego_seleccionado_flag = False
            frame_count += 1
            frame = rgb_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            if frame is None or depth_frame is None:
                continue

            rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            bgr_data = cv2.flip(bgr_data, 1)
            bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

            # Crear la máscara que considera solo los valores entre dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

            # Aplicar filtros más suaves para preservar toques reales (igual que calibrate_area.py)
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
            
            # Aplicar apertura morfológica para eliminar ruido pequeño
            kernel = np.ones((2, 2), np.uint8)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
            
            # También aplicar cierre para conectar áreas cercanas
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)

            # No procesar toques durante el delay inicial
            if frame_count < initialization_delay:
                draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # Encontrar los contornos de los toques (usar la máscara filtrada)
            contours, _ = cv2.findContours(touch_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Limpiar historial antiguo periódicamente
            current_time = time.time()
            if frame_count % history_cleanup_interval == 0:
                # Eliminar entradas del historial más antiguas de max_history_age segundos
                for key in list(touch_history.keys()):
                    touch_history[key] = [
                        touch for touch in touch_history[key] 
                        if current_time - touch[3] < max_history_age
                    ]
                    # Si el historial está vacío, eliminarlo
                    if not touch_history[key]:
                        del touch_history[key]
            
            # Limitar el tamaño del historial por posición
            for key in list(touch_history.keys()):
                if len(touch_history[key]) > min_touch_frames + 5:
                    touch_history[key] = touch_history[key][-(min_touch_frames + 5):]

            # Procesar cada contorno
            valid_touches_this_frame = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Validar que el área esté en el rango esperado
                if min_touch_area <= area <= max_touch_area:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Mapeo de coordenadas de ventana a viewport
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
                        
                        # Agregar a historial (usar coordenadas discretizadas para agrupar toques cercanos)
                        touch_key = (x_touch // 25, y_touch // 25)  # Agrupar toques dentro de 25 píxeles
                        touch_history[touch_key].append((x_touch, y_touch, area, current_time))
                        
                        # Verificar si el toque ha persistido lo suficiente
                        if len(touch_history[touch_key]) >= min_touch_frames:
                            # Verificar cooldown
                            if current_time - last_valid_touch_time > touch_cooldown:
                                # Determinar si ha habido inactividad reciente
                                time_since_last_touch = current_time - last_valid_touch_time
                                is_inactive = time_since_last_touch > inactivity_threshold
                                
                                # Si hay inactividad, ser más estricto (requerir más frames y área más grande)
                                required_frames = min_touch_frames + (2 if is_inactive else 0)
                                required_min_area = int(min_touch_area * (1.5 if is_inactive else 1.0))
                                
                                # Obtener los últimos required_frames toques (o todos si hay menos)
                                available_touches = len(touch_history[touch_key])
                                if available_touches >= required_frames:
                                    recent_touches = touch_history[touch_key][-required_frames:]
                                    
                                    # Validar que todos los toques sean recientes
                                    time_window = 0.8 if is_inactive else 1.0
                                    all_recent = all(current_time - touch[3] < time_window for touch in recent_touches)
                                    
                                    # Validar que el área sea razonablemente consistente
                                    if all_recent and len(recent_touches) >= required_frames:
                                        areas = [touch[2] for touch in recent_touches]
                                        avg_area = sum(areas) / len(areas)
                                        
                                        # Validación: área promedio debe estar en rango válido
                                        # Si hay inactividad, ser más estricto con la variación
                                        if min(areas) > 0:
                                            area_variance = max(areas) / min(areas)
                                            max_variance = 2.5 if is_inactive else 3.5
                                            if area_variance < max_variance and required_min_area <= avg_area <= max_touch_area:
                                                valid_touches_this_frame.append((x_touch, y_touch, touch_key))
                                        elif required_min_area <= avg_area <= max_touch_area:
                                            # Si no hay variación suficiente, solo permitir si está en rango y no hay inactividad
                                            if not is_inactive:
                                                valid_touches_this_frame.append((x_touch, y_touch, touch_key))
            
            # Procesar solo los toques válidos (que han persistido lo suficiente)
            for x_touch, y_touch, touch_key in valid_touches_this_frame:
                # Limpiar el historial de este toque después de procesarlo
                if touch_key in touch_history:
                    del touch_history[touch_key]
                last_valid_touch_time = current_time

                # Primero verificar si se tocó la card de bocina
                if detectar_close_card_touch_main(x_touch, y_touch):
                    print("Card de bocina tocada en menú principal")
                    
                    # Cambiar el estado de mute (usando variable global del módulo niveles_clasificacion)
                    niveles_clasificacion._bocina_muted_global = not niveles_clasificacion._bocina_muted_global
                    bocina_muted = niveles_clasificacion._bocina_muted_global
                    print(f"Bocina {'muteada' if bocina_muted else 'activada'}")
                    
                    # Controlar el audio según el estado
                    if niveles_clasificacion._background_music_loaded:
                        if bocina_muted:
                            # Detener el audio cuando está muteada
                            pygame.mixer.music.stop()
                            print("Audio de fondo detenido")
                        else:
                            # Asegurar que el volumen esté al 1% antes de reproducir
                            pygame.mixer.music.set_volume(0.05)
                            # Reproducir el audio en bucle cuando está activada
                            pygame.mixer.music.play(-1)  # -1 significa bucle infinito
                            print("Audio de fondo iniciado (bucle)")
                    
                    # Redibujar la pantalla con el nuevo estado (sin efecto de elevación)
                    temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                    for y in range(view_height):
                        ratio = y / view_height
                        r = int(255 * (0.3 + 0.4 * ratio))
                        g = int(200 * (0.5 + 0.3 * ratio))
                        b = int(255 * (0.8 - 0.3 * ratio))
                        temp_screen[y, :] = [b, g, r]
                    draw_logo(temp_screen)
                    draw_game_cards(temp_screen, game_positions)
                    draw_close_card_main(temp_screen, elevated=False, muted=bocina_muted)
                    videobeam_screen = temp_screen
                    videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                    cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                    continue
                
                # Detectar si se seleccionó un juego
                if not juego_seleccionado_flag:
                    juego_seleccionado = detectar_juego_seleccionado(x_touch, y_touch, game_positions)
                    if juego_seleccionado:
                        print(f"Juego seleccionado: '{juego_seleccionado}' (tipo: {type(juego_seleccionado)})")
                        print(f"Comparación con 'Historias': {juego_seleccionado == 'Historias'}")
                        print(f"Comparación con 'Juego de Clasificacion': {juego_seleccionado == 'Juego de Clasificacion'}")
                        print(f"Comparación con 'Absurdos Logicos': {juego_seleccionado == 'Absurdos Logicos'}")
                        
                        # Mostrar efecto de elevación de la card (animación rápida)
                        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                        for y in range(view_height):
                            ratio = y / view_height
                            r = int(255 * (0.3 + 0.4 * ratio))
                            g = int(200 * (0.5 + 0.3 * ratio))
                            b = int(255 * (0.8 - 0.3 * ratio))
                            temp_screen[y, :] = [b, g, r]
                        draw_logo(temp_screen, view_width, view_height)
                        draw_game_cards(temp_screen, game_positions, elevated_card=juego_seleccionado)
                        draw_close_card_main(temp_screen, elevated=False, muted=bocina_muted)
                        temp_screen_scaled = scale_to_videobeam(temp_screen)
                        cv2.imshow("Menú de Juegos", temp_screen_scaled)
                        cv2.waitKey(100)  # Pausa breve antes de cambiar
                        
                        # Si es el juego de Clasificación, mostrar selección de niveles
                        if juego_seleccionado == "Juego de Clasificacion":
                            # No cerrar la ventana, reutilizarla para transición suave
                            nivel_seleccionado = mostrar_seleccion_niveles_clasificacion(
                                device, coordenadas, dmax_map, dmin_map, draw_logo,
                                existing_window_name="Menú de Juegos"
                            )
                            if nivel_seleccionado:
                                print(f"Procesando nivel seleccionado: {nivel_seleccionado}")
                                # Aquí puedes agregar la lógica para iniciar el juego con el nivel seleccionado
                                # Por ejemplo: juego_clasificacion(device, nivel=nivel_seleccionado, ...)
                            
                            # Volver al menú principal después de seleccionar nivel o cancelar (incluyendo cuando se presiona X)
                            # Recrear el menú
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen, view_width, view_height)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break
                        # Si es Absurdos Logicos, ir directamente al juego de absurdos visuales
                        elif juego_seleccionado == "Absurdos Logicos":
                            # Reutilizar la misma ventana del menú en el videobeam (como en Historias).
                            # Así al dar X el juego no cierra la ventana y aquí la actualizamos con el menú.
                            juego_absurdos_reconocimiento_voz(
                                device, coordenadas, dmax_map, dmin_map, sentence_transformer_model,
                                existing_window_name="Menú de Juegos"
                            )
                            
                            # El juego retornó, volver al menú principal SOLO aquí (cuando el juego termina).
                            # Recrear el menú desde cero.
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen, view_width, view_height)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True  # Establecer flag para salir del bucle de detección
                            break  # Salir del bucle de detección de toques para mostrar el menú principal
                        # Si es Historias, mostrar vista de selección de historias (sujetos, acciones, lugares, botón Siguiente)
                        elif juego_seleccionado == "Historias":
                            print(f"Redirigiendo a selección de historias para: {juego_seleccionado}")
                            try:
                                resultado_seleccion = _get_mostrar_seleccion_historias()(
                                    device, coordenadas, dmax_map, dmin_map, draw_logo,
                                    existing_window_name="Menú de Juegos"
                                )
                                if resultado_seleccion:
                                    if isinstance(resultado_seleccion, dict):
                                        sujetos_seleccionados = resultado_seleccion.get('sujetos', [])
                                        acciones_seleccionadas = resultado_seleccion.get('acciones', [])
                                        lugares_seleccionados = resultado_seleccion.get('lugares', [])
                                        print(f"Sujetos seleccionados: {sujetos_seleccionados}")
                                        print(f"Acciones seleccionadas: {acciones_seleccionadas}")
                                        print(f"Lugares seleccionados: {lugares_seleccionados}")
                                    elif isinstance(resultado_seleccion, list):
                                        print(f"Sujetos seleccionados: {resultado_seleccion}")
                            except Exception as e:
                                print(f"Error al llamar a mostrar_seleccion_historias: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Volver al menú principal después de seleccionar escenario o cancelar
                            # Recrear el menú
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            draw_logo(videobeam_screen, view_width, view_height)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            # Verificar si la ventana existe antes de recrearla
                            try:
                                prop = cv2.getWindowProperty("Menú de Juegos", cv2.WND_PROP_VISIBLE)
                                if prop < 0:
                                    # La ventana no existe, crearla
                                    cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                    cv2.moveWindow("Menú de Juegos", 1920, 0)
                                    cv2.waitKey(50)
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                                else:
                                    # La ventana existe, solo asegurar que esté en pantalla completa
                                    cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                    cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            except:
                                # Si hay error, crear la ventana
                                cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
                                cv2.moveWindow("Menú de Juegos", 1920, 0)
                                cv2.waitKey(50)
                                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            cv2.waitKey(50)
                            cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break
                        else:
                            print(f"Juego no reconocido: '{juego_seleccionado}'. Mostrando mensaje de 'disponible pronto'")
                            mostrar_mensaje_juego(juego_seleccionado)
                            
                            # Redibujar el menú después del mensaje
                            videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
                            # Recrear el fondo degradado
                            for y in range(view_height):
                                ratio = y / view_height
                                r = int(255 * (0.3 + 0.4 * ratio))
                                g = int(200 * (0.5 + 0.3 * ratio))
                                b = int(255 * (0.8 - 0.3 * ratio))
                                videobeam_screen[y, :] = [b, g, r]
                            # Redibujar logo y cards
                            draw_logo(videobeam_screen, view_width, view_height)
                            draw_game_cards(videobeam_screen, game_positions)
                            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
                            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
                            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
                            # Reiniciar el contador de frames para evitar detecciones inmediatas
                            frame_count = 0
                            juego_seleccionado_flag = True
                            break

            # Asegurar que la card de bocina esté dibujada en cada frame
            # Actualizar el estado de la bocina desde la variable global (por si cambió en otra vista)
            bocina_muted = niveles_clasificacion._bocina_muted_global
            
            draw_close_card_main(videobeam_screen, elevated=False, muted=bocina_muted)
            # Escalar a la resolución del videobeam antes de mostrar
            videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
            # Mostrar la ventana
            cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
            # Forzar pantalla completa en cada frame
            cv2.waitKey(10)
            try:
                cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
            except:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Detener los streams de la cámara y cerrar las ventanas
        rgb_stream.stop()
        depth_stream.stop()
        cv2.destroyAllWindows()
    else:
        # Si no hay dmax_map, solo mostrar el menú sin detección de toques
        print("\n[INFO] El menú se mostrará pero la detección de toques no estará disponible.")
        print("       Ejecuta la calibración para habilitar la detección de toques.\n")
        
        # Mostrar el menú estático
        videobeam_screen_scaled = scale_to_videobeam(videobeam_screen)
        cv2.imshow("Menú de Juegos", videobeam_screen_scaled)
        # Forzar pantalla completa
        cv2.waitKey(50)
        cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Menú de Juegos", VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT)
        
        # Esperar hasta que se presione 'q'
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    # Función auxiliar (no usada pero mantenida para compatibilidad)
    def seleccionar_opciones_clasificacion():
        """
        Muestra opciones para seleccionar el modo de clasificación y el tipo de piezas.
        Si se selecciona "Virtuales", permite ajustar el número de fichas.
        """
        # Crear una pantalla negra para las opciones
        opciones_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)

        # Definir las opciones
        opciones_modo = ["Figuras", "Colores"]
        opciones_tipo = ["Virtuales", "Físicas"]

        # Configuración de botones
        button_width = 200   # Ancho de los botones
        button_height = 80   # Alto de los botones
        padding_x = 50
        padding_y = 50
        spacing_x = 50
        spacing_y = 30

        # Calcular el ancho y alto disponibles dentro del área de trabajo
        available_width = xv_max - xv_min - 2 * padding_x
        available_height = yv_max - yv_min - 2 * padding_y

        # Calcular posición X inicial para centrar las columnas
        total_buttons_width = button_width * 2 + spacing_x
        x_start = xv_min + (available_width - total_buttons_width) // 2 + padding_x

        # Posición inicial en Y para los botones de modo y tipo
        initial_y = yv_min + padding_y + 50  # Añadimos 50 para espacio del encabezado

        # Posiciones de los botones de modo y tipo
        positions = {}
        for idx in range(max(len(opciones_modo), len(opciones_tipo))):
            # Columna 1: Opciones de modo
            if idx < len(opciones_modo):
                opcion = opciones_modo[idx]
                x = x_start
                y = initial_y + idx * (button_height + spacing_y)
                positions[opcion] = (x, y)
            # Columna 2: Opciones de tipo
            if idx < len(opciones_tipo):
                opcion = opciones_tipo[idx]
                x = x_start + button_width + spacing_x
                y = initial_y + idx * (button_height + spacing_y)
                positions[opcion] = (x, y)

        # Inicializamos num_piezas pero no mostramos los controles hasta que se seleccione "Virtuales"
        num_piezas = 5  # Valor inicial
        num_piezas_min = 5
        num_piezas_max = 12

        # Bandera para indicar si se debe mostrar la selección de número de piezas
        mostrar_num_piezas = False

        # Dibujar los elementos en la pantalla
        def draw_elements(screen):
            # Limpiar pantalla
            screen[:] = 0

            fuente_encabezado = cv2.FONT_HERSHEY_SIMPLEX
            escala_fuente_encabezado = 1.0
            color_encabezado = (255, 255, 255)
            grosor_encabezado = 2

            # Dibujar encabezados para los botones de modo y tipo
            # Encabezado de la columna de modo
            encabezado_modo = "Modo de Juego"
            tamaño_texto, _ = cv2.getTextSize(encabezado_modo, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
            x_text_modo = x_start + (button_width - tamaño_texto[0]) // 2
            y_text_modo = initial_y - 20  # Ajustar para que quede encima
            cv2.putText(screen, encabezado_modo, (x_text_modo, y_text_modo), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

            # Encabezado de la columna de tipo
            encabezado_tipo = "Tipo de Piezas"
            tamaño_texto, _ = cv2.getTextSize(encabezado_tipo, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
            x_text_tipo = x_start + button_width + spacing_x + (button_width - tamaño_texto[0]) // 2
            y_text_tipo = initial_y - 20  # Ajustar para que quede encima
            cv2.putText(screen, encabezado_tipo, (x_text_tipo, y_text_tipo), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

            # Dibujar los botones de modo y tipo usando componente
            for opcion, (x, y) in positions.items():
                # Dibujar botón con borde blanco y sin relleno
                draw_rectangular_button(
                    screen,
                    x, y, button_width, button_height,
                    opcion,
                    bg_color=(0, 0, 0),  # Background color (not used when fill=False)
                    border_color=(255, 255, 255),
                    border_thickness=2,
                    text_color=(255, 255, 255),
                    font_scale=0.8,
                    bold=False,
                    shadow=False,
                    fill=False  # Border only, no fill
                )

            # Si se ha seleccionado "Virtuales", mostramos la selección del número de fichas
            if mostrar_num_piezas:
                # Posiciones para la selección del número de fichas (debajo de los botones)
                num_piezas_area = {
                    'x': x_start,
                    'y': initial_y + 2 * (button_height + spacing_y) + 50,  # Ajustar posición debajo de los botones
                    'width': button_width * 2 + spacing_x,
                    'height': button_height
                }

                # Áreas para los botones de aumentar y disminuir
                decrease_button = {
                    'x1': num_piezas_area['x'],
                    'y1': num_piezas_area['y'],
                    'x2': num_piezas_area['x'] + 80,
                    'y2': num_piezas_area['y'] + button_height
                }

                increase_button = {
                    'x1': num_piezas_area['x'] + num_piezas_area['width'] - 80,
                    'y1': num_piezas_area['y'],
                    'x2': num_piezas_area['x'] + num_piezas_area['width'],
                    'y2': num_piezas_area['y'] + button_height
                }

                # Área donde se muestra el número de piezas actual
                num_display_area = {
                    'x': decrease_button['x2'] + 10,
                    'y': num_piezas_area['y'],
                    'width': num_piezas_area['width'] - 2 * (80 + 10),
                    'height': button_height
                }

                # Dibujar título para la selección de número de fichas
                titulo_piezas = "Numero de Fichas"
                tamaño_texto, _ = cv2.getTextSize(titulo_piezas, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
                x_text_piezas = num_piezas_area['x'] + (num_piezas_area['width'] - tamaño_texto[0]) // 2
                y_text_piezas = num_piezas_area['y'] - 20  # Ajustar para que quede encima
                cv2.putText(screen, titulo_piezas, (x_text_piezas, y_text_piezas), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

                # Dibujar botón de disminuir usando componente
                draw_rectangular_button(
                    screen,
                    decrease_button['x1'], decrease_button['y1'],
                    decrease_button['x2'] - decrease_button['x1'],
                    decrease_button['y2'] - decrease_button['y1'],
                    "-",
                    bg_color=(0, 0, 0),  # Background color (not used when fill=False)
                    border_color=(255, 255, 255),
                    border_thickness=2,
                    text_color=(255, 255, 255),
                    font_scale=2.0,
                    bold=False,
                    shadow=False,
                    fill=False  # Border only, no fill
                )

                # Dibujar botón de aumentar usando componente
                draw_rectangular_button(
                    screen,
                    increase_button['x1'], increase_button['y1'],
                    increase_button['x2'] - increase_button['x1'],
                    increase_button['y2'] - increase_button['y1'],
                    "+",
                    bg_color=(0, 0, 0),  # Background color (not used when fill=False)
                    border_color=(255, 255, 255),
                    border_thickness=2,
                    text_color=(255, 255, 255),
                    font_scale=2.0,
                    bold=False,
                    shadow=False,
                    fill=False  # Border only, no fill
                )

                # Dibujar área de visualización del número de piezas
                cv2.rectangle(screen, (num_display_area['x'], num_display_area['y']),
                            (num_display_area['x'] + num_display_area['width'], num_display_area['y'] + num_display_area['height']), (255, 255, 255), 2)
                # Mostrar el número actual
                texto_num = str(num_piezas)
                tamaño_texto, _ = cv2.getTextSize(texto_num, fuente_encabezado, escala_fuente_encabezado, grosor_encabezado)
                text_x = num_display_area['x'] + (num_display_area['width'] - tamaño_texto[0]) // 2
                text_y = num_display_area['y'] + (num_display_area['height'] + tamaño_texto[1]) // 2
                cv2.putText(screen, texto_num, (text_x, text_y), fuente_encabezado, escala_fuente_encabezado, color_encabezado, grosor_encabezado)

                # Agregar las áreas de los botones de aumentar y disminuir al diccionario de áreas
                areas_opciones["decrease"] = decrease_button
                areas_opciones["increase"] = increase_button

        # Inicialmente dibujamos los elementos
        areas_opciones = {}  # Definir el diccionario aquí para que esté accesible dentro de draw_elements
        draw_elements(opciones_screen)

        # Mostrar la pantalla en la proyección
        cv2.namedWindow("Opciones de Clasificación", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Opciones de Clasificación", 1920, 0)
        cv2.setWindowProperty("Opciones de Clasificación", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Opciones de Clasificación", opciones_screen)

        # Crear áreas de selección para los botones
        for opcion, (x, y) in positions.items():
            areas_opciones[opcion] = {
                'x1': x,
                'y1': y,
                'x2': x + button_width,
                'y2': y + button_height
            }

        # Iniciar los streams de cámara
        rgb_stream = device.create_color_stream()
        depth_stream = device.create_depth_stream()
        rgb_stream.start()
        depth_stream.start()

        modo_seleccionado = None
        tipo_seleccionado = None
        seleccion_realizada = False

        # Variables para controlar un solo incremento/decremento por toque
        increase_button_pressed = False
        decrease_button_pressed = False

        # Definir la función 'detectar_opcion_seleccionada' dentro de 'seleccionar_opciones_clasificacion'
        def detectar_opcion_seleccionada(x_touch, y_touch, areas_opciones):
            """
            Dado un punto de toque (x_touch, y_touch), determina qué opción ha sido seleccionada.
            """
            for opcion, area in areas_opciones.items():
                if area['x1'] <= x_touch <= area['x2'] and area['y1'] <= y_touch <= area['y2']:
                    return opcion
            return None

        while True:
            frame = rgb_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            if frame is None or depth_frame is None:
                continue

            rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            bgr_data = cv2.flip(bgr_data, 1)
            bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

            # Crear la máscara que considera solo los valores entre dmin y dmax
            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255

            # Operaciones morfológicas
            kernel = np.ones((3, 3), np.uint8)
            touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)

            # Encontrar los contornos de los toques
            contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Reiniciar las banderas al inicio de cada iteración
            increase_button_still_pressed = False
            decrease_button_still_pressed = False

            # Procesar cada contorno
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Ajusta el umbral según sea necesario
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"]) + yw_min

                        # Mapeo de coordenadas de ventana a viewport
                        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))

                        # Detectar si se seleccionó una opción
                        opcion_seleccionada = detectar_opcion_seleccionada(x_touch, y_touch, areas_opciones)
                        if opcion_seleccionada:
                            print(f"Opción seleccionada: {opcion_seleccionada}")
                            # Determinar si es modo o tipo
                            if opcion_seleccionada in opciones_modo and modo_seleccionado is None:
                                modo_seleccionado = opcion_seleccionada
                                # Marcar visualmente la selección
                                cv2.rectangle(opciones_screen,
                                            (areas_opciones[modo_seleccionado]['x1'], areas_opciones[modo_seleccionado]['y1']),
                                            (areas_opciones[modo_seleccionado]['x2'], areas_opciones[modo_seleccionado]['y2']),
                                            (0, 255, 0), 3)
                                cv2.imshow("Opciones de Clasificación", opciones_screen)
                            elif opcion_seleccionada in opciones_tipo and tipo_seleccionado is None:
                                tipo_seleccionado = opcion_seleccionada
                                # Marcar visualmente la selección
                                cv2.rectangle(opciones_screen,
                                            (areas_opciones[tipo_seleccionado]['x1'], areas_opciones[tipo_seleccionado]['y1']),
                                            (areas_opciones[tipo_seleccionado]['x2'], areas_opciones[tipo_seleccionado]['y2']),
                                            (0, 255, 0), 3)
                                cv2.imshow("Opciones de Clasificación", opciones_screen)
                                if tipo_seleccionado == "Virtuales":
                                    mostrar_num_piezas = True
                                    # Recalcular las áreas de los botones de incremento y decremento
                                    areas_opciones.clear()
                                    for opcion, (x, y) in positions.items():
                                        areas_opciones[opcion] = {
                                            'x1': x,
                                            'y1': y,
                                            'x2': x + button_width,
                                            'y2': y + button_height
                                        }
                                    draw_elements(opciones_screen)
                                    cv2.imshow("Opciones de Clasificación", opciones_screen)
                                else:
                                    mostrar_num_piezas = False
                                    areas_opciones = {}
                                    for opcion, (x, y) in positions.items():
                                        areas_opciones[opcion] = {
                                            'x1': x,
                                            'y1': y,
                                            'x2': x + button_width,
                                            'y2': y + button_height
                                        }
                            elif mostrar_num_piezas:
                                if opcion_seleccionada == "increase":
                                    increase_button_still_pressed = True
                                    if not increase_button_pressed:
                                        if num_piezas < num_piezas_max:
                                            num_piezas += 1
                                            print(f"Número de fichas incrementado a {num_piezas}")
                                            # Redibujar los elementos para actualizar visualmente los cambios
                                            draw_elements(opciones_screen)
                                            cv2.imshow("Opciones de Clasificación", opciones_screen)
                                        increase_button_pressed = True  # Marcar que el botón está presionado
                                elif opcion_seleccionada == "decrease":
                                    decrease_button_still_pressed = True
                                    if not decrease_button_pressed:
                                        if num_piezas > num_piezas_min:
                                            num_piezas -= 1
                                            print(f"Número de fichas decrementado a {num_piezas}")
                                            # Redibujar los elementos para actualizar visualmente los cambios
                                            draw_elements(opciones_screen)
                                            cv2.imshow("Opciones de Clasificación", opciones_screen)
                                        decrease_button_pressed = True  # Marcar que el botón está presionado

                            if modo_seleccionado and tipo_seleccionado:
                                if tipo_seleccionado == "Físicas":
                                    seleccion_realizada = True
                                    break  # Salir del bucle de detección de toques
                                elif tipo_seleccionado == "Virtuales" and mostrar_num_piezas:
                                    # Esperar a que el usuario termine de ajustar el número de piezas y luego toque en algún lugar para continuar
                                    if not (opcion_seleccionada == "increase" or opcion_seleccionada == "decrease"):
                                        seleccion_realizada = True
                                        break

            # Actualizar el estado de los botones presionados
            increase_button_pressed = increase_button_still_pressed
            decrease_button_pressed = decrease_button_still_pressed

            # Mostrar la ventana de opciones
            cv2.imshow("Opciones de Clasificación", opciones_screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Detener los streams de la cámara y cerrar las ventanas
        rgb_stream.stop()
        depth_stream.stop()
        cv2.destroyWindow("Opciones de Clasificación")

        if modo_seleccionado and tipo_seleccionado:
            # Convertir las opciones seleccionadas a los parámetros esperados por juego_clasificacion
            modo_clasificacion = modo_seleccionado.lower()  # "figuras" o "colores"
            tipo_pieza = tipo_seleccionado.lower()  # "virtuales" o "físicas"
            if tipo_pieza == "virtuales":
                print(f"Iniciando juego de clasificación con modo: {modo_clasificacion}, tipo: {tipo_pieza}, número de fichas: {num_piezas}")
                tipo_pieza = False
                juego_clasificacion(device, modo_clasificacion, tipo_pieza, num_piezas)
            else:
                print(f"Iniciando juego de clasificación con modo: {modo_clasificacion}, tipo: {tipo_pieza}")
                tipo_pieza = True
                juego_clasificacion(device, modo_clasificacion, tipo_pieza, 0)
        else:
            print("No se seleccionaron todas las opciones necesarias. Volviendo al menú principal.")