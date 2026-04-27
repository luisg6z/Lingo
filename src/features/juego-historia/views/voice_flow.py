"""
Voice flow view: listen -> transcribe -> Ollama -> result screen.
"""
import os
import sys
import time
import math
import re
import json
import unicodedata
import threading
import cv2
import numpy as np
from collections import defaultdict

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

_views_dir = os.path.dirname(os.path.abspath(__file__))
_feature_dir = os.path.dirname(_views_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_feature_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.font_utils import get_ubuntu_font, calculate_word_positions
from src.features.confetti import ConfettiSystem
from src.components.rectangular_button import draw_rectangular_button, is_point_in_rectangular_button
from .helpers import (
    put_text_safe_historia,
    get_historia_story_voice,
    build_historia_voice_prep_cards,
    draw_historia_voice_prep_cards,
    historia_resumen_status_line_y,
)

def run_historia_voice_flow(window_name, view_width, view_height, scale_to_videobeam, draw_logo_func,
                             sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados,
                             device, coordenadas, dmax_map, dmin_map,
                             draw_close_card_final_fn, detectar_close_card_touch_final_fn,
                             close_card_detection_x_final, close_card_detection_y_final,
                             close_card_detection_w_final, close_card_detection_h_final,
                             VIDEOBEAM_WIDTH, VIDEOBEAM_HEIGHT, difficulty=3):
    """
    Run listen -> transcribe -> thinking -> Ollama -> result screen. Returns "MENU" when user presses X on result.
    """
    def _gradient_screen():
        screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            screen[y, :] = [b, g, r]
        return screen

    # Asegurar que la ventana esté siempre en pantalla completa en el videobeam
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        window_exists = visible >= 0
    except Exception:
        window_exists = False

    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        try:
            cv2.moveWindow(window_name, 1920, 0)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    sv = get_historia_story_voice()
    get_required_words = sv.get_required_words
    listen_and_transcribe = sv.listen_and_transcribe
    verify_story_ollama = sv.verify_story_ollama

    config_path = os.path.join(_feature_dir, "config", "palabras_imagenes.json")
    required_words = get_required_words(config_path, sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados)

    # Asegurar inicialización básica de pygame (para usar superficies como loader)
    try:
        import pygame
        if not pygame.get_init():
            pygame.init()
    except Exception:
        pygame = None

    # Helper: dibujar un reloj analógico como loader usando pygame y volcarlo en una imagen OpenCV.
    # La manecilla da vueltas completas y, cuando el modelo termina, esperamos a que vuelva a las 12
    # antes de cambiar de vista.
    def _draw_clock_loader(target_screen, center_x, center_y, radius, elapsed_ms):
        if pygame is None:
            return
        try:
            size = radius * 2 + 4
            clock_surface = pygame.Surface((size, size), pygame.SRCALPHA)

            # Progreso de giro periódico (vuelta completa cada 5s aprox.)
            rotation_ms = 5000.0
            progress = (elapsed_ms / rotation_ms) % 1.0

            # Fondo blanco sólido de la esfera
            clock_surface.fill((0, 0, 0, 0))
            pygame.draw.circle(clock_surface, (255, 255, 255, 255), (size // 2, size // 2), radius)
            pygame.draw.circle(clock_surface, (0, 0, 0, 255), (size // 2, size // 2), radius, 3)

            # Manecilla: empieza en las 12 (ángulo -pi/2 = arriba) y da vueltas horarias continuas.
            angle = -math.pi / 2.0 + 2.0 * math.pi * progress
            hand_len = int(radius * 0.75)
            cx, cy = size // 2, size // 2
            hx = cx + int(hand_len * math.cos(angle))
            hy = cy + int(hand_len * math.sin(angle))
            # Manecilla negra y un poco más gruesa
            pygame.draw.line(clock_surface, (0, 0, 0, 255), (cx, cy), (hx, hy), 8)

            # Cubrir cualquier punto de color en el centro del reloj
            # para que el punto central se vea negro (mismo color que la manecilla).
            pygame.draw.circle(clock_surface, (0, 0, 0, 255), (cx, cy), max(4, radius // 10))

            # Números de las horas (1 a 12)
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                font_num = pygame.font.SysFont(None, max(14, int(radius * 0.35)))
                for hour in range(1, 13):
                    # Ángulo para cada número (12 arriba)
                    ang = (hour / 12.0) * 2 * math.pi
                    num_radius = radius * 0.78
                    nx = cx + int(num_radius * math.sin(ang))
                    ny = cy - int(num_radius * math.cos(ang))
                    text_surface = font_num.render(str(hour), True, (0, 0, 0))
                    tw, th = text_surface.get_size()
                    clock_surface.blit(text_surface, (nx - tw // 2, ny - th // 2))
            except Exception:
                pass

            # Convertir surface de pygame a imagen BGRA de OpenCV
            clock_string = pygame.image.tostring(clock_surface, "RGBA")
            clock_np = np.frombuffer(clock_string, np.uint8)
            clock_img = clock_np.reshape((size, size, 4))
            clock_img = cv2.cvtColor(clock_img, cv2.COLOR_RGBA2BGRA)

            h, w = clock_img.shape[:2]
            x1 = max(0, center_x - w // 2)
            y1 = max(0, center_y - h // 2)
            x2 = min(target_screen.shape[1], x1 + w)
            y2 = min(target_screen.shape[0], y1 + h)
            if x2 <= x1 or y2 <= y1:
                return

            crop_w = x2 - x1
            crop_h = y2 - y1
            clock_crop = clock_img[0:crop_h, 0:crop_w]

            alpha = clock_crop[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            roi = target_screen[y1:y2, x1:x2].astype(np.float32)
            clock_rgb = clock_crop[:, :, :3].astype(np.float32)
            blended = alpha * clock_rgb + (1.0 - alpha) * roi
            target_screen[y1:y2, x1:x2] = blended.astype(np.uint8)
        except Exception:
            return

    # 1) Mismas cards que el resumen; sin botón Hablar en la escucha (desaparece al mostrar "Ahora habla")
    font = cv2.FONT_HERSHEY_DUPLEX
    fs_prep, th_prep = 1.5, 3
    prep_card_positions = build_historia_voice_prep_cards(
        sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados, view_width
    )

    prep_base_screen = _gradient_screen()
    draw_logo_func(prep_base_screen)
    if prep_card_positions:
        draw_historia_voice_prep_cards(prep_base_screen, prep_card_positions)
    try:
        draw_close_card_final_fn(prep_base_screen, False)
    except Exception:
        pass

    ty_msg_base = historia_resumen_status_line_y() if prep_card_positions else view_height // 2
    ty_msg_listening = ty_msg_base + 40

    def _draw_voice_mic_prep_screen(status_message, text_y):
        screen = prep_base_screen.copy()
        msg = status_message
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_ubuntu = get_ubuntu_font(font_scale=fs_prep, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), msg, font=font_ubuntu)
                text_w = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font_ubuntu.getbbox(msg) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                text_w = bbox[2] - bbox[0]
        except Exception:
            text_size, _ = cv2.getTextSize(msg, font, fs_prep, th_prep)
            text_w = text_size[0]
        tx = (view_width - text_w) // 2
        put_text_safe_historia(screen, msg, (tx + 2, text_y + 2), font, fs_prep, (0, 0, 0), th_prep + 1, bold=True)
        put_text_safe_historia(screen, msg, (tx, text_y), font, fs_prep, (255, 255, 255), th_prep, bold=True)
        cv2.imshow(window_name, scale_to_videobeam(screen))
        cv2.waitKey(1)

    # Vista "Ahora habla" en el hilo principal lo antes posible; luego se abre el mic (sin esperar al callback).
    _draw_voice_mic_prep_screen("Ahora habla", ty_msg_listening)
    # Sin tope duro de 10 s: la frase corta cuando hay ~2 s de silencio (silence_seconds en story_voice).
    text = listen_and_transcribe(
        timeout=10, phrase_time_limit=None, silence_seconds=2.0, on_listening_started=None
    )
    if not text:
        text = ""

    # 2) Transcription screen with typewriter animation
    trans_screen = _gradient_screen()
    draw_logo_func(trans_screen)
    tit = "Tu historia:"
    put_text_safe_historia(trans_screen, tit, (50, 120), font, 0.8, (255, 255, 255), 2)
    
    # Aumentar tamaño de la transcripción
    font_scale = 1.2  # Aumentado de 1.0 a 1.2 para texto más grande
    thickness = 3
    y_pos = 180
    max_width = view_width - 100
    
    # Usar PIL para medir texto correctamente y evitar cortes en medio de palabras
    try:
        from PIL import Image, ImageDraw
        from src.core.font_utils import get_ubuntu_font
        font_ubuntu = get_ubuntu_font(font_scale=font_scale, bold=False)
        use_pil = True
    except:
        use_pil = False
    
    # Dividir texto en palabras para animación tipo typewriter sin cortar palabras
    words = text.split() if text else []
    
    for i in range(1, len(words) + 1):
        trans_screen = _gradient_screen()
        draw_logo_func(trans_screen)
        put_text_safe_historia(trans_screen, tit, (50, 120), font, 0.8, (255, 255, 255), 2)
        
        # Construir texto hasta la palabra i-ésima (sin cortar palabras)
        current_words = words[:i]
        display_text = " ".join(current_words)
        
        # Word wrap usando PIL para medición precisa
        x_start = 50
        y_current = y_pos
        line_height = int(font_scale * 40)  # Espacio entre líneas basado en tamaño de fuente
        
        if use_pil:
            # Usar PIL para medir y dibujar con word wrapping preciso
            img_pil = Image.fromarray(cv2.cvtColor(trans_screen, cv2.COLOR_BGR2RGB))
            draw_pil = ImageDraw.Draw(img_pil)
            
            # Dividir en palabras y construir líneas
            words_to_draw = display_text.split()
            current_line = ""
            
            for word in words_to_draw:
                test_line = current_line + (" " if current_line else "") + word
                try:
                    bbox = draw_pil.textbbox((0, 0), test_line, font=font_ubuntu)
                    text_width = bbox[2] - bbox[0]
                except AttributeError:
                    bbox = font_ubuntu.getbbox(test_line) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                    text_width = bbox[2] - bbox[0]
                
                if text_width > max_width and current_line:
                    # Dibujar línea actual y pasar a la siguiente
                    put_text_safe_historia(trans_screen, current_line, (x_start, y_current), font, font_scale, (255, 255, 255), thickness)
                    y_current += line_height
                    current_line = word
                else:
                    current_line = test_line
            
            # Dibujar última línea si hay contenido
            if current_line:
                put_text_safe_historia(trans_screen, current_line, (x_start, y_current), font, font_scale, (255, 255, 255), thickness)
        else:
            # Fallback a método anterior si PIL no está disponible
            line_cur = ""
            for w in current_words:
                test = line_cur + (" " if line_cur else "") + w
                (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
                if tw > max_width and line_cur:
                    put_text_safe_historia(trans_screen, line_cur, (x_start, y_current), font, font_scale, (255, 255, 255), thickness)
                    y_current += line_height
                    line_cur = w
                else:
                    line_cur = test
            if line_cur:
                put_text_safe_historia(trans_screen, line_cur, (x_start, y_current), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imshow(window_name, scale_to_videobeam(trans_screen))
        # Hacer la animación más lenta: esperar más tiempo entre palabras
        # Esperar al menos 100ms por palabra, con un mínimo de 50ms
        wait_time = max(50, min(150, 800 // max(len(words), 1)))
        cv2.waitKey(wait_time)
    cv2.waitKey(800)

    # 3) Thinking indicator + Ollama
    result_holder = [None]
    done_holder = [False]
    def _ollama_thread():
        result_holder[0] = verify_story_ollama(
                text,
                subjects=sujetos_seleccionados,
                actions=acciones_seleccionadas,
                places=lugares_seleccionados,
                model="gemini-3-flash-preview:cloud",
                difficulty=difficulty,
            )
        done_holder[0] = True
    thr = threading.Thread(target=_ollama_thread, daemon=True)
    thr.start()
    start_thinking = time.time()
    finishing_clock = False  # Fase de "aterrizar" la manecilla en las 12
    while True:
        elapsed = int((time.time() - start_thinking) * 1000)
        dots = "." * ((elapsed // 500) % 4)
        think_screen = _gradient_screen()
        draw_logo_func(think_screen)
        msg_think = "Comprobando tu historia" + dots
        
        # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
        font_scale_think = 1.5  # Aumentado de 1.0 a 1.5
        thickness_think = 3  # Aumentado de 2 a 3
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_ubuntu = get_ubuntu_font(font_scale=font_scale_think, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(think_screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), msg_think, font=font_ubuntu)
                text_width_think = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font_ubuntu.getbbox(msg_think) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                text_width_think = bbox[2] - bbox[0]
        except:
            text_size_think, _ = cv2.getTextSize(msg_think, font, font_scale_think, thickness_think)
            text_width_think = text_size_think[0]
        
        # Centrar texto horizontalmente
        tx_think = (view_width - text_width_think) // 2
        ty_think = view_height // 2
        
        # Dibujar texto con sombra y más grande
        put_text_safe_historia(think_screen, msg_think, (tx_think + 2, ty_think + 2), font, font_scale_think, (0, 0, 0), thickness_think + 1, bold=True)
        put_text_safe_historia(think_screen, msg_think, (tx_think, ty_think), font, font_scale_think, (255, 255, 255), thickness_think, bold=True)

        # Dibujar reloj analógico de carga debajo del texto usando pygame
        clock_radius = 50
        clock_center_x = view_width // 2
        clock_center_y = ty_think + 120
        _draw_clock_loader(
            think_screen,
            clock_center_x,
            clock_center_y,
            clock_radius,
            elapsed,
        )

        cv2.imshow(window_name, scale_to_videobeam(think_screen))
        cv2.waitKey(1)  # Cambiar a 1ms para que no se quede pegado

        # Lógica de salida:
        # - Mientras el modelo no termina, el reloj sigue girando.
        # - Cuando termina, esperamos a que la manecilla vuelva a pasar por las 12
        #   (progreso cerca de 0) y recién ahí rompemos el bucle.
        if not done_holder[0]:
            continue

        # Modelo ya terminó: iniciar/usar fase de aterrizaje.
        # Recalculamos el progreso del reloj con el mismo período que en _draw_clock_loader.
        rotation_ms = 5000.0
        clock_progress = (elapsed / rotation_ms) % 1.0
        # Cerca de 12 cuando el progreso está muy cerca de 0 o de 1.
        if clock_progress <= 0.03 or clock_progress >= 0.97:
            break
    result = result_holder[0] or {"correct": False, "tips": ["Revisa tu oración."]}

    # Initialize confetti system if story is correct
    confetti_system = None
    if result.get("correct", False):
        confetti_system = ConfettiSystem(view_width, view_height, num_particles=120)
        confetti_system.start(multiple_bursts=True, num_burst_points=3)
        # Reproducir sonido de correcto (mismo que en otras vistas) cuando la historia es correcta
        try:
            import pygame
            # Asegurar que el mixer esté inicializado
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
            except:
                pygame.mixer.init()
            try:
                correct_sound = pygame.mixer.Sound("sounds/correct.mp3")
                correct_sound.set_volume(1.0)
                correct_sound.play()
            except Exception as e:
                print(f"⚠ Error al reproducir sonido de correcto en juego-historia: {e}")
        except Exception as e:
            print(f"⚠ Error al inicializar pygame para sonido de correcto en juego-historia: {e}")
    # Acelerar confetti en historias (en otros juegos se percibe más rápido por mayor FPS)
    confetti_speed_mult = 4  # Aumentado de 2 a 4 para hacer el confetti más rápido

    # 4) Pantalla de resultado: LingoBien/LingoMal + oración coloreada + consejos (sin botón X)
    # Cargar logo según si la historia es correcta o no
    is_correct = result.get("correct", False)
    if is_correct:
        lingo_path = os.path.join(_project_root, "images", "LingoBien.png")
        if not os.path.exists(lingo_path):
            lingo_path = "images/LingoBien.png"
    else:
        lingo_path = os.path.join(_project_root, "images", "LingoMal.png")
        if not os.path.exists(lingo_path):
            lingo_path = "images/LingoMal.png"
    lingo_img = cv2.imread(lingo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(lingo_path) else None

    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Vocabulario JSON para coloreo: una sola lectura (antes se hacía json.load en cada redibujado → lag enorme con confetti).
    def _dedupe_historia_vocab(items):
        seen = set()
        out = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    _historia_subjects_vocab = []
    _historia_actions_vocab = []
    _historia_places_vocab = []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _data_palabras_hist = json.load(f)
        for s in (sujetos_seleccionados or []):
            _historia_subjects_vocab.extend(_data_palabras_hist.get("sujetos", {}).get(s, []))
        for a in (acciones_seleccionadas or []):
            _historia_actions_vocab.extend(_data_palabras_hist.get("acciones", {}).get(a, []))
        for p in (lugares_seleccionados or []):
            _historia_places_vocab.extend(_data_palabras_hist.get("lugares", {}).get(p, []))
    except Exception:
        pass
    _historia_subjects_vocab = _dedupe_historia_vocab(_historia_subjects_vocab)
    _historia_actions_vocab = _dedupe_historia_vocab(_historia_actions_vocab)
    _historia_places_vocab = _dedupe_historia_vocab(_historia_places_vocab)

    def _draw_result_screen(screen):
        ratio_col = np.linspace(0.0, 1.0, view_height, dtype=np.float32)[:, np.newaxis]
        screen[:, :, 0] = (255 * (0.8 - 0.3 * ratio_col)).astype(np.uint8)
        screen[:, :, 1] = (200 * (0.5 + 0.3 * ratio_col)).astype(np.uint8)
        screen[:, :, 2] = (255 * (0.3 + 0.4 * ratio_col)).astype(np.uint8)
        draw_logo_func(screen)
        # LingoBien/LingoMal (más pequeño, LingoMal aún más pequeño)
        if lingo_img is not None:
            h_img, w_img = lingo_img.shape[:2]
            # LingoMal más pequeño que LingoBien
            if not is_correct:
                max_h = 100  # LingoMal más pequeño: 100 píxeles (reducido de 120)
            else:
                max_h = 150  # LingoBien: 150 píxeles
            scale = max_h / h_img
            w_new = int(w_img * scale)
            h_new = int(h_img * scale)
            resized = cv2.resize(lingo_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            lx = (view_width - w_new) // 2
            # Ajustar posición vertical según si es LingoBien o LingoMal
            if not is_correct:
                ly = 140  # LingoMal más abajo (de 120 a 140) para evitar choque con el logo
            else:
                ly = 120  # LingoBien en posición normal
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    screen[ly:ly + h_new, lx:lx + w_new, c] = (
                        alpha * resized[:, :, c] + (1 - alpha) * screen[ly:ly + h_new, lx:lx + w_new, c])
            else:
                screen[ly:ly + h_new, lx:lx + w_new] = resized[:, :, :3]
        
        # Sentence with highlights (word-by-word with line wrap) and styling for parts
        display_text = text if text else "No se pudo entender. Intenta de nuevo."
        y_pos = 350  # Bajado de 320 a 350 para dar más espacio y bajar el texto
        # Aumentar un poco más el tamaño de la transcripción coloreada
        font_scale = 1.2
        thickness = 3
        max_width = view_width - 80
        
        # Coloreo alineado con la consola (parts del modelo): primero fragmentos de Ollama en orden
        # sujeto → acción → lugar, una aparición por fragmento en índices libres; luego vocabulario JSON
        # solo en palabras no reclamadas (evita que acción "cocina" y lugar "la cocina" queden del mismo color).
        subjects_vocab = _historia_subjects_vocab
        actions_vocab = _historia_actions_vocab
        places_vocab = _historia_places_vocab
        parts = result.get("parts", {}) or {}
        o_subjects = parts.get("subjects", []) or []
        o_actions = parts.get("actions", []) or []
        o_places = parts.get("places", []) or []

        def _as_fragment_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return [str(x).strip() for x in val if x is not None and str(x).strip()]
            s = str(val).strip()
            return [s] if s else []

        # Normalizar para coincidir frases aunque cambien tildes (estación / estacion)
        def normalize_word(word):
            w = re.sub(r"[^\wáéíóúñü]", "", (word or "").lower())
            w = unicodedata.normalize("NFD", w)
            return "".join(c for c in w if unicodedata.category(c) != "Mn")

        def _clean_token_edges(word):
            s = str(word or "").strip()
            s = re.sub(r'^[^\wáéíóúñüÁÉÍÓÚÑÜ]+', "", s)
            s = re.sub(r'[^\wáéíóúñüÁÉÍÓÚÑÜ]+$', "", s)
            return s

        def _model_fragment_word_match(token, frag_word):
            """Coincide token del STT con palabra del fragmento del modelo (mayúsculas / forma exacta)."""
            ct = _clean_token_edges(token)
            fw = (frag_word or "").strip()
            if not fw:
                return False
            if ct == fw:
                return True
            if ct.lower() == fw.lower():
                return True
            return normalize_word(ct) == normalize_word(fw)

        def _span_dict_from_range(wp_data, si, ei, phrase, kind):
            start_wp = wp_data[si]
            end_wp = wp_data[ei]
            return {
                "start_idx": si,
                "end_idx": ei,
                "x_start": start_wp["x_start"],
                "x_end": end_wp["x_end"],
                "y": start_wp["y"],
                "phrase": phrase,
                "kind": kind,
            }

        def _try_claim_phrase_model(phrase, kind, wp_data, claimed):
            phrase_words = phrase.split()
            n = len(phrase_words)
            if n == 0:
                return None
            for i in range(len(wp_data) - n + 1):
                idxs = range(i, i + n)
                if any(j in claimed for j in idxs):
                    continue
                if all(_model_fragment_word_match(wp_data[i + j]["word"], phrase_words[j]) for j in range(n)):
                    for j in idxs:
                        claimed.add(j)
                    return _span_dict_from_range(wp_data, i, i + n - 1, phrase, kind)
            return None

        def collect_model_part_spans(o_subj, o_act, o_plc, wp_data, claimed):
            out = []
            for phrase in _as_fragment_list(o_subj):
                sp = _try_claim_phrase_model(phrase, "subject", wp_data, claimed)
                if sp:
                    out.append(sp)
            for phrase in _as_fragment_list(o_act):
                sp = _try_claim_phrase_model(phrase, "action", wp_data, claimed)
                if sp:
                    out.append(sp)
            for phrase in _as_fragment_list(o_plc):
                sp = _try_claim_phrase_model(phrase, "place", wp_data, claimed)
                if sp:
                    out.append(sp)
            return out

        def find_vocab_spans_leftmost(phrase_list, wp_data, claimed, kind):
            """Una coincidencia por entrada de lista, la más a la izquierda sin solapar índices ya usados."""
            matches = []
            for phrase in phrase_list:
                if not phrase or not phrase.strip():
                    continue
                phrase_words = phrase.split()
                n = len(phrase_words)
                if n == 1:
                    phrase_norm = normalize_word(phrase_words[0])
                    found_idx = None
                    for idx, wpd in enumerate(wp_data):
                        if idx in claimed:
                            continue
                        word_norm = normalize_word(wpd["word"])
                        if (
                            word_norm == phrase_norm
                            or word_norm.startswith(phrase_norm)
                            or phrase_norm.startswith(word_norm)
                        ):
                            found_idx = idx
                            break
                    if found_idx is not None:
                        claimed.add(found_idx)
                        matches.append(_span_dict_from_range(wp_data, found_idx, found_idx, phrase, kind))
                else:
                    phrase_norms = [normalize_word(w) for w in phrase_words]
                    for i in range(len(wp_data) - n + 1):
                        idxs = list(range(i, i + n))
                        if any(j in claimed for j in idxs):
                            continue
                        ok = True
                        for j, pn in enumerate(phrase_norms):
                            word_norm = normalize_word(wp_data[i + j]["word"])
                            if (
                                word_norm != pn
                                and not (word_norm.startswith(pn) or pn.startswith(word_norm))
                            ):
                                ok = False
                                break
                        if ok:
                            for j in idxs:
                                claimed.add(j)
                            matches.append(_span_dict_from_range(wp_data, i, i + n - 1, phrase, kind))
                            break
            return matches

        # Colores por categoría (BGR)
        subject_color = (0, 220, 255)      # Sujeto: amarillo
        place_color = (255, 232, 57)       # Lugar: cian
        action_color = (85, 227, 57)      # Acción: verde
        
        # Calculate word positions using utility function
        word_positions_data = calculate_word_positions(display_text, 40, y_pos, font_scale, max_width, bold=False)

        claimed = set()
        accepted_spans = collect_model_part_spans(
            o_subjects, o_actions, o_places, word_positions_data, claimed
        )
        # Vocabulario JSON: sujeto → acción → lugar (misma prioridad que al aplicar parts) si la misma forma encaja en dos listas
        accepted_spans.extend(
            find_vocab_spans_leftmost(subjects_vocab, word_positions_data, claimed, "subject")
        )
        accepted_spans.extend(
            find_vocab_spans_leftmost(actions_vocab, word_positions_data, claimed, "action")
        )
        accepted_spans.extend(
            find_vocab_spans_leftmost(places_vocab, word_positions_data, claimed, "place")
        )

        subject_word_idxs = set()
        place_word_idxs = set()
        action_word_idxs = set()
        for sp in accepted_spans:
            try:
                si = int(sp["start_idx"])
                ei = int(sp["end_idx"])
            except Exception:
                continue
            if ei < si:
                si, ei = ei, si
            idxs = range(si, ei + 1)
            kind = sp.get("kind")
            if kind == "subject":
                subject_word_idxs.update(idxs)
            elif kind == "place":
                place_word_idxs.update(idxs)
            elif kind == "action":
                action_word_idxs.update(idxs)

        # Refuerzo: trío frecuente "estación/estacion" + "de" + "policía/policia" => todo lugar
        # (si la frase larga no matcheó palabra a palabra, evita colorear solo "policía" como sujeto).
        n_words = len(word_positions_data)
        for i in range(n_words - 2):
            w0 = normalize_word(word_positions_data[i]["word"])
            w1 = normalize_word(word_positions_data[i + 1]["word"])
            w2 = normalize_word(word_positions_data[i + 2]["word"])
            if w0 == "estacion" and w1 == "de" and w2 == "policia":
                for j in (i, i + 1, i + 2):
                    subject_word_idxs.discard(j)
                    action_word_idxs.discard(j)
                    place_word_idxs.add(j)

        underline_info = [
            {
                "start_idx": int(sp["start_idx"]),
                "end_idx": int(sp["end_idx"]),
                "is_subject": sp.get("kind") == "subject",
                "is_action": sp.get("kind") == "action",
                "is_place": sp.get("kind") == "place",
                "phrase": sp.get("phrase", ""),
            }
            for sp in accepted_spans
        ]

        # Fuente alternativa para "Acción" (otra fuente). Intentar Italic del sistema; fallback a Ubuntu bold.
        def _get_action_font(font_scale_local):
            try:
                from PIL import ImageFont
                from src.core.font_utils import get_ubuntu_font_path, get_ubuntu_font
                base = get_ubuntu_font_path()
                candidates = []
                if base:
                    font_dir = os.path.dirname(base)
                    candidates.extend([
                        os.path.join(font_dir, "Ubuntu-Italic.ttf"),
                        os.path.join(font_dir, "Ubuntu-BoldItalic.ttf"),
                        os.path.join(font_dir, "Ubuntu-MediumItalic.ttf"),
                    ])
                if sys.platform == "win32":
                    windir = os.environ.get("WINDIR", "C:/Windows")
                    candidates.extend([
                        os.path.join(windir, "Fonts", "ariali.ttf"),
                        os.path.join(windir, "Fonts", "calibrii.ttf"),
                        os.path.join(windir, "Fonts", "timesi.ttf"),
                    ])
                font_size = max(int(22 * font_scale_local), 16)
                for p in candidates:
                    if p and os.path.exists(p):
                        return ImageFont.truetype(p, font_size)
                return get_ubuntu_font(font_scale=font_scale_local, bold=True)
            except Exception:
                return None

        action_font = _get_action_font(font_scale)

        shadow_bgr = (35, 35, 35)
        shadow_offsets = ((2, 2), (1, 2), (2, 1))

        def _put_text_with_font_override(img, text_to_draw, position, pil_font, color_bgr, with_shadow=False):
            """Dibuja texto con fuente PIL (acción); opcional sombra para legibilidad."""
            try:
                from PIL import Image, ImageDraw
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                x, y = int(position[0]), int(position[1])
                if with_shadow:
                    sh_rgb = (shadow_bgr[2], shadow_bgr[1], shadow_bgr[0])
                    for dx, dy in shadow_offsets:
                        draw.text((x + dx, y + dy), text_to_draw, fill=sh_rgb, font=pil_font)
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                draw.text((x, y), text_to_draw, fill=color_rgb, font=pil_font)
                img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                img[:] = img_result[:]
            except Exception:
                if with_shadow:
                    for dx, dy in shadow_offsets:
                        put_text_safe_historia(
                            img, text_to_draw, (position[0] + dx, position[1] + dy),
                            font, font_scale, shadow_bgr, thickness, bold=True,
                        )
                put_text_safe_historia(img, text_to_draw, position, font, font_scale, color_bgr, thickness, bold=True)

        def _put_word_shadow_then_color(img, text, pos, fg_bgr, bold=True):
            """Sombreado + texto con Ubuntu (sujeto / lugar)."""
            x, y = int(pos[0]), int(pos[1])
            for dx, dy in shadow_offsets:
                put_text_safe_historia(img, text, (x + dx, y + dy), font, font_scale, shadow_bgr, thickness, bold=bold)
            put_text_safe_historia(img, text, (x, y), font, font_scale, fg_bgr, thickness, bold=bold)

        # Dibujar texto palabra por palabra: sujeto / acción / lugar con sombra; resto sin sombra
        for idx, wp_data in enumerate(word_positions_data):
            word = wp_data["word"]

            is_action = idx in action_word_idxs
            is_subject = idx in subject_word_idxs
            is_place = idx in place_word_idxs

            x_start = wp_data["x_start"]
            y_base = wp_data["y"]

            if is_action or is_subject or is_place:
                if is_action:
                    fg_color = action_color
                elif is_subject:
                    fg_color = subject_color
                else:
                    fg_color = place_color

                if is_action:
                    if action_font is not None:
                        _put_text_with_font_override(
                            screen, word, (x_start, y_base), action_font, fg_color, with_shadow=True
                        )
                    else:
                        _put_word_shadow_then_color(screen, word, (x_start, y_base), fg_color, bold=True)
                elif is_subject:
                    _put_word_shadow_then_color(screen, word, (x_start, y_base), fg_color, bold=True)
                else:
                    _put_word_shadow_then_color(screen, word, (x_start, y_base), fg_color, bold=True)
            else:
                put_text_safe_historia(screen, word, (x_start, y_base), font, font_scale, (255, 255, 255), thickness)

        # --- Leyenda (sin subrayados): "abc" con color/fuente ---
        legend_x = view_width - 220  # Movido más a la derecha (de 290 a 220)
        legend_y = view_height - 130
        legend_font_scale = 0.55
        legend_thickness = 2
        legend_line_height = 28

        legend_bg_height = 3 * legend_line_height + 20
        legend_bg_width = 200  # Ancho del fondo de la leyenda (reducido de 270)
        legend_bg = np.zeros((legend_bg_height, legend_bg_width, 3), dtype=np.uint8)
        legend_bg[:] = (40, 40, 40)
        screen[legend_y - 10:legend_y - 10 + legend_bg_height, legend_x - 10:legend_x - 10 + legend_bg_width] = \
            cv2.addWeighted(screen[legend_y - 10:legend_y - 10 + legend_bg_height, legend_x - 10:legend_x - 10 + legend_bg_width], 0.7, legend_bg, 0.3, 0)

        sample = "abc"
        label_color = (255, 255, 255)
        sample_x = legend_x
        label_x = legend_x + 60
        y0 = legend_y

        # Leyenda: muestras "abc" con el mismo sombreado que en la oración
        for dx, dy in shadow_offsets:
            put_text_safe_historia(
                screen, sample, (sample_x + dx, y0 + dy), font, legend_font_scale, shadow_bgr, legend_thickness, bold=True
            )
        put_text_safe_historia(screen, sample, (sample_x, y0), font, legend_font_scale, subject_color, legend_thickness, bold=True)
        put_text_safe_historia(screen, "Sujeto", (label_x, y0), font, legend_font_scale, label_color, legend_thickness)
        y0 += legend_line_height

        for dx, dy in shadow_offsets:
            put_text_safe_historia(
                screen, sample, (sample_x + dx, y0 + dy), font, legend_font_scale, shadow_bgr, legend_thickness, bold=True
            )
        put_text_safe_historia(screen, sample, (sample_x, y0), font, legend_font_scale, place_color, legend_thickness, bold=True)
        put_text_safe_historia(screen, "Lugar", (label_x, y0), font, legend_font_scale, label_color, legend_thickness)
        y0 += legend_line_height

        if action_font is not None:
            _put_text_with_font_override(screen, sample, (sample_x, y0), action_font, action_color, with_shadow=True)
        else:
            for dx, dy in shadow_offsets:
                put_text_safe_historia(
                    screen, sample, (sample_x + dx, y0 + dy), font, legend_font_scale, shadow_bgr, legend_thickness, bold=True
                )
            put_text_safe_historia(screen, sample, (sample_x, y0), font, legend_font_scale, action_color, legend_thickness, bold=True)
        put_text_safe_historia(screen, "Acción", (label_x, y0), font, legend_font_scale, label_color, legend_thickness)

        # Tips - mostrar como lista numerada con fuente más grande SOLO cuando la
        # historia NO está completamente bien (is_correct == False) y haya tips.
        tips = result.get("tips") or []
        if not is_correct and tips:
            # Calcular posición inicial para tips (bajado más y ajustado para no chocar con leyenda)
            y_tips = y_pos + 160  # Aumentado de 140 a 160 para bajar más los consejos y el título
            tips_font_scale = 0.75  # Aumentado de 0.55 a 0.75
            tips_thickness = 3  # Aumentado de 2 a 3
            # Reducir ancho máximo para evitar que choque con la leyenda (que está más a la derecha ahora)
            tips_max_width = view_width - 250  # Reducido de 100 a 250 para dejar espacio para la leyenda

            # Dibujar el título "Consejos:" con fuente más grande
            put_text_safe_historia(screen, "Consejos:", (40, y_tips), font, 0.8, (200, 200, 255), 3, bold=True)

            # Dibujar tips como lista numerada
            tips_x = 60  # Indentación para la lista
            tips_y = y_tips + 35
            tips_line_height = 35  # Espacio entre líneas (aumentado)

            for idx, tip in enumerate(tips, 1):
                # Crear texto con número de lista: "1. [tip text]"
                tip_text = f"{idx}. {tip}"

                # Dividir el tip en palabras para word wrapping
                tip_words = tip_text.split()
                current_line = ""

                for word in tip_words:
                    test_line = current_line + (" " if current_line else "") + word
                    # Calcular ancho del texto de prueba
                    try:
                        from PIL import Image, ImageDraw
                        from src.core.font_utils import get_ubuntu_font
                        font_ubuntu = get_ubuntu_font(font_scale=tips_font_scale, bold=False)
                        img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        try:
                            bbox = draw.textbbox((0, 0), test_line, font=font_ubuntu)
                            test_width = bbox[2] - bbox[0]
                        except AttributeError:
                            bbox = font_ubuntu.getbbox(test_line) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                            test_width = bbox[2] - bbox[0]
                    except:
                        test_size, _ = cv2.getTextSize(test_line, font, tips_font_scale, tips_thickness)
                        test_width = test_size[0]

                    # Si el texto excede el ancho máximo, dibujar la línea actual y empezar una nueva
                    if test_width > tips_max_width and current_line:
                        put_text_safe_historia(screen, current_line, (tips_x, tips_y), font, tips_font_scale, (255, 255, 255), tips_thickness)
                        tips_y += tips_line_height
                        current_line = word
                    else:
                        current_line = test_line

                # Dibujar la última línea del tip si hay contenido
                if current_line:
                    put_text_safe_historia(screen, current_line, (tips_x, tips_y), font, tips_font_scale, (255, 255, 255), tips_thickness)
                    tips_y += tips_line_height
        # Draw confetti if story is correct (will be drawn after this function returns)

    # Mostrar la vista de resultados con tips primero
    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()
    # Pantalla de resultado estática: un solo _draw_result_screen (muy pesado); en animación solo se copia + confetti.
    result_screen_static = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    _draw_result_screen(result_screen_static)
    cv2.imshow(window_name, scale_to_videobeam(result_screen_static))
    cv2.waitKey(1)  # Actualizar la ventana inmediatamente para que se muestre rápido
    result_display_buf = np.empty_like(result_screen_static)
    
    # Leer los tips por voz DESPUÉS de mostrar la pantalla de resultados
    tips = result.get("tips") or []
    tips_thread = None
    if tips:
        tips_text = ". ".join(tips)  # Unir los tips con puntos
        def _speak_tips():
            if pyttsx3 is None:
                return
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                for v in engine.getProperty("voices"):
                    if "spanish" in v.name.lower() or "español" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(tips_text)
                engine.runAndWait()
            except Exception as e:
                print(f"⚠ Error al reproducir TTS de tips: {e}")
        
        # Reproducir tips en un hilo separado (no daemon para poder esperar a que termine)
        tips_thread = threading.Thread(target=_speak_tips, daemon=False)
        tips_thread.start()
    
    # Esperar a que el TTS termine y luego tiempo adicional antes de mostrar el overlay
    # Esperar a que el hilo de TTS termine (si existe) y medir el tiempo que tomó
    tts_start_time = time.time()
    if tips_thread is not None:
        tips_thread.join()  # Esperar a que termine el TTS
    tts_duration = time.time() - tts_start_time
    
    # Calcular tiempo mínimo total que debe esperarse (igual para ambos casos)
    # Cuando es "inténtalo de nuevo", normalmente hay tips más largos, así que el TTS toma más tiempo
    # Cuando es "Muy bien", puede haber tips más cortos o no haber tips
    # Establecer un tiempo mínimo total de espera para ambos casos
    min_total_wait_time = 4.0  # Tiempo mínimo total: 4 segundos (TTS + espera adicional)
    tips_display_duration = max(2.0, min_total_wait_time - tts_duration)  # Asegurar tiempo mínimo total
    
    # Esperar tiempo adicional después de que termine el TTS
    tips_display_time = 0
    
    while tips_display_time < tips_display_duration:
        tips_display_time += 0.1
        time.sleep(0.1)

        if confetti_system is not None:
            confetti_system.update(steps=confetti_speed_mult)
            np.copyto(result_display_buf, result_screen_static)
            confetti_system.draw(result_display_buf)
            cv2.imshow(window_name, scale_to_videobeam(result_display_buf))
        else:
            cv2.imshow(window_name, scale_to_videobeam(result_screen_static))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rgb_stream.stop()
            depth_stream.stop()
            return None
    
    # Ahora mostrar la nueva vista con overlay claro sobre la vista de resultados
    is_correct = result.get("correct", False)
    message_text = "¡Muy Bien!" if is_correct else "Vamos inténtalo de nuevo"
    
    def _draw_final_screen(base_screen, elevated_salir=False, elevated_reintentar=False):
        # Usar la vista de resultados como base (ya tiene la historia y consejos)
        # Trabajar directamente sobre base_screen para que los cambios se reflejen
        
        # Agregar overlay oscuro/transparente sobre la vista de resultados
        # Crear una capa oscura (negro semi-transparente) para que se vea la vista de fondo
        dark_overlay = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        alpha = 0.4  # 40% de opacidad oscura para que se vea mejor la vista de fondo
        cv2.addWeighted(base_screen, 1 - alpha, dark_overlay, alpha, 0, base_screen)
        
        # Dibujar imagen LingoBien centrada
        if lingo_img is not None:
            h_img, w_img = lingo_img.shape[:2]
            max_h = 220
            scale = max_h / h_img
            w_new = int(w_img * scale)
            h_new = int(h_img * scale)
            resized = cv2.resize(lingo_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            lx = (view_width - w_new) // 2
            ly = 140
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                alpha_img = resized[:, :, 3] / 255.0
                for c in range(3):
                    base_screen[ly:ly + h_new, lx:lx + w_new, c] = (
                        alpha_img * resized[:, :, c] + (1 - alpha_img) * base_screen[ly:ly + h_new, lx:lx + w_new, c])
            else:
                base_screen[ly:ly + h_new, lx:lx + w_new] = resized[:, :, :3]
            
            # Dibujar texto "Muy bien" o "Vamos inténtalo de nuevo" más grande y centrado abajo de la imagen
            message_y = ly + h_new + 40  # 40 píxeles abajo de la imagen
            message_font_scale = 1.5  # Más grande
            message_thickness = 3
            message_color = (0, 255, 0) if is_correct else (0, 200, 255)  # Verde si correcto, naranja si no
            
            # Usar PIL para obtener tamaño preciso del texto con Ubuntu font para centrado correcto
            try:
                from PIL import Image, ImageDraw
                from src.core.font_utils import get_ubuntu_font
                font_ubuntu = get_ubuntu_font(font_scale=message_font_scale, bold=True)
                img_pil = Image.fromarray(cv2.cvtColor(base_screen, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                try:
                    bbox = draw.textbbox((0, 0), message_text, font=font_ubuntu)
                    message_width = bbox[2] - bbox[0]
                except AttributeError:
                    bbox = font_ubuntu.getbbox(message_text) if hasattr(font_ubuntu, "getbbox") else (0, 0, 0, 0)
                    message_width = bbox[2] - bbox[0]
            except:
                message_size, _ = cv2.getTextSize(message_text, font, message_font_scale, message_thickness)
                message_width = message_size[0]
            
            # Centrar texto horizontalmente
            message_x = (view_width - message_width) // 2
            
            # Dibujar texto con sombra
            put_text_safe_historia(base_screen, message_text, (message_x + 2, message_y + 2), 
                                   font, message_font_scale, (0, 0, 0), message_thickness + 1, bold=True)
            put_text_safe_historia(base_screen, message_text, (message_x, message_y), 
                                   font, message_font_scale, message_color, message_thickness, bold=True)
        
        # Dibujar botones rectangulares: "Salir" y "Volver a intentarlo"
        button_width_salir = 200
        button_width_reintentar = 280  # Más ancho para "Volver a intentarlo"
        button_height = 60
        button_spacing = 50
        button_y = view_height - 90 - button_height  # Misma posición que botones "Siguiente" y "Jugar"
        button_center_x = view_width // 2
        
        # Botón "Salir" (izquierda) - rojo
        button_salir_x = button_center_x - button_width_salir - button_spacing // 2
        button_salir_bounds = draw_rectangular_button(
            base_screen,
            button_salir_x, button_y, button_width_salir, button_height,
            "Salir",
            bg_color=(0, 0, 200),  # Rojo
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.0,  # Tamaño reducido
            bold=True,
            shadow=True
        )
        
        # Botón "Volver a intentarlo" o "Volver a jugar" (derecha) - verde
        button_reintentar_x = button_center_x + button_spacing // 2
        # Cambiar el texto del botón según si la historia es correcta o no
        button_text_reintentar = "Volver a jugar" if is_correct else "Volver a intentarlo"
        button_reintentar_bounds = draw_rectangular_button(
            base_screen,
            button_reintentar_x, button_y, button_width_reintentar, button_height,
            button_text_reintentar,
            bg_color=(100, 255, 100),  # Verde
            border_color=(255, 255, 255),
            border_thickness=3,
            text_color=(255, 255, 255),
            font_scale=1.0,  # Tamaño reducido
            bold=True,
            shadow=True
        )
        
        return button_salir_bounds, button_reintentar_bounds
    
    # Mostrar la nueva vista final usando la vista de resultados como base (copia rápida de la estática rasterizada).
    result_screen_final = np.empty_like(result_screen_static)
    np.copyto(result_screen_final, result_screen_static)

    # Ahora agregar el overlay oscuro con imagen Lingo, texto y botones
    button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_screen_final, elevated_salir=False, elevated_reintentar=False)
    
    # Draw confetti in the final overlay if active
    if confetti_system is not None:
        confetti_system.draw(result_screen_final)
    
    cv2.imshow(window_name, scale_to_videobeam(result_screen_final))
    
    salir_elevated = False
    reintentar_elevated = False
    frame_count_final = 0
    touch_history_final = defaultdict(list)
    
    while True:
        frame_count_final += 1
        frame = rgb_stream.read_frame()
        depth_frame = depth_stream.read_frame()
        if frame is None or depth_frame is None:
            np.copyto(result_display_buf, result_screen_static)
            if confetti_system is not None:
                confetti_system.update(steps=confetti_speed_mult)
            button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_display_buf, elevated_salir=salir_elevated, elevated_reintentar=reintentar_elevated)
            if confetti_system is not None:
                confetti_system.draw(result_display_buf)
            cv2.imshow(window_name, scale_to_videobeam(result_display_buf))
            cv2.waitKey(1)
            continue
        
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask = cv2.medianBlur(touch_mask, 3)
        kernel = np.ones((2, 2), np.uint8)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_OPEN, kernel)
        touch_mask = cv2.morphologyEx(touch_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(touch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Procesar toques
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 <= area <= 50000:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x_touch = int(xv_min + cx * (xv_max - xv_min) / (xw_max - xw_min))
                    y_touch = int(yv_min + cy * (yv_max - yv_min) / (yw_max - yw_min))
                    
                    # Verificar si se tocó el botón "Salir"
                    if button_salir_bounds and is_point_in_rectangular_button(x_touch, y_touch, button_salir_bounds):
                        print("Botón 'Salir' presionado - Volviendo al menú principal")
                        rgb_stream.stop()
                        depth_stream.stop()
                        return "MENU"
                    
                    # Verificar si se tocó el botón "Volver a intentarlo" o "Volver a jugar"
                    if button_reintentar_bounds and is_point_in_rectangular_button(x_touch, y_touch, button_reintentar_bounds):
                        button_text = "Volver a jugar" if is_correct else "Volver a intentarlo"
                        print(f"Botón '{button_text}' presionado")
                        rgb_stream.stop()
                        depth_stream.stop()
                        # Si la historia es correcta, volver a la primera selección del juego
                        # Si no es correcta, volver a la vista final (donde está el botón Hablar)
                        if is_correct:
                            return "RESTART_FULL"  # Volver al inicio del juego
                        else:
                            return "RESTART"  # Volver a la vista final
        
        np.copyto(result_display_buf, result_screen_static)
        if confetti_system is not None:
            confetti_system.update(steps=confetti_speed_mult)
        button_salir_bounds, button_reintentar_bounds = _draw_final_screen(result_display_buf, elevated_salir=salir_elevated, elevated_reintentar=reintentar_elevated)
        if confetti_system is not None:
            confetti_system.draw(result_display_buf)
        cv2.imshow(window_name, scale_to_videobeam(result_display_buf))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rgb_stream.stop()
            depth_stream.stop()
            return None
    return "MENU"
