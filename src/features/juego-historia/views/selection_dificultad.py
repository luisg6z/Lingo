"""
Difficulty selection view for juego-historia.
Shows 3 difficulty cards: Fácil, Intermedio, Difícil.
"""
import os
import sys
import time
import cv2
import numpy as np

_views_dir = os.path.dirname(os.path.abspath(__file__))
_feature_dir = os.path.dirname(_views_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_feature_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.font_utils import put_text_ubuntu
from src.core.ui_utils import scale_to_videobeam

# Resolución del videobeam (segunda pantalla)
VIDEOBEAM_WIDTH = 1920
VIDEOBEAM_HEIGHT = 1080


def mostrar_seleccion_dificultad(device, coordenadas, dmax_map, dmin_map, draw_logo_func, existing_window_name=None):
    """
    Muestra la vista de selección de dificultad con 3 cards.

    Args:
        device: Dispositivo OpenNI2
        coordenadas: Diccionario con las coordenadas de calibración
        dmax_map: Mapa de profundidad máximo
        dmin_map: Mapa de profundidad mínimo
        draw_logo_func: Función para dibujar el logo en la pantalla
        existing_window_name: Nombre de ventana existente para reutilizar (opcional)

    Returns:
        int o None: Nivel de dificultad seleccionado (1, 2 o 3) o None si se canceló
    """
    # Extraer coordenadas
    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]

    # Tamaño de la pantalla del videobeam (viewport)
    view_width = 1280
    view_height = 800

    # ── helpers de dibujo ──────────────────────────────────────────────

    def _gradient_screen():
        screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        for y in range(view_height):
            ratio = y / view_height
            r = int(255 * (0.3 + 0.4 * ratio))
            g = int(200 * (0.5 + 0.3 * ratio))
            b = int(255 * (0.8 - 0.3 * ratio))
            screen[y, :] = [b, g, r]
        return screen

    def draw_close_card(screen, elevated=False):
        """Dibuja la card circular roja con X para cerrar."""
        base_card_radius = 50
        card_margin_x = 180
        card_margin_y = 80

        elevation_offset = 0
        scale_factor = 1.0
        shadow_offset_base = 5

        if elevated:
            elevation_offset = -15
            scale_factor = 1.08
            shadow_offset_base = 10

        card_radius = int(base_card_radius * scale_factor)
        card_center = (
            view_width - card_margin_x - int(base_card_radius * scale_factor),
            card_margin_y + int(base_card_radius * scale_factor) + elevation_offset,
        )

        # Sombra
        shadow_offset = int(shadow_offset_base * scale_factor)
        for i in range(3, 0, -1):
            shadow_alpha = i / 3.0 * 0.3
            shadow_color = tuple(int(c * shadow_alpha) for c in (100, 0, 0))
            offset = shadow_offset + (3 - i)
            cv2.circle(screen, (card_center[0] + offset, card_center[1] + offset), card_radius, shadow_color, -1)

        color_intensity = 1.15 if elevated else 1.0
        base_red_light = min(255, int(100 * color_intensity))
        base_red_medium = min(255, int(50 * color_intensity))
        base_red_dark = min(255, int(30 * color_intensity))

        cv2.circle(screen, card_center, card_radius, (base_red_light, base_red_light, 255), -1)
        cv2.circle(screen, card_center, int(card_radius * 0.85), (base_red_medium, base_red_medium, 255), -1)
        cv2.circle(screen, card_center, int(card_radius * 0.7), (base_red_dark, base_red_dark, 255), -1)

        cv2.circle(screen, card_center, card_radius, (255, 255, 255), 4)
        cv2.circle(screen, card_center, card_radius - 2, (200, 200, 200), 2)

        x_size = int(card_radius * 0.5)
        thickness = 5
        cv2.line(screen, (card_center[0] - x_size + 2, card_center[1] - x_size + 2),
                 (card_center[0] + x_size + 2, card_center[1] + x_size + 2), (150, 150, 150), thickness)
        cv2.line(screen, (card_center[0] - x_size + 2, card_center[1] + x_size + 2),
                 (card_center[0] + x_size + 2, card_center[1] - x_size + 2), (150, 150, 150), thickness)
        cv2.line(screen, (card_center[0] - x_size, card_center[1] - x_size),
                 (card_center[0] + x_size, card_center[1] + x_size), (255, 255, 255), thickness)
        cv2.line(screen, (card_center[0] - x_size, card_center[1] + x_size),
                 (card_center[0] + x_size, card_center[1] - x_size), (255, 255, 255), thickness)

    # Detección de la X
    close_card_radius = 50
    close_card_margin_x = 180
    close_card_margin_y = 80
    close_card_center_x = view_width - close_card_margin_x - close_card_radius
    close_card_center_y = close_card_margin_y + close_card_radius

    close_det_size = close_card_radius * 2.4
    close_det_x = close_card_center_x - close_card_radius * 1.2
    close_det_y = close_card_center_y - close_card_radius * 1.2
    close_det_w = close_det_size
    close_det_h = close_det_size

    def detectar_close(x_t, y_t):
        return close_det_x <= x_t <= close_det_x + close_det_w and close_det_y <= y_t <= close_det_y + close_det_h

    # ── Definir 3 dificultades ─────────────────────────────────────────

    dificultades = [
        {"id": 1, "nombre": "Facil",       "label": "Facil",       "stars": 1, "color_bg": (120, 220, 120), "color_border": (60, 180, 60)},
        {"id": 2, "nombre": "Intermedio",   "label": "Intermedio",  "stars": 2, "color_bg": (100, 190, 255), "color_border": (40, 140, 220)},
        {"id": 3, "nombre": "Dificil",      "label": "Dificil",     "stars": 3, "color_bg": (130, 120, 255), "color_border": (80, 60, 220)},
    ]

    # Dimensiones de las cards
    card_width = 300
    card_height = 350
    card_spacing = 50

    total_width = len(dificultades) * card_width + (len(dificultades) - 1) * card_spacing
    start_x = (view_width - total_width) // 2
    start_y = 230  # debajo del título

    dificultad_positions = {}
    for idx, d in enumerate(dificultades):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        dificultad_positions[d["id"]] = {
            "x": x, "y": y, "width": card_width, "height": card_height,
            "color_bg": d["color_bg"], "color_border": d["color_border"],
            "label": d["label"], "stars": d["stars"],
        }

    # ── Dibujar cards ──────────────────────────────────────────────────

    def draw_dificultad_cards(screen, elevated_card=None):
        for d_id, pos in dificultad_positions.items():
            x, y = pos["x"], pos["y"]
            w, h = pos["width"], pos["height"]
            bg = pos["color_bg"]
            border = pos["color_border"]
            label = pos["label"]
            stars = pos["stars"]

            is_elevated = (elevated_card == d_id)
            elevation_offset = -20 if is_elevated else 0
            sf = 1.06 if is_elevated else 1.0
            shadow_base = 14 if is_elevated else 8

            w_s = int(w * sf)
            h_s = int(h * sf)
            x_s = x - (w_s - w) // 2
            y_s = y + elevation_offset - (h_s - h) // 2
            x_s = max(0, min(x_s, screen.shape[1] - w_s))
            y_s = max(0, min(y_s, screen.shape[0] - h_s))

            # Sombra
            shadow_offset = int(shadow_base * sf)
            for i in range(3, 0, -1):
                sa = i / 3.0
                sc = tuple(int(c * sa) for c in (50, 50, 50))
                ol = shadow_offset + (3 - i)
                cv2.rectangle(screen, (x_s + ol, y_s + ol), (x_s + w_s + ol, y_s + h_s + ol), sc, -1)

            # Fondo
            if is_elevated:
                bg = tuple(min(255, int(c * 1.15)) for c in bg)
            cv2.rectangle(screen, (x_s, y_s), (x_s + w_s, y_s + h_s), bg, -1)

            # Borde
            bt = 6 if is_elevated else 4
            cv2.rectangle(screen, (x_s, y_s), (x_s + w_s, y_s + h_s), border, bt)

            # Estrellas
            stars_text = "\u2605 " * stars  # ★
            stars_fs = 2.0 * sf
            stars_th = 3
            stars_size, _ = cv2.getTextSize(stars_text.strip(), cv2.FONT_HERSHEY_SIMPLEX, stars_fs, stars_th)
            stars_x = x_s + (w_s - stars_size[0]) // 2
            stars_y = y_s + int(140 * sf)
            # Sombra de las estrellas
            cv2.putText(screen, stars_text.strip(), (stars_x + 2, stars_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, stars_fs, (0, 0, 0), stars_th + 1)
            cv2.putText(screen, stars_text.strip(), (stars_x, stars_y),
                        cv2.FONT_HERSHEY_SIMPLEX, stars_fs, (0, 200, 255), stars_th)

            # Label
            label_fs = 1.3 * sf
            label_th = 3
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, label_fs, label_th)
            label_x = x_s + (w_s - label_size[0]) // 2
            label_y = y_s + int(230 * sf)
            put_text_ubuntu(screen, label, (label_x + 2, label_y + 2), label_fs, (0, 0, 0), label_th + 1, bold=True)
            put_text_ubuntu(screen, label, (label_x, label_y), label_fs, (255, 255, 255), label_th, bold=True)

    # ── Título ─────────────────────────────────────────────────────────

    def draw_titulo(screen):
        titulo = "Elige la dificultad"
        fs = 1.5
        th = 3
        try:
            from PIL import Image, ImageDraw
            from src.core.font_utils import get_ubuntu_font
            font_u = get_ubuntu_font(font_scale=fs, bold=True)
            img_pil = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                bbox = draw.textbbox((0, 0), titulo, font=font_u)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                bbox = font_u.getbbox(titulo) if hasattr(font_u, "getbbox") else (0, 0, 0, 0)
                tw = bbox[2] - bbox[0]
        except Exception:
            text_size, _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_DUPLEX, fs, th)
            tw = text_size[0]
        tx = (view_width - tw) // 2
        ty = 185
        put_text_ubuntu(screen, titulo, (tx + 2, ty + 2), fs, (0, 0, 0), th + 1, bold=True)
        put_text_ubuntu(screen, titulo, (tx, ty), fs, (255, 255, 255), th, bold=True)

    # ── Configurar ventana ─────────────────────────────────────────────

    window_name = existing_window_name if existing_window_name else "Seleccion Dificultad"
    window_exists = False
    if existing_window_name:
        try:
            prop = cv2.getWindowProperty(existing_window_name, cv2.WND_PROP_VISIBLE)
            if prop >= 0:
                window_exists = True
        except Exception:
            window_exists = False

    if not window_exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1920, 0)
        cv2.waitKey(50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(50)
    else:
        try:
            cv2.moveWindow(window_name, 1920, 0)
            cv2.waitKey(10)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    # Dibujo inicial
    init_screen = _gradient_screen()
    draw_logo_func(init_screen)
    draw_titulo(init_screen)
    draw_dificultad_cards(init_screen)
    draw_close_card(init_screen)
    cv2.imshow(window_name, scale_to_videobeam(init_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT))

    # ── Streams de cámara ──────────────────────────────────────────────

    rgb_stream = device.create_color_stream()
    depth_stream = device.create_depth_stream()
    rgb_stream.start()
    depth_stream.start()

    close_elevated = False
    frame_count = 0
    initialization_delay = 10

    # Debounce
    from collections import defaultdict
    touch_history = defaultdict(list)
    min_touch_frames = 2
    min_touch_area = 100
    max_touch_area = 50000
    last_valid_touch_time = time.time()
    touch_cooldown = 0.15
    history_cleanup_interval = 30
    max_history_age = 1.0

    # ── Bucle principal ────────────────────────────────────────────────

    try:
        while True:
            frame_count += 1
            frame = rgb_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            if frame is None or depth_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            rgb_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            bgr_data = cv2.flip(bgr_data, 1)
            bgr_data = bgr_data[yw_min:yw_max, xw_min:xw_max]

            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)
            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
            kernel = np.ones((2, 2), np.uint8)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)

            # Redibujar
            temp_screen = _gradient_screen()
            draw_logo_func(temp_screen)
            draw_titulo(temp_screen)
            draw_dificultad_cards(temp_screen)
            draw_close_card(temp_screen, elevated=close_elevated)
            cv2.imshow(window_name, scale_to_videobeam(temp_screen, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT))

            if frame_count < initialization_delay:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            contours, _ = cv2.findContours(touch_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_time = time.time()
            if frame_count % history_cleanup_interval == 0:
                for key_h in list(touch_history.keys()):
                    touch_history[key_h] = [t for t in touch_history[key_h] if current_time - t[3] < max_history_age]
                    if not touch_history[key_h]:
                        del touch_history[key_h]

            valid_touches = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_touch_area <= area <= max_touch_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x_touch = int(xv_min + cx * (xv_max - xv_min) / (xw_max - xw_min))
                        y_touch = int(yv_min + cy * (yv_max - yv_min) / (yw_max - yw_min))

                        touch_key = (x_touch // 25, y_touch // 25)
                        touch_history[touch_key].append((x_touch, y_touch, area, current_time))

                        if len(touch_history[touch_key]) >= min_touch_frames:
                            if current_time - last_valid_touch_time > touch_cooldown:
                                recent = touch_history[touch_key][-min_touch_frames:]
                                all_recent = all(current_time - t[3] < 1.0 for t in recent)
                                if all_recent and len(recent) >= min_touch_frames:
                                    areas = [t[2] for t in recent]
                                    avg_area = sum(areas) / len(areas)
                                    if min(areas) > 0:
                                        variance = max(areas) / min(areas)
                                        if variance < 3.5 and min_touch_area <= avg_area <= max_touch_area:
                                            valid_touches.append((x_touch, y_touch, touch_key))

            for x_touch, y_touch, touch_key in valid_touches:
                if touch_key in touch_history:
                    del touch_history[touch_key]
                last_valid_touch_time = current_time

                # ¿Cerrar?
                if detectar_close(x_touch, y_touch):
                    print("Card de cerrar tocada en selección de dificultad")
                    close_elevated = True
                    temp = _gradient_screen()
                    draw_logo_func(temp)
                    draw_titulo(temp)
                    draw_dificultad_cards(temp)
                    draw_close_card(temp, elevated=True)
                    cv2.imshow(window_name, scale_to_videobeam(temp, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT))
                    cv2.waitKey(200)
                    rgb_stream.stop()
                    depth_stream.stop()
                    return None

                # ¿Seleccionó una dificultad?
                for d_id, pos in dificultad_positions.items():
                    px, py = pos["x"], pos["y"]
                    pw, ph = pos["width"], pos["height"]
                    if px <= x_touch <= px + pw and py <= y_touch <= py + ph:
                        print(f"Dificultad seleccionada: {d_id} ({pos['label']})")
                        # Efecto de elevación
                        temp = _gradient_screen()
                        draw_logo_func(temp)
                        draw_titulo(temp)
                        draw_dificultad_cards(temp, elevated_card=d_id)
                        draw_close_card(temp)
                        cv2.imshow(window_name, scale_to_videobeam(temp, videobeam_width=VIDEOBEAM_WIDTH, videobeam_height=VIDEOBEAM_HEIGHT))
                        cv2.waitKey(200)
                        rgb_stream.stop()
                        depth_stream.stop()
                        return d_id

                close_elevated = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            close_elevated = False

    finally:
        try:
            rgb_stream.stop()
            depth_stream.stop()
        except Exception:
            pass

    return None
