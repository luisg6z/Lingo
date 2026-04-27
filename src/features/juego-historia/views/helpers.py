"""
Helpers for juego-historia views: text rendering wrapper and TTS.
"""
import os
import sys
import threading
import importlib.util

import cv2

# Ensure project root is on path (views are under src/features/juego-historia/views/)
_views_dir = os.path.dirname(os.path.abspath(__file__))
_feature_dir = os.path.dirname(_views_dir)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_feature_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.font_utils import put_text_ubuntu as _put_text_ubuntu_core


def put_text_safe_historia(img, text, position, font_face, font_scale, color, thickness, line_type=cv2.LINE_AA, bold=False):
    """
    Render text using Ubuntu font. Wrapper for compatibility (font_face ignored).
    """
    _put_text_ubuntu_core(img, text, position, font_scale, color, thickness, line_type=line_type, bold=bold)


# TTS: speak card name when selected (non-blocking, daemon thread)
_tts_lock = threading.Lock()

HISTORIA_TTS_SUJETO = {
    "Niño": "El niño",
    "Niña": "La niña",
    "Doctor": "El doctor",
    "Maestra": "La maestra",
    "Policia": "El policía",
    "Perro": "El perro",
}
HISTORIA_TTS_LUGAR = {
    "Calle": "La calle",
    "Clinica": "La clínica",
    "Estacion-Policia": "La estación de policía",
    "Escuela": "La escuela",
    "Casa": "La casa",
    "Parque": "El parque",
}
HISTORIA_TTS_ACCION = {
    "Dar": "Dar",
    "Ayudar": "Ayudar",
    "Correr": "Correr",
    "Jugar": "Jugar",
    "Llamar": "Llamar",
    "Trabajar": "Trabajar",
}


def historia_tts_speak(name, tipo=None):
    """
    Speak the given name using pyttsx3 in a daemon thread.
    tipo: 'sujeto' | 'accion' | 'lugar' to use the correct article; None = use name as-is.
    """
    try:
        import pyttsx3
    except ImportError:
        return
    if tipo == "sujeto":
        text = HISTORIA_TTS_SUJETO.get(name, name)
    elif tipo == "lugar":
        text = HISTORIA_TTS_LUGAR.get(name, name)
    elif tipo == "accion":
        text = HISTORIA_TTS_ACCION.get(name, name)
    else:
        text = name

    def _run():
        if not _tts_lock.acquire(blocking=False):
            return
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            voices = engine.getProperty("voices")
            for v in voices:
                if "spanish" in v.name.lower() or "español" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as e:
            if "run loop" not in str(e).lower():
                print(f"TTS error: {e}")
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            _tts_lock.release()

    t = threading.Thread(target=_run, daemon=True)
    t.start()


_historia_story_voice_module = None


def get_historia_story_voice():
    """Load and return the story_voice module (juego-historia/story_voice.py)."""
    global _historia_story_voice_module
    if _historia_story_voice_module is None:
        story_voice_path = os.path.join(_feature_dir, "story_voice.py")
        spec = importlib.util.spec_from_file_location("historia_story_voice", story_voice_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _historia_story_voice_module = mod
    return _historia_story_voice_module


def _historia_asset_png(filename):
    return os.path.join(_feature_dir, "assets", "images", filename)


# Misma geometría que `vista_final` (cards + botón Hablar): una sola fuente de verdad
HISTORIA_RESUMEN_CARD_WIDTH = 250
HISTORIA_RESUMEN_CARD_HEIGHT = 300
HISTORIA_RESUMEN_CARD_SPACING = 30
HISTORIA_RESUMEN_CARDS_START_Y = 250

HISTORIA_HABLAR_RADIUS = 60
HISTORIA_HABLAR_FRANJA_BOTTOM = 100


def historia_hablar_button_layout(view_width, view_height):
    """Centro y radio del botón circular Hablar (alineado con vista_final)."""
    r = HISTORIA_HABLAR_RADIUS
    cx = int(view_width) // 2
    cy = int(view_height) - HISTORIA_HABLAR_FRANJA_BOTTOM - r
    return cx, cy, r


def historia_resumen_status_line_y():
    """Coordenada Y (baseline aprox.) del mensaje bajo las imágenes, como la franja sobre Hablar."""
    return HISTORIA_RESUMEN_CARDS_START_Y + HISTORIA_RESUMEN_CARD_HEIGHT + 20


def build_historia_voice_prep_cards(
    sujetos_seleccionados,
    acciones_seleccionadas,
    lugares_seleccionados,
    view_width,
    card_width=HISTORIA_RESUMEN_CARD_WIDTH,
    card_height=HISTORIA_RESUMEN_CARD_HEIGHT,
    card_spacing=HISTORIA_RESUMEN_CARD_SPACING,
    start_y=HISTORIA_RESUMEN_CARDS_START_Y,
):
    """
    Selección actual (hasta 2 sujetos, 1 acción, 1 lugar): rutas e imágenes cargadas, posiciones en fila centrada.
    Devuelve dict ordenado nombre -> {x, y, width, height, color, imagen}.
    """
    historias = [
        {"nombre": "Niño", "color": (255, 150, 200), "file": "Niño.png"},
        {"nombre": "Niña", "color": (200, 150, 255), "file": "Niña.png"},
        {"nombre": "Cocinera", "color": (150, 255, 200), "file": "Cocinera.png"},
        {"nombre": "Policia", "color": (200, 255, 150), "file": "Policia.png"},
        {"nombre": "Doctor", "color": (150, 200, 255), "file": "Doctor.png"},
        {"nombre": "Maestra", "color": (255, 200, 150), "file": "Maestra.png"},
    ]
    acciones = [
        {"nombre": "Ayudar", "color": (255, 150, 200), "file": "Ayudar.png"},
        {"nombre": "Trabajar", "color": (200, 150, 255), "file": "Trabajar.png"},
        {"nombre": "Cocinar", "color": (150, 255, 200), "file": "Cocinar.png"},
        {"nombre": "Correr", "color": (255, 200, 150), "file": "Correr.png"},
        {"nombre": "Llamar", "color": (200, 255, 150), "file": "Llamar.png"},
        {"nombre": "Jugar", "color": (150, 200, 255), "file": "Jugar.png"},
    ]
    lugares = [
        {"nombre": "Casa", "color": (255, 150, 200), "file": "Casa.png"},
        {"nombre": "Clinica", "color": (200, 150, 255), "file": "Clinica.png"},
        {"nombre": "Escuela", "color": (150, 255, 200), "file": "Escuela.png"},
        {"nombre": "Estacion-policia", "color": (255, 200, 150), "file": "Estacion-Policia.png"},
        {"nombre": "Parque", "color": (200, 255, 150), "file": "Parque.png"},
        {"nombre": "Cocina", "color": (150, 200, 255), "file": "Cocina.png"},
    ]

    def _load_map(rows):
        m = {}
        for row in rows:
            path = _historia_asset_png(row["file"])
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    m[row["nombre"]] = img
        return m

    historia_images = _load_map(historias)
    accion_images = _load_map(acciones)
    lugar_images = _load_map(lugares)

    cards_final = []
    for sujeto in (sujetos_seleccionados or [])[:2]:
        for h in historias:
            if h["nombre"] == sujeto:
                cards_final.append(
                    {
                        "nombre": sujeto,
                        "tipo": "sujeto",
                        "color": h["color"],
                        "imagen": historia_images.get(sujeto),
                    }
                )
                break
    if acciones_seleccionadas and len(acciones_seleccionadas) > 0:
        accion = acciones_seleccionadas[0]
        for a in acciones:
            if a["nombre"] == accion:
                cards_final.append(
                    {
                        "nombre": accion,
                        "tipo": "accion",
                        "color": a["color"],
                        "imagen": accion_images.get(accion),
                    }
                )
                break
    if lugares_seleccionados and len(lugares_seleccionados) > 0:
        lugar = lugares_seleccionados[0]
        for lug in lugares:
            if lug["nombre"] == lugar:
                cards_final.append(
                    {
                        "nombre": lugar,
                        "tipo": "lugar",
                        "color": lug["color"],
                        "imagen": lugar_images.get(lugar),
                    }
                )
                break

    card_positions = {}
    if not cards_final:
        return card_positions

    n = len(cards_final)
    total_width = n * card_width + (n - 1) * card_spacing
    start_x = max(0, (view_width - total_width) // 2)
    for idx, card in enumerate(cards_final):
        x = start_x + idx * (card_width + card_spacing)
        y = start_y
        card_positions[card["nombre"]] = {
            "x": x,
            "y": y,
            "width": card_width,
            "height": card_height,
            "color": card["color"],
            "imagen": card["imagen"],
        }
    return card_positions


# ── Difficulty badge ────────────────────────────────────────────────

_DIFFICULTY_META = {
    1: {"label": "Facil",      "stars": 1, "bg_color": (80, 190, 80),   "text_color": (255, 255, 255)},
    2: {"label": "Intermedio",  "stars": 2, "bg_color": (60, 160, 240),  "text_color": (255, 255, 255)},
    3: {"label": "Dificil",     "stars": 3, "bg_color": (120, 80, 220),  "text_color": (255, 255, 255)},
}


def draw_difficulty_badge(screen, difficulty_level, view_width=1280):
    """
    Draw a small coloured pill badge in the top-left area showing the
    current difficulty level (stars + label).

    Args:
        screen: numpy image (BGR)
        difficulty_level: 1, 2 or 3
        view_width: viewport width (default 1280)
    """
    if difficulty_level not in _DIFFICULTY_META:
        return

    meta = _DIFFICULTY_META[difficulty_level]
    stars_text = "\u2605" * meta["stars"]  # ★
    label = f" {stars_text}  {meta['label']} "
    bg = meta["bg_color"]
    fg = meta["text_color"]

    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 0.7
    th = 2
    (tw, th_px), baseline = cv2.getTextSize(label, font, fs, th)

    badge_x = 15
    badge_y = 148
    pad_x = 10
    pad_y = 6
    badge_w = tw + 2 * pad_x
    badge_h = th_px + baseline + 2 * pad_y

    # Rounded-rect background (simulated with filled rect + circles at corners)
    x1, y1 = badge_x, badge_y
    x2, y2 = badge_x + badge_w, badge_y + badge_h
    radius = min(12, badge_h // 2)

    # Draw filled rounded rectangle
    cv2.rectangle(screen, (x1 + radius, y1), (x2 - radius, y2), bg, -1)
    cv2.rectangle(screen, (x1, y1 + radius), (x2, y2 - radius), bg, -1)
    cv2.circle(screen, (x1 + radius, y1 + radius), radius, bg, -1)
    cv2.circle(screen, (x2 - radius, y1 + radius), radius, bg, -1)
    cv2.circle(screen, (x1 + radius, y2 - radius), radius, bg, -1)
    cv2.circle(screen, (x2 - radius, y2 - radius), radius, bg, -1)

    # Border
    cv2.rectangle(screen, (x1 + radius, y1), (x2 - radius, y1 + 1), (255, 255, 255), 1)
    cv2.rectangle(screen, (x1 + radius, y2 - 1), (x2 - radius, y2), (255, 255, 255), 1)
    cv2.rectangle(screen, (x1, y1 + radius), (x1 + 1, y2 - radius), (255, 255, 255), 1)
    cv2.rectangle(screen, (x2 - 1, y1 + radius), (x2, y2 - radius), (255, 255, 255), 1)

    # Text
    text_x = badge_x + pad_x
    text_y = badge_y + pad_y + th_px
    cv2.putText(screen, label, (text_x, text_y), font, fs, fg, th, cv2.LINE_AA)


def draw_historia_voice_prep_cards(screen, card_positions):
    """Dibuja cards con imagen o color de respaldo (misma idea que vista final)."""
    for _nombre, pos in card_positions.items():
        x, y = pos["x"], pos["y"]
        w, h = pos["width"], pos["height"]
        color = pos["color"]
        img = pos.get("imagen")

        shadow_offset = 5
        shadow_color = (40, 40, 40)
        for i in range(3, 0, -1):
            shadow_alpha = i / 3.0
            shadow_color_layer = tuple(int(c * shadow_alpha) for c in shadow_color)
            offset_layer = shadow_offset + (3 - i)
            cv2.rectangle(
                screen,
                (x + offset_layer, y + offset_layer),
                (x + w + offset_layer, y + h + offset_layer),
                shadow_color_layer,
                -1,
            )

        if img is not None:
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
                alpha = img_resized[:, :, 3] / 255.0
                img_bgr = img_resized[:, :, :3]
                for c in range(3):
                    screen[y : y + h, x : x + w, c] = (
                        alpha * img_bgr[:, :, c] + (1 - alpha) * screen[y : y + h, x : x + w, c]
                    )
            else:
                screen[y : y + h, x : x + w] = img_resized[:, :, :3]
        else:
            cv2.rectangle(screen, (x, y), (x + w, y + h), color, -1)

        cv2.rectangle(screen, (x, y), (x + w, y + h), (100, 100, 100), 2)
