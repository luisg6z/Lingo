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
