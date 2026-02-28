"""
Voice listen and Ollama validation for juego-historia story.
"""
import os
import json
import re
import threading
import time
import speech_recognition as sr

try:
    import ollama
except ImportError:
    ollama = None


def get_required_words(config_path, sujetos_seleccionados, acciones_seleccionadas, lugares_seleccionados):
    """
    Load palabras_imagenes.json and return a flat list of Spanish words that must appear
    in the story for the selected cards.

    Args:
        config_path: Path to palabras_imagenes.json
        sujetos_seleccionados: List of selected subject names (e.g. ["Niño", "Niña"])
        acciones_seleccionadas: List of one action name (e.g. ["Dar"])
        lugares_seleccionados: List of one place name (e.g. ["Casa"])

    Returns:
        List of strings (lowercase) that the story should include. May contain variants (e.g. "niño", "nino").
    """
    required = []
    if not os.path.exists(config_path):
        return required
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return required

    for key in (sujetos_seleccionados or []):
        words = (data.get("sujetos") or {}).get(key)
        if words:
            required.extend(w.lower() for w in words)
    for key in (acciones_seleccionadas or [])[:1]:
        words = (data.get("acciones") or {}).get(key)
        if words:
            required.extend(w.lower() for w in words)
    for key in (lugares_seleccionados or [])[:1]:
        words = (data.get("lugares") or {}).get(key)
        if words:
            required.extend(w.lower() for w in words)
    return list(dict.fromkeys(required))  # unique, order preserved


def listen_and_transcribe(timeout=10, phrase_time_limit=10, language="es-ES", on_listening_started=None):
    """
    Listen to the microphone for up to phrase_time_limit seconds and return transcribed text.
    Uses a longer pause_threshold to avoid cutting phrases. Call on_listening_started (e.g. to
    show "Ahora puedes hablar") only after listening has actually started.

    Args:
        timeout: Max seconds to wait for speech to start.
        phrase_time_limit: Max recording length in seconds (10 for the game).
        language: Language code for recognition (es-ES for Spanish).
        on_listening_started: Optional callback invoked after listening has started (so the UI
            can show "now you can talk" without missing the beginning of speech).

    Returns:
        Transcribed text string, or None if nothing heard or recognition failed.
    """
    recognizer = sr.Recognizer()
    # Avoid cutting phrases: require longer silence before stopping (default 0.8 is too short)
    recognizer.pause_threshold = 1.3
    recognizer.phrase_threshold = 0.3
    recognizer.dynamic_energy_threshold = True
    try:
        microphone = sr.Microphone()
    except Exception as e:
        print(f"Error al inicializar micrófono: {e}")
        return None
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
    except Exception as e:
        print(f"Error al ajustar ruido ambiente: {e}")

    result_holder = [None]
    exception_holder = [None]

    def _listen_thread():
        try:
            with microphone as source:
                audio = recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
                result_holder[0] = audio
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            exception_holder[0] = e

    thread = threading.Thread(target=_listen_thread, daemon=True)
    thread.start()
    # Give listen() time to start capturing; then signal so UI can show "Ahora puedes hablar"
    time.sleep(0.5)
    if callable(on_listening_started):
        try:
            on_listening_started()
        except Exception:
            pass
    thread.join()

    if exception_holder[0]:
        print(f"Error al escuchar: {exception_holder[0]}")
        return None
    audio = result_holder[0]
    if audio is None:
        return None
    try:
        text = recognizer.recognize_google(audio, language=language)
        return (text or "").strip() or None
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Error del servicio de reconocimiento: {e}")
        return None
    except Exception as e:
        print(f"Error al transcribir: {e}")
        return None


def verify_story_ollama(sentence, subjects=None, actions=None, places=None, model="deepseek-r1"):
    """
    Ask Ollama to validate the story using the given personajes, acciones y lugares.
    Target: children ~7 years old; respond with JSON only.

    Args:
        sentence: The user's sentence/story.
        subjects: List of character names (e.g. ["Niño", "Niña"]).
        actions: List of action names (e.g. ["Dar"]).
        places: List of place names (e.g. ["Casa"]).
        model: Ollama model name.

    Returns:
        dict: {"correct": bool, "tips": list of str}. On error, returns {"correct": False, "tips": ["Revisa tu oración."]}.
    """
    default_fail = {"correct": False, "tips": ["Revisa tu oración."], "parts": {"subjects": [], "actions": [], "predicates": []}}
    if not sentence or not sentence.strip():
        return default_fail
    if ollama is None:
        return default_fail

    subjects_str = ", ".join(subjects) if subjects else "(ninguno)"
    actions_str = ", ".join(actions) if actions else "(ninguna)"
    places_str = ", ".join(places) if places else "(ninguno)"

    prompt = f"""Eres un profesor de español para niños de 7 años. Tu objetivo es evaluar historias cortas de forma alentadora y flexible.

CRITERIOS DE EVALUACIÓN:
1. ELEMENTOS: Debe incluir en la historia los siguientes elementos (o usar variaciones de): 
   - Personajes: {subjects_str}
   - Acciones: {actions_str}
   - Lugares: {places_str}
2. ENLACES: Debe usar al menos un conector (ej: "y", "entonces", "luego", "después", "porque").
3. CIERRE/CONCLUSIÓN: La historia debe tener un final. No tiene que ser un "Fin" formal; cuenta como conclusión cualquier resultado o consecuencia de las acciones (ej: "se quedaron dormidos", "ganaron el juego", "se pusieron felices" o "se fueron a casa").

REGLAS DE ORO:
- Sé muy flexible con la gramática y tiempos verbales.
- Si falta alguna de las palabras obligatorias, da un consejo breve, dulce y específico para completar la historia.
- No des lecciones de ortografía, enfócate en la narrativa.
- En caso de que la historia tenga todo lo necesario, solo felicita al niño por su historia.
- Extrae TODOS los sujetos, acciones (verbos) y predicados de la historia y devuélvelos en el campo "parts".

Historia del niño: "{sentence}"

Responde ÚNICAMENTE con un JSON válido en una línea. Formato exacto:
{{"correct": true o false, "tips": ["consejo1", "consejo2"], "parts": {{"subjects": ["sujeto1", "sujeto2"], "actions": ["verbo1", "verbo2"], "predicates": ["predicado1", "predicado2"]}}}}
- "subjects": Lista de todas las palabras que son sujetos en la historia
- "actions": Lista de todas las palabras que son verbos/acciones en la historia
- "predicates": Lista de todas las palabras que forman parte de los predicados (puede incluir verbos si forman parte del predicado)
Si la historia es correcta (tiene todos los elementos requeridos), devuelve true en el campo correct.
Los consejos deben ser dirigidos al niño, no a un profesor.
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response or {}).get("message") or {}
        content = (raw.get("content") or "").strip()
        if not content:
            return default_fail
        # Extract JSON: allow markdown code block or plain JSON
        content = re.sub(r"^```\w*\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)
        content = content.strip()
        # Find first { ... }
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        data = json.loads(content)
        correct = data.get("correct", False)
        tips = data.get("tips")
        parts = data.get("parts", {})
        print(f"Model Answer: {content}")
        if not isinstance(tips, list):
            tips = [str(tips)] if tips else []
        # Extract parts (subjects, actions, predicates) - ensure they are lists
        subjects = parts.get("subjects", [])
        actions = parts.get("actions", [])
        predicates = parts.get("predicates", [])
        if not isinstance(subjects, list):
            subjects = [subjects] if subjects else []
        if not isinstance(actions, list):
            actions = [actions] if actions else []
        if not isinstance(predicates, list):
            predicates = [predicates] if predicates else []
        return {
            "correct": correct,
            "tips": tips,
            "parts": {
                "subjects": subjects,
                "actions": actions,
                "predicates": predicates
            }
        }
    except json.JSONDecodeError:
        return default_fail
    except Exception as e:
        print(f"Error Ollama: {e}")
        return default_fail
