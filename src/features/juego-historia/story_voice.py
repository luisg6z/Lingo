"""
Voice listen and Ollama validation for juego-historia story.
"""
import os
import json
import re
import threading
import time
from dotenv import load_dotenv
import speech_recognition as sr
from ollama import Client


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


def listen_and_transcribe(timeout=10, phrase_time_limit=10, language="es-ES", on_listening_started=None,
                         silence_seconds=5.0):
    """
    Listen to the microphone for up to phrase_time_limit seconds and return transcribed text.
    Uses a configurable silence duration so that only sustained silence ends the phrase,
    avoiding cuts mid-sentence when the speaker pauses briefly.

    Args:
        timeout: Max seconds to wait for speech to start.
        phrase_time_limit: Max recording length in seconds (10 for the game).
        language: Language code for recognition (es-ES for Spanish).
        on_listening_started: Optional callback invoked after listening has started (so the UI
            can show "now you can talk" without missing the beginning of speech).
        silence_seconds: Seconds of continuous silence required before stopping the recording.
            Higher values (e.g. 4.0–6.0) reduce mid-sentence cuts; lower values end sooner.

    Returns:
        Transcribed text string, or None if nothing heard or recognition failed.
    """
    recognizer = sr.Recognizer()
    # Emparejar configuración con el flujo que ya funciona mejor en otros juegos.
    # `adjust_for_ambient_noise()` terminará recalibrando, pero este umbral ayuda como punto de partida.
    recognizer.energy_threshold = 300
    # Require sustained silence before stopping: avoids cutting when someone pauses briefly
    recognizer.pause_threshold = max(1.0, float(silence_seconds))
    recognizer.phrase_threshold = 0.3
    # Keep a bit of non-speaking audio so we don't clip the very end of the phrase
    recognizer.non_speaking_duration = 0.5
    recognizer.operation_timeout = None  # Evitar timeouts raros en operaciones de escucha
    recognizer.dynamic_energy_threshold = True
    # Permite cambiar el micrófono si el dispositivo 2 no coincide en tu PC.
    # Si no existe la variable, usamos el mismo índice que ya viene en el juego.
    device_index = int(os.environ.get("LINGO_MIC_DEVICE_INDEX", "2"))
    try:
        microphone = sr.Microphone(device_index=device_index)
        print(f"Usando micrófono (device_index={device_index}) para juego-historia.")
    except Exception as e:
        # Fallback a "dispositivo por defecto" si el índice no es válido.
        try:
            microphone = sr.Microphone()
            print(f"Advertencia: no se pudo abrir el micrófono index={device_index}. Usando dispositivo por defecto.")
        except Exception:
            print(f"Error al inicializar micrófono: {e}")
            return None
    try:
        with microphone as source:
            # Si el stream no se abrió correctamente, evitamos que SpeechRecognition
            # lance errores internos al cerrar un stream nulo.
            if getattr(source, "stream", None) is None:
                print("Error al ajustar ruido ambiente: el micrófono no se inicializó correctamente (stream vacío).")
                print("Revisa la configuración de audio antes de volver a intentar.")
                return None
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
    except Exception as e:
        print(f"Error al ajustar ruido ambiente: {e}")
        return None

    result_holder = [None]
    exception_holder = [None]

    def _listen_thread():
        try:
            with microphone as source:
                if getattr(source, "stream", None) is None:
                    exception_holder[0] = RuntimeError(
                        "El micrófono no se inicializó correctamente (stream vacío) al intentar escuchar."
                    )
                    return
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
        # Probar acentos alternativos si el español del usuario no coincide con `es-ES`.
        candidate_languages = [language]
        if (language or "").lower() in {"es-es", "es_es", "es-pt", "es"}:
            # es-419 suele cubrir mejor gran parte de español latino.
            if "es-419" not in candidate_languages:
                candidate_languages.append("es-419")

        last_unknown = None
        for lang in candidate_languages:
            try:
                text = recognizer.recognize_google(audio, language=lang)
                return (text or "").strip() or None
            except sr.UnknownValueError as e_unknown:
                last_unknown = e_unknown
                continue

        # Si ninguna variante pudo entenderlo
        return None
    except sr.RequestError as e:
        print(f"Error del servicio de reconocimiento: {e}")
        return None
    except Exception as e:
        print(f"Error al transcribir: {e}")
        return None


def _get_ollama_client():
    """
    Crea un cliente de Ollama usando las variables de entorno
    OLLAMA_HOST y OLLAMA_API_KEY para conectarse al servicio en la nube.
    """
    load_dotenv()
    host = os.environ.get("OLLAMA_HOST")
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not host or not api_key:
        print("Error: OLLAMA_HOST u OLLAMA_API_KEY no están configuradas en el entorno.")
        return None
    try:
        return Client(host=host, headers={'Authorization': 'Bearer ' + api_key})
    except Exception as e:
        print(f"Error al crear el cliente de Ollama: {e}")
        return None


def verify_story_ollama(sentence, subjects=None, actions=None, places=None, model="deepseek-v3"):
    """
    Pide a Ollama que valide la historia usando los personajes, acciones y lugares dados.
    Público objetivo: niños de ~7 años. La respuesta debe ser SOLO JSON.

    Args:
        sentence: Historia u oración del niño.
        subjects: Lista de nombres de personajes (ej. ["Niño", "Niña"]).
        actions: Lista de nombres de acciones (ej. ["Dar"]).
        places: Lista de nombres de lugares (ej. ["Casa"]).
        model: Nombre del modelo de Ollama.

    Returns:
        dict: {
            "correct": bool,
            "tips": list[str],
            "parts": {
                "subjects": list[str],  # fragmentos de la historia que corresponden a los sujetos dados
                "actions": list[str],   # fragmentos de la historia que corresponden a las acciones dadas
                "places": list[str],    # fragmentos de la historia que corresponden a los lugares dados
            },
        }
        En caso de error se devuelve:
        {"correct": False, "tips": ["Revisa tu oración."], "parts": {"subjects": [], "actions": [], "places": []}}
    """
    default_fail = {
        "correct": False,
        "tips": ["Revisa tu oración."],
        "parts": {"subjects": [], "actions": [], "places": []},
    }
    if not sentence or not sentence.strip():
        print("No se proporcionó historia.")
        return default_fail

    client = _get_ollama_client()
    if client is None:
        print("No se pudo crear el cliente de Ollama.")
        return default_fail

    subjects_str = ", ".join(str(s) for s in (subjects or [])) if subjects else "(ninguno)"
    actions_str = ", ".join(str(a) for a in (actions or [])) if actions else "(ninguna)"
    places_str = ", ".join(str(p) for p in (places or [])) if places else "(ninguno)"

    # Sanitizar la historia para no romper el prompt (comillas, saltos de línea o control)
    sentence_clean = (sentence or "").strip().replace("\r", " ").replace("\n", " ")
    sentence_clean = sentence_clean.replace('"', "'")[:2000]  # límite razonable de longitud

    prompt = f"""Eres un profesor de español para niños de 7 años. Evalúa historias cortas de forma alentadora según los criterios siguientes.



CRITERIOS DE EVALUACIÓN:

1. SENTIDO GRAMATICAL CORRECTO: ¿Se entiende la idea? Debe haber coherencia básica.

2. TIEMPOS VERBALES CORRECTOS: Uso correcto de presente, pasado o futuro.

3. PARTÍCULAS DE ENLACE: Debe usar al menos uno (y, entonces, luego, porque, pero, etc.).

4. ELEMENTOS: Debe incluir (o variaciones claras de):

   - Personajes (sujetos): {subjects_str}

   - Acciones: {actions_str}

   - Lugares: {places_str}

5. CIERRE/CONCLUSIÓN:La historia no puede quedar a medias; debe tener un final.



INSTRUCCIONES PARA LOS "TIPS":

Sé breve y muy amable.

Si hay un error, usa el formato: "Dijiste '[error]', pero quedaría mejor así: '[corrección]'".

Si la historia es perfecta, usa el primer tip para felicitar un punto específico (ej: "¡Me encantó cómo usaste el conector 'porque'!") y deja el resto del array vacío.



ANÁLISIS PARA "parts":

- "subjects": fragmentos EXACTOS del texto del niño que correspondan a los sujetos dados.

- "actions": fragmentos EXACTOS del texto del niño que correspondan a las acciones dadas.

- "places": fragmentos EXACTOS del texto del niño que correspondan a los lugares dados.



Historia del niño: "{sentence_clean}"



FORMATO DE RESPUESTA (SOLO JSON EN UNA LÍNEA, sin otro texto):

{{"correct": true o false, "tips": ["consejo1", "consejo2"], "parts": {{"subjects": ["fragmento1"], "actions": ["fragmento2"], "places": ["fragmento3"]}}}}
"""

    try:
        messages = [{"role": "user", "content": prompt}]

        content_parts = []
        for part in client.chat(model, messages=messages, stream=True):
            # Algunos APIs devuelven error en el stream
            if part.get("error"):
                print(f"Ollama devolvió error en stream: {part.get('error')}")
                return default_fail
            msg = (part.get("message") or {}).get("content")
            if msg:
                content_parts.append(msg)

        content = "".join(content_parts).strip()
        if not content:
            print("No se recibió respuesta de Ollama (contenido vacío).")
            print(f"content parts: {content_parts}")
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
        # Extraer partes (subjects, actions, places) y asegurar que sean listas
        subjects = parts.get("subjects", [])
        actions = parts.get("actions", [])
        places = parts.get("places", [])
        if not isinstance(subjects, list):
            subjects = [subjects] if subjects else []
        if not isinstance(actions, list):
            actions = [actions] if actions else []
        if not isinstance(places, list):
            places = [places] if places else []
        return {
            "correct": correct,
            "tips": tips,
            "parts": {
                "subjects": subjects,
                "actions": actions,
                "places": places,
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error al decodificar el JSON de Ollama: {e}")
        return default_fail
    except Exception as e:
        import traceback
        print(f"Error Ollama ({type(e).__name__}): {e}")
        traceback.print_exc()
        return default_fail
