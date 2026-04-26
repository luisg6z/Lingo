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

# Muletillas sueltas que Google STT suele volcar tal cual; se quitan solo como palabras completas.
_TRANSCRIPTION_FILLER_RE = re.compile(
    r"\b(?:"
    r"m{2,}|m+h+m*|mhmm|mhm+|"
    r"ehm+|eh+|em+|umm?|"
    r"ajá|aja|"
    r"uu+h*|"
    r"\buf\b"
    r")(?:[,.])?\s*",
    re.IGNORECASE,
)


def polish_transcription_for_display(text):
    """
    Quita muletillas aisladas típicas del STT (mmm, ajá, eh…) y normaliza espacios.
    No reescribe gramática ni corrige palabras de contenido; solo aligera la redacción mostrada.
    """
    if not text:
        return ""
    t = " ".join(str(text).split())
    t = re.sub(r"\bo\s+sea\b", "", t, flags=re.IGNORECASE)
    t = " ".join(t.split())
    for _ in range(24):
        nt = _TRANSCRIPTION_FILLER_RE.sub("", t)
        nt = " ".join(nt.split())
        if nt == t:
            break
        t = nt
    t = re.sub(r"^[,\.;:–—\-]+\s*", "", t)
    return t.strip()


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


def listen_and_transcribe(timeout=10, phrase_time_limit=None, language="es-ES", on_listening_started=None,
                         silence_seconds=2.0):
    """
    Escucha el micrófono hasta que haya silencio sostenido (por defecto 2 s) o hasta un tope
    opcional de duración de frase. Devuelve el texto transcrito.

    Args:
        timeout: Segundos máximos esperando a que empiece el habla.
        phrase_time_limit: Segundos máximos de grabación de la frase; ``None`` = sin tope
            (la frase termina sobre todo por ``silence_seconds`` de silencio).
        language: Language code for recognition (es-ES for Spanish).
        on_listening_started: Optional callback invoked after listening has started (so the UI
            can show "now you can talk" without missing the beginning of speech).
        silence_seconds: Segundos de silencio continuo para considerar terminada la frase
            (``pause_threshold`` del reconocedor). Por defecto 2.0.
        LINGO_MIC_AMBIENT_CALIB_SEC: segundos de calibración de ruido ambiente (default 0 = desactivada).
            Pon p. ej. 0.4 o 1 si en un aula muy ruidosa corta mal el fin de frase.

    Returns:
        Transcribed text string (tras quitar muletillas sueltas típicas del STT), o None
        si no hubo audio o falló el reconocimiento.
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
    # Juego historia: micrófono fijo índice 2 (PyAudio / lista de dispositivos del sistema).
    device_index = 2
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
            # Calibración de ruido ambiente: opcional; por defecto desactivada (flujo inmediato, sin espera).
            # Actívala con LINGO_MIC_AMBIENT_CALIB_SEC=0.4 (o hasta 2) si hace falta en ambientes ruidosos.
            _ambient = float(os.environ.get("LINGO_MIC_AMBIENT_CALIB_SEC", "0"))
            if _ambient > 0:
                recognizer.adjust_for_ambient_noise(source, duration=min(_ambient, 2.0))
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
    # Breve margen para que el hilo entre en `listen()`; la UI de "Ahora habla" puede
    # pintarse antes en voice_flow (hilo principal) para no esperar aquí.
    time.sleep(0.03)
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
                raw = (text or "").strip()
                if not raw:
                    return None
                polished = polish_transcription_for_display(raw)
                # Si solo había muletillas, conservar el bruto para no perder la “intención” de haber hablado.
                out = polished if polished else raw
                return out or None
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


def verify_story_ollama(sentence, subjects=None, actions=None, places=None, model="gemini-3-flash-preview"):
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

    prompt = f"""
Eres un profesor de español para niños de 7 años. Evalúa historias cortas de forma alentadora según los criterios siguientes.

CRITERIOS DE EVALUACIÓN:

1. SENTIDO GRAMATICAL CORRECTO: ¿Se entiende la idea? Debe haber coherencia básica.
2. TIEMPOS VERBALES CORRECTOS: Uso correcto de presente, pasado o futuro.
3. PARTÍCULAS DE ENLACE: Debe usar al menos uno (y, entonces, luego, porque, pero, etc.).
4. ELEMENTOS: DEBE incluir (ya sean variaciones, conjugaciones, etc.) OBLIGATORIAMENTE:
   - Personajes (sujetos): {subjects_str}
   - Acciones: {actions_str}
   - Lugares: {places_str}
   - ACCIONES (estricto): No aceptes sinónimos ni descripciones de la acción. La historia debe reflejar la misma acción pedida con su verbo (conjugaciones, gerundio, infinitivo, etc.). 
   Ejemplo: si la acción es "TRABAJAR", frases como "atender pacientes" o "hacer la oficina" son INCORRECTAS para cumplir la acción; debe decir explícitamente "trabaja", "trabajó", "trabajando", etc. 
   Marca "correct": false y da un tip si solo usan equivalentes descriptivos en lugar del verbo de la acción pedida.
5. CIERRE/CONCLUSIÓN: La historia no puede quedar a medias; debe tener un sentido de finalidad. Se aceptan finales cerrados (ej: "y se durmió")
 o finales de suspenso/abiertos (ej: "¡y de repente algo se movió en la oscuridad!"), siempre que la oración sea gramaticalmente completa.
 También cuenta como cierre VÁLIDO una subordinada de propósito con "para que" que exprese claramente para qué o con qué intención ocurre la acción, aunque no narre un desenlace adicional: eso ya cierra la intención de la historia. No marques "incorrecto" solo porque esperabas otra frase de cierre si ya hay una oración completa con "para que" bien formada.
6. Las tildes no son obligatorias, así que no hagas corrección de ellas.
6b. MAYÚSCULAS: No corrijas ni des tips sobre mayúsculas. 
7. IDIOMA: Si la historia del niño está mayormente en un idioma que NO sea español, NO evalúes gramática, elementos ni el resto de criterios. Responde con "correct": false, "parts" vacíos (subjects, actions, places como listas vacías) y en "tips" pon ÚNICAMENTE un solo consejo, exactamente este texto y ningún otro: "Recuerda que debe ser en español para que todos podemos entenderlo". No añadas más tips ni correcciones en ese caso.

INSTRUCCIONES PARA LOS "TIPS":

- Si aplica el criterio 7 (otro idioma), ignora el resto de estas instrucciones para tips: solo el único mensaje indicado allí.
- Sé breve y muy amable.
- Si hay un error (que no sea solo mayúsculas; ver 6b), puedes usar el formato: "Dijiste '[error]', pero quedaría mejor así: '[corrección]'".
- Si FALTA un elemento ya sea sujeto, acción o lugar, has un tip para indicarle que falta.
- Si la historia es perfecta, usa el primer tip para felicitar un punto específico (ej: "¡Me encantó cómo usaste el conector 'porque'!") y deja el resto del array vacío.

ANÁLISIS PARA "parts":
- "subjects": fragmentos EXACTOS del texto del niño que correspondan a los sujetos dados.
- "actions": fragmentos EXACTOS del texto del niño que correspondan a las acciones dadas.
- "places": fragmentos EXACTOS del texto del niño que correspondan a los lugares dados.
- Palabra igual o parecida para ACCIÓN y para LUGAR (ej. "la maestra cocina" = acción cocinar; "en la cocina" = lugar la cocina): en "actions" va lo que expresa la acción del tema (el verbo o fragmento que narra qué hace el personaje); en "places" va el espacio o sitio (ej. "la cocina", "en la cocina"). No confundas la acción con el lugar. Copia cada fragmento EXACTAMENTE como en el texto del niño y en "places" prefiere la frase completa del sitio cuando aparezca (ej. "la cocina") para no repetir la misma cadena que pusiste en "actions".

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
