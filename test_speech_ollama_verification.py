"""
Script de prueba para reconocimiento de voz y verificación de texto usando SpeechRecognition y Ollama.

Este script permite:
1. Grabar audio del micrófono con capacidad de detener la grabación
2. Convertir el audio a texto usando SpeechRecognition
3. Verificar si la oración tiene sentido o está bien escrita usando Ollama (llama3)

Requisitos:
- SpeechRecognition: pip install SpeechRecognition
- pyaudio: pip install pyaudio (para acceso al micrófono)
- ollama: pip install ollama
- Ollama debe estar instalado y corriendo en el sistema
- El modelo llama3 debe estar disponible en Ollama (ejecutar: ollama pull llama3)

Uso:
1. Ejecuta el script: python test_speech_ollama_verification.py
2. Selecciona una opción de grabación
3. Habla en español
4. El script convertirá tu voz a texto y lo verificará con Ollama

Notas:
- El reconocimiento de voz requiere conexión a internet (usa Google Speech Recognition)
- Para detener la grabación manualmente, presiona Ctrl+C
- La grabación se detiene automáticamente después de 2 segundos de silencio
"""

import json
import re
import speech_recognition as sr
import threading
import time
import sys

# Intentar importar ollama
try:
    import ollama
except ImportError:
    print("Error: ollama no está instalado. Instálalo con: pip install ollama")
    print("Asegúrate de que Ollama esté instalado y corriendo en tu sistema.")
    sys.exit(1)


class SpeechRecorder:
    """Clase para manejar la grabación de audio con capacidad de detener"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Evitar que corte frases: más silencio antes de parar (default 0.8 es corto)
        self.recognizer.pause_threshold = 1.3
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = None
        self.is_recording = False
        self.audio_data = None
        self.recording_thread = None
        
    def initialize_microphone(self):
        """Inicializa el micrófono"""
        try:
            self.microphone = sr.Microphone()
            print("Micrófono inicializado correctamente.")
            return True
        except Exception as e:
            print(f"Error al acceder al micrófono: {e}")
            return False
    
    def adjust_for_ambient_noise(self, duration=1):
        """Ajusta el reconocedor para el ruido ambiente"""
        if not self.microphone:
            print("Error: El micrófono no está inicializado.")
            return False
        
        try:
            print("Ajustando ruido ambiente... Por favor, mantén silencio por un momento.")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            print("Ruido ambiente ajustado.")
            return True
        except Exception as e:
            print(f"Error al ajustar el ruido ambiente: {e}")
            return False
    
    def start_listening(self, timeout=None, phrase_time_limit=None):
        """
        Inicia la grabación de audio en un hilo separado.
        
        Args:
            timeout: Tiempo máximo de espera para comenzar a escuchar (None = sin límite)
            phrase_time_limit: Tiempo máximo de grabación (None = sin límite)
        """
        if not self.microphone:
            print("Error: El micrófono no está inicializado.")
            return False
        
        if self.is_recording:
            print("Ya se está grabando audio.")
            return False
        
        self.is_recording = True
        self.audio_data = None
        
        def record_audio():
            try:
                with self.microphone as source:
                    result_holder = [None]
                    exception_holder = [None]

                    def do_listen():
                        try:
                            if phrase_time_limit:
                                a = self.recognizer.listen(
                                    source,
                                    timeout=timeout,
                                    phrase_time_limit=phrase_time_limit
                                )
                            else:
                                a = self.recognizer.listen(source, timeout=timeout)
                            result_holder[0] = a
                        except sr.WaitTimeoutError:
                            pass
                        except Exception as e:
                            exception_holder[0] = e

                    listen_thread = threading.Thread(target=do_listen, daemon=True)
                    listen_thread.start()
                    time.sleep(0.5)  # Dar tiempo a que la captura esté activa
                    print("Ahora puedes hablar.")
                    listen_thread.join()
                    if exception_holder[0]:
                        raise exception_holder[0]
                    self.audio_data = result_holder[0]
                    print("Grabación finalizada.")
            except sr.WaitTimeoutError:
                print("Tiempo de espera agotado. No se detectó ningún audio.")
                self.audio_data = None
            except Exception as e:
                print(f"Error durante la grabación: {e}")
                self.audio_data = None
            finally:
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=record_audio, daemon=True)
        self.recording_thread.start()
        return True
    
    def stop_listening(self):
        """Detiene la grabación de audio"""
        if not self.is_recording:
            print("No hay ninguna grabación en curso.")
            return False
        
        # Nota: speech_recognition no tiene un método directo para detener
        # La grabación se detendrá cuando termine el phrase_time_limit o timeout
        # Para una mejor implementación, podríamos usar pyaudio directamente
        print("Solicitando detener la grabación...")
        self.is_recording = False
        return True
    
    def wait_for_recording(self):
        """Espera a que termine la grabación"""
        if self.recording_thread:
            self.recording_thread.join()
    
    def recognize_speech(self, language="es-ES"):
        """
        Convierte el audio grabado a texto.
        
        Args:
            language: Idioma para el reconocimiento (default: es-ES para español)
        
        Returns:
            str: Texto reconocido o None si hay error
        """
        if self.audio_data is None:
            print("No hay audio grabado para procesar.")
            return None
        
        try:
            print("Procesando audio...")
            texto = self.recognizer.recognize_google(self.audio_data, language=language)
            print(f"Texto reconocido: {texto}")
            return texto
        except sr.UnknownValueError:
            print("No se pudo entender el audio. Intenta hablar más claro.")
            return None
        except sr.RequestError as e:
            print(f"Error al conectar con el servicio de reconocimiento: {e}")
            print("Verifica tu conexión a internet.")
            return None
        except Exception as e:
            print(f"Error inesperado durante el reconocimiento: {e}")
            return None


class OllamaVerifier:
    """Clase para verificar texto/historias usando Ollama"""

    def __init__(self, model="deepseek-r1"):
        self.model = model
        self.base_url = "http://localhost:11434"  # URL por defecto de Ollama

    def _build_story_prompt(self, sentence, subjects_str, actions_str, places_str):
        return f"""Eres un profesor de español para niños de 7 años. Tu objetivo es evaluar historias cortas de forma alentadora y flexible.

CRITERIOS DE EVALUACIÓN:
1. ELEMENTOS: Debe incluir (o usar variaciones de): 
   - Personajes: {subjects_str}
   - Acciones: {actions_str}
   - Lugares: {places_str}
2. ENLACES: Debe usar al menos un conector (ej: "y", "entonces", "luego", "después", "porque").
3. CIERRE/CONCLUSIÓN: La historia debe tener un final. No tiene que ser un "Fin" formal; cuenta como conclusión cualquier resultado o consecuencia de las acciones (ej: "se quedaron dormidos", "ganaron el juego", "se pusieron felices" o "se fueron a casa").

REGLAS DE ORO:
- Sé muy flexible con la gramática y tiempos verbales (ej: acepta "corrió" por "correr").
- Si falta algo, da un consejo breve, dulce y específico para completar la historia.
- No des lecciones de ortografía, enfócate en la narrativa.

Historia del niño: "{sentence}"

Responde ÚNICAMENTE con un JSON válido en una línea. Formato exacto:
{{"correct": true o false, "tips": ["consejo1", "consejo2"]}}
Los consejos deben ser dirigidos al niño, no a un profesor."""

    def verify_story(self, sentence, subjects=None, actions=None, places=None):
        """
        Verifica una historia corta con los criterios de personajes, acciones y lugares.
        Recibe los mismos parámetros del prompt (subjects_str, actions_str, places_str).

        Args:
            sentence: Historia u oración del niño.
            subjects: Lista de personajes (ej: ["Niño", "Niña"]).
            actions: Lista de acciones (ej: ["Dar", "Jugar"]).
            places: Lista de lugares (ej: ["Casa", "Parque"]).

        Returns:
            dict: {"correct": bool, "tips": list} o str con mensaje de error.
        """
        if not sentence or not sentence.strip():
            return {"correct": False, "tips": ["No se proporcionó historia."]}
        subjects_str = ", ".join(subjects) if subjects else "(ninguno)"
        actions_str = ", ".join(actions) if actions else "(ninguna)"
        places_str = ", ".join(places) if places else "(ninguno)"
        prompt = self._build_story_prompt(sentence, subjects_str, actions_str, places_str)
        try:
            print(f"Enviando historia a Ollama (modelo: {self.model})...")
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            content = (response or {}).get("message") or {}
            content = (content.get("content") or "").strip()
            if not content:
                return "Error: No se recibió respuesta de Ollama."
            content = re.sub(r"^```\w*\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
            content = content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            data = json.loads(content)
            return {"correct": data.get("correct", False), "tips": data.get("tips") or []}
        except json.JSONDecodeError:
            return {"correct": False, "tips": ["No se pudo interpretar la respuesta del modelo."]}
        except Exception as e:
            return f"Error al comunicarse con Ollama: {e}\nAsegúrate de que Ollama esté instalado y corriendo."

    def verify_text(self, text, subjects=None, actions=None, places=None):
        """
        Verifica texto: si se pasan subjects/actions/places usa el prompt de historia;
        si no, usa verificación genérica (compatibilidad).

        Args:
            text: Texto u historia a verificar.
            subjects: Opcional. Lista de personajes para evaluación de historia.
            actions: Opcional. Lista de acciones.
            places: Opcional. Lista de lugares.

        Returns:
            str o dict: Si se usan subjects/actions/places devuelve dict con correct/tips;
            si no, devuelve str con la respuesta del modelo (comportamiento anterior).
        """
        if subjects is not None or actions is not None or places is not None:
            result = self.verify_story(text, subjects=subjects or [], actions=actions or [], places=places or [])
            if isinstance(result, dict):
                tips = result.get("tips", [])
                correct = result.get("correct", False)
                return f"Correcto: {correct}\nConsejos: " + (", ".join(tips) if tips else "Ninguno")
            return str(result)
        # Comportamiento anterior: prompt genérico
        if not text:
            return "Error: No se proporcionó texto para verificar."
        prompt = f"""Eres un experto en gramática y redacción en español, 
        con experiencia en pedagogía y educación. 
Analiza la siguiente oración u oraciones y verifica:
1. Si tiene sentido semántico
2. Si está bien escrita (gramática, ortografía, sintaxis)
3. Si hay errores, indícalos

Oración a verificar: "{text}"

Proporciona una respuesta clara y concisa en español indicando:
- Si la oración tiene sentido
- Si está bien escrita
- Si hay errores, cuáles son y cómo corregirlos"""
        try:
            print(f"Enviando texto a Ollama (modelo: {self.model})...")
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            if response and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            return "Error: No se recibió una respuesta válida de Ollama."
        except Exception as e:
            return f"Error al comunicarse con Ollama: {e}\nAsegúrate de que Ollama esté instalado y corriendo."


def main():
    """Función principal del script"""
    print("=" * 60)
    print("Sistema de Verificación de Texto por Voz")
    print("=" * 60)
    print()
    
    # Inicializar el grabador de audio
    recorder = SpeechRecorder()
    
    if not recorder.initialize_microphone():
        print("No se pudo inicializar el micrófono. Saliendo...")
        return
    
    # Ajustar para el ruido ambiente
    if not recorder.adjust_for_ambient_noise(duration=1):
        print("Advertencia: No se pudo ajustar el ruido ambiente.")
    
    # Inicializar el verificador de Ollama
    verifier = OllamaVerifier(model="deepseek-r1")

    def _get_story_params():
        """Opcional: pedir personajes, acciones y lugares para el prompt de historia."""
        print("Parámetros para evaluación de historia (vacío = usar valores por defecto):")
        s = input("  Personajes (ej: Niño, Niña): ").strip()
        subjects = [x.strip() for x in s.split(",") if x.strip()] if s else None
        a = input("  Acciones (ej: Dar, Jugar): ").strip()
        actions = [x.strip() for x in a.split(",") if x.strip()] if a else None
        p = input("  Lugares (ej: Casa, Parque): ").strip()
        places = [x.strip() for x in p.split(",") if x.strip()] if p else None
        if not subjects and not actions and not places:
            subjects, actions, places = ["Niño", "Niña"], ["Dar"], ["Casa"]
            print(f"  Usando por defecto: personajes={subjects}, acciones={actions}, lugares={places}")
        return subjects, actions, places

    while True:
        print("\n" + "-" * 60)
        print("Opciones:")
        print("1. Grabar audio (con límite de tiempo específico)")
        print("2. Grabar audio (se detiene automáticamente después de silencio)")
        print("3. Salir")
        print("-" * 60)
        print("Nota: Para detener manualmente durante la grabación, presiona Ctrl+C")
        print("-" * 60)
        
        choice = input("Selecciona una opción (1-3): ").strip()
        
        if choice == "1":
            # Grabación con límite de tiempo
            try:
                duration = float(input("Ingresa la duración de la grabación en segundos (ej: 5): "))
            except ValueError:
                print("Duración inválida. Usando 5 segundos por defecto.")
                duration = 5
            
            recorder.start_listening(timeout=10, phrase_time_limit=duration)
            recorder.wait_for_recording()
            
            # Convertir audio a texto
            texto = recorder.recognize_speech(language="es-ES")
            
            if texto:
                subjects, actions, places = _get_story_params()
                print("\n" + "=" * 60)
                print("Verificación de historia con Ollama:")
                print("=" * 60)
                result = verifier.verify_story(texto, subjects=subjects, actions=actions, places=places)
                if isinstance(result, dict):
                    print(f"Correcto: {result.get('correct')}")
                    print("Consejos:", result.get("tips", []))
                else:
                    print(result)
                print("=" * 60)

        elif choice == "2":
            # Grabación continua - el usuario puede hablar y la grabación se detiene automáticamente
            print("\nIniciando grabación...")
            print("Verás 'Ahora puedes hablar' cuando el micrófono esté listo. La grabación se detendrá tras unos segundos de silencio.")
            print("(Máximo 30 segundos de grabación)")
            
            # Iniciar grabación con límite de tiempo
            # phrase_time_limit controla cuánto tiempo puede durar una frase
            # El reconocedor se detendrá automáticamente después de silencio
            recorder.start_listening(timeout=10, phrase_time_limit=30)
            recorder.wait_for_recording()
            
            # Convertir audio a texto
            texto = recorder.recognize_speech(language="es-ES")
            
            if texto:
                subjects, actions, places = _get_story_params()
                print("\n" + "=" * 60)
                print("Verificación de historia con Ollama:")
                print("=" * 60)
                result = verifier.verify_story(texto, subjects=subjects, actions=actions, places=places)
                if isinstance(result, dict):
                    print(f"Correcto: {result.get('correct')}")
                    print("Consejos:", result.get("tips", []))
                else:
                    print(result)
                print("=" * 60)
            else:
                print("No se pudo reconocer el audio. Intenta de nuevo.")

        elif choice == "3":
            print("Saliendo...")
            break
        
        else:
            print("Opción inválida. Por favor, selecciona 1, 2 o 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()

