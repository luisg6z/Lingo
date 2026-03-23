# Funcionamiento y Lógica Estructural del "Juego de Historia"

Este documento detalla la arquitectura algorítmica y el funcionamiento del motor de reconocimiento y validación semántica del **Juego de Historia**, definido en `src/features/juego-historia/story_voice.py`.

El objetivo principal de este módulo de backend es permitir que el usuario (generalmente un niño de edad preescolar/escolar) construya y narre una historia de forma verbal basándose en personajes (sujetos), acciones y lugares seleccionados previamente mediante cartas o imágenes en la pantalla. El sistema capta la voz, la transcribe, y emplea Inteligencia Artificial generativa para evaluar la coherencia estructural y los elementos presentes de la historia en tiempo real.

---

## 1. Captura y Transcripción de Audio en Tiempo Real

Para la interacción verbal fluida, el sistema invoca de la función asíncrona `listen_and_transcribe`, la cual está diseñada para lidiar con el entorno de aula/terapia evadiendo congelamientos en el hilo principal de la interfaz gráfica.

### 1.1. Manejo del Micrófono y Ruido Ambiente
- Se emplea la librería `speech_recognition` conectada a un dispositivo de entrada de señal de hardware (`device_index=3`).
- **Calibración Dinámica:** Antes de escuchar el dictado directo, el sistema capta de forma encubierta un fragmento de `0.8` segundos para configurar automáticamente el umbral de energía algorítmico (`dynamic_energy_threshold`) al nivel de ruido ambiente del salón.
- **Tolerancia Extrema de Pausas:** Sabiendo que los niños suelen titubear mientras imaginan una narración, se previene que la grabación se interrumpa abruptamente aumentando el `pause_threshold` a `1.3` segundos.

### 1.2. Hilo Paralelo (Multithreading)
- La captura pesada de hardware en el sistema operativo se delega a un hilo daemon paralelo (`threading.Thread`). Esto es crucial para la fluidez, ya que le permite a la UI pintar un mensaje (ej. *"Ahora puedes hablar"*) en pantalla únicamente usando la señal de evento asíncrono `on_listening_started`, en el momento exacto donde la escucha ha arrancado en el buffer interno.
- La traducción digital en tiempo real de estas señales de onda a texto (Speech-to-Text) se concreta inyectando el audio base en nube empleando la API de **Google Speech Recognition** (`recognize_google`) con perfilación al dialecto Español (`es-ES`).

---

## 2. Preparación Semántica de Diccionarios Base

Para evitar que el usuario realice una historia inventada ajena a la dinámica de trabajo mostrada en el tablero, el sistema extrae las palabras necesarias de inclusión inquebrantable empleando el método estructurado `get_required_words`.

- Hace recolección de configuraciones a través del mapa matricial `palabras_imagenes.json`.
- Cruza localmente las categorías algorítmicas de `sujetos_seleccionados`, `acciones_seleccionadas` y `lugares_seleccionados` para aplanar variaciones lingüísticas. 
- *A prueba de fallos:* Si por ejemplo se seleccionó al personaje "Niño", el sistema compila este sustantivo en minúsculas y expande arreglos como `("niño", "nino")`. Así, incluso si Google devuelve la palabra sin ñ o con ambigüedades numéricas, el sistema posee resiliencia evaluativa semántica de fondo.

---

## 3. Validación Inteligente de la Historia Computada (LLM / Ollama)

El núcleo que vuelve "mágica" la dinámica de juego es confiado a la función `verify_story_ollama`. Aquí, se aplica el análisis semántico y gramatical del texto a través de un **Modelo de Lenguaje Grande (LLM)** desplegado localmente o de forma orquestada a través del entorno **Ollama**.

### 3.1. Cliente de Inteligencia de Modelos
- En el backend se instancia el túnel de Ollama inyectando credenciales por variables de entorno `OLLAMA_HOST` y `OLLAMA_API_KEY` por medio del gestor seguro `dotenv`.
- Tras la validación, envía las iteraciones léxicas haciendo un requerimiento con data stream al modelo seleccionado (Por defecto se plantea el uso de motores analíticos profundos como `deepseek-v3`).

### 3.2 Prompting Pedagógico y Criterios Neurales
El código fuente inyecta un *Prompt Textual Cero-Shot* altamente especializado, obligando al LLM a parametrizar a través de un rol o *System-prompt* definido como: **"Profesor de español para niños de 7 años"**. Tras inyectar y sanitizar la historia del niño, la red neuronal se enfrenta a 5 criterios de calificación férreos:
1. **Sentido Gramatical:** Aceptación mínima de la idea formulada (cohesión).
2. **Tiempos Verbales:** Lógica continua del presente, pasado o futuro.
3. **Pivotes de Enlace:** Constatación del uso de conectores explícitos (*y, entonces, luego, porque*).
4. **Elementos Obligatorios:** Verificación referencial cruzada de los sujetos, acciones y lugares (evaluado contra la matriz extraída de paso previo).
5. **Estructura Cíclica (Cierre):** Exigencia de un final claro en todo texto provisto.

### 3.3. Estructuración JSON y Feedback Dinámico (Resiliencia)
Para poder extraer del "texto natural" del LLM información consumible por la UI estructurada, el Prompt de Ollama acorrala al modelo a responder estrictamente en formato de sintaxis cruzada **JSON** de bajo nivel.
Una vez obtenidos los tokens (`stream`), el algoritmo ejecuta expresiones regulares y sustracción (`json.loads`) para obtener:
- **`correct` (Booleano):** True, si se cumplen impecablemente todos los parámetros del cuento.
- **`tips` (Array de strings):** Alimenta directamente el "feedback" del juego al niño siendo ultra asertivo. Si el niño cometió fallas, dice de forma amable cómo corregirlo *"Dijiste esto, pero quedaría mejor así..."*. Si dictaminó True (Perfecto) elogia específicamente una faceta positiva (ej. *"Me encantó cómo usaste la palabra donde"*).
- **`parts` (Array de objetos):** Devuelve las coincidencias analíticas en pedazos (`subjects`, `actions`, `places`) que el LLM localizó explícitamente y que ayudan al código fuente a hacer debug o subrayar la oración a futuro.

Para garantizar robustez plena en la interacción táctil en tiempo real, `default_fail` se encarga de retornar de forma encubierta un fallo sintético manejable si el LLM colapsa, el audio deforma la matriz base o se apaga el servidor externo; respondiendo diplomáticamente *"Revisa tu oración"*.
