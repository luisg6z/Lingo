# Funcionamiento y Bases del Juego "Absurdos Visuales"

Este documento detalla la arquitectura, bases conceptuales y el funcionamiento técnico del módulo del juego de **Absurdos Visuales** (Reconocimiento de voz) definido en `src/features/absurdos-visuales/game.py`.

El objetivo principal de este juego es mostrar imágenes que contienen una incongruencia lógica o "absurdo". El usuario (generalmente un niño) debe interactuar con el sistema tocando un botón físico-virtual (reconocido por un sensor Kinect) y describir en voz alta qué es lo que está mal en la imagen. El sistema evalúa semánticamente lo que dijo para decidir si acertó o no.

---

## 1. Arquitectura y Flujo General del Sistema

El juego está diseñado para proyectarse a través de un videobeam y se basa en un flujo iterativo continuo (`while True`) hasta que el usuario decide salir al menú principal.

### Flujo de Ejecución:
1. **Configuración y Carga de Datos:**
   - Se inicializan variables de interfaz y se establece una ruta JSON (`absurdos.json`) de donde se obtiene la lista de imágenes "absurdas".
   - Se selecciona al azar un absurdo que no haya sido mostrado previamente en la misma sesión de juego.
   - Se implementa un sistema flexible de coincidencia de nombres de archivos para cargar la imagen con tolerancia a errores (Maneja variaciones como guiones, guiones bajos o diferencias en mayúsculas/minúsculas empleando algoritmos de distancia de strings y similitud de Jaccard).

2. **Renderizado de la Interfaz Interactiva:**
   - Para la parte visual se emplea **OpenCV** y componentes locales. Se aplica un fondo con degradado dinámico y se instancia una imagen del "absurdo" escalada de manera responsiva al viewport.
   - Se dibuja un botón interactivo llamado **"Hablar"**. 

3. **Detección de Toques (Interacción con Kinect):**
   - El sistema abre los flujos de la cámara (RGB y profundidad `depth_stream`).
   - Mediante operaciones morfológicas, contornos en *OpenCV* e intersecciones matemáticas entre el ROI de profundidad y los límites pre-calibrados, identifica si la mano del usuario "tocó" el botón virtual de "Hablar".
   - Una vez presionado, se interrumpe momentáneamente la cámara y el sistema indica al usuario que está "Escuchando...".

4. **Captura y Transcripción de Audio:**
   - Empleando `pyttsx3`, la primera vez que se ejecuta se le da al usuario una indicación por TTS: *"Ahora habla, di qué está mal"*.
   - A través de la librería `speech_recognition`, el programa enciende el micrófono (con ajuste automatizado para el ruido ambiente) y aguarda el dictado.
   - Se usa la API de **Google Speech Recognition** para transcribir el audio en tiempo real al idioma Español (`es-ES`).

---

## 2. Validación Semántica mediante Inteligencia Artificial (Sentence Transformers)

El pilar principal que vuelve "inteligente" a este juego es la forma en la que se valida si la vocalización del niño fue correcta o no. Como el lenguaje natural es ambiguo y existen diferentes maneras correctas de describir un mismo error (ej. en vez de decir "el perro vuela", el niño dice "hay un perro en el cielo volando"), una comparación exacta (palabra por palabra) fracasaría enormemente.

Para resolver esto se emplean los **Sentence Transformers**.

### 2.1. Carga del Modelo Pre-entrenado
El sistema carga el modelo multilingüe:
```python
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```
**MiniLM-L12-v2** es una red neuronal optimizada que genera *embeddings* numéricos de texto, capturando eficientemente el significado semántico profundo e ignorando las diferencias gramaticales superficiales. Este modelo en concreto posee un perfil Multilingüe lo cual le permite entender intrínsecamente el español. 

### 2.2. Procedimiento de Evaluación
1. **Obtención de Respuestas Correctas:**
   Se lee del archivo `respuestas_correctas.json` un conjunto de respuestas modelo válidas predefinidas para esa imagen específica.

2. **Generación de Embeddings:**
   Se codifica el texto transcrito dicho por el usuario y se transforma a un vector (`texto_usuario_embedding`). Posteriormente, el sistema procesa el texto de las respuestas base y crea sus propios vectores densos (`respuestas_embeddings`).

3. **Cálculo de Similitud Coseno:**
   Para conocer la "distancia semántica" entre la frase capturada y las potenciales respuestas correctas, se calcula iterativamente la **Similitud Coseno** de sus matrices empleando funciones nativas de NumPy (`numpy.dot` y `np.linalg.norm`).
   $$ \text{Similitud} = \frac{A \cdot B}{||A|| \cdot ||B||} $$

4. **Umbral de Aceptación (Threshold):**
   ```python
   SIMILARITY_THRESHOLD = 0.7
   ```
   Se extrae el valor máximo comparativo (`max_similarity`). Si la similitud calculada entre lo que dictó el usuario y la variante de respuesta correcta más parecida alcanza o supera el umbral pre-fijado de `0.7` (o 70% de similitud de significado), la frase dictada se dictamina como válida.

---

## 3. Retroalimentación o Feedback Final

Una vez realizada la inferencia, de forma ininterrumpida se da feedback audiovisual sobre del éxito o fracaso del intento:
- **Sonoros:** Mediante `pygame.mixer` se emite un audio indicando éxito (`correct.mp3`) o un zumbido de nuevo intento (`incorrect.mp3`).
- **Visuales Dinámicos:** Una banda de color parpadea en el borde inferior. Verde si es correcto, rojo si la inferencia falla, naranja si ocurrió un error en el audio/red.
- **Pantalla de Celebración:** Si el dictamen de las inteligencias artificiales marca un acierto exitoso, el juego quita los streams, lanza chispas generadas particuladamente por el componente paramétrico `ConfettiSystem`, muestra una mascota visual estática (`LingoBien.png`) felicitando la respuesta acertada, y espera que el usuario presione el botón "Siguiente" o "Salir" por los métodos hápticos de la cámara para rotear de nuevo al bucle mientras limpia la memoria.
