# Lingo

**Lingo** es un entorno interactivo y entorno de realidad aumentada espacial diseñado para fomentar habilidades cognitivas, motrices y sociales en niños de preescolar mediante juegos colaborativos y educativos. Este entorno combina hardware como el sensor Kinect y un videobeam con un software avanzado que permite la detección, proyección y manipulación de elementos interactivos sobre una superficie física.

---

## 🚀 Características Principales

- **Interacción Natural**: Detección de toques sobre cualquier superficie sin necesidad de pantallas táctiles costosas.
- **Realidad Aumentada Espacial**: Proyección de juegos y actividades directamente sobre la mesa de trabajo o suelo.
- **Multijugador y Colaborativo**: Permite múltiples toques simultáneos para que varios niños jueguen a la vez.
- **Independiente de la Iluminación y Color**: Utiliza mapas de profundidad para detectar interacciones, ignorando los colores de la superficie y tolerando variaciones de luz.

---

## ⚙️ Funcionamiento del Sistema

El sistema opera combinando visión por computadora (RGB-D) y proyección interactiva estructurada en estas fases:

1. **Detección de Toques mediante Profundidad**: El sensor Kinect captura continuamente mapas de profundidad. Cuando una mano o un objeto toca la superficie, la profundidad disminuye en esa región. El sistema filtra variaciones o ruido y compara la lectura actual con un mapa base (superficie vacía) para registrar el toque.
2. **Transformación de Coordenadas (Homografía)**: Los toques detectados en la cámara (Kinect) se mapean exactamente al espacio de proyección (Videobeam) usando una matriz de homografía, corrigiendo así la perspectiva y la distorsión geométrica.
3. **Interacción con Interfaz (Cards)**: El toque ya transformado a coordenadas de proyección se evalúa contra las áreas activas (ej. las "Cards" del menú principal). Si colisiona, se emiten eventos o animaciones (como la elevación de la tarjeta y transición al juego).

---

## 📐 Calibración del Entorno

Para que la proyección y el reconocimiento táctil coincidan de manera precisa, es necesario calibrar el sistema en el ambiente inicial (`calibrate_area_mejorado.py`).

### Pasos de la Calibración:
1. **Marcadores de Referencia**: El videobeam proyecta cuadrados blancos en las esquinas de la zona interactiva.
2. **Detección**: La cámara RGB del Kinect detecta estos cuadrados.
3. **Mapeo**: Se calcula la matriz de homografía entre lo que ve el Kinect y lo que proyecta el videobeam.
4. **Mapa Base (dmax_map)**: El sensor recolecta 500 frames del mapa de profundidad de la mesa vacía. Esto establece la "coraza" neutra para luego poder detectar cuándo algo (una mano) interrumpe esta profundidad.
5. **Guardado**: La configuración se guarda en la carpeta `config/` para usarse al iniciar los juegos.

---

## 🛠️ Instalación y Configuración

### 1. Hardware Necesario
- Computadora con Windows (o sistema compatible).
- Sensor Kinect (y adaptador de corriente/USB).
- Videobeam (Proyector).

### 2. Instalación de Kinect SDK para Windows
1. Descarga el SDK de Kinect oficial de Microsoft ([Kinect SDK](https://www.microsoft.com/en-us/download/details.aspx?id=40278)).
2. Instala el SDK y los drivers.
3. Conecta el sensor Kinect y verifica su funcionamiento con **Kinect Studio** (la cámara RGB y de profundidad deben responder al entorno).

### 3. Instalación de Software Relacionado
Asegúrate de tener **uv** y **Python** instalados. Clona el repositorio e instala las dependencias:

```bash
git clone https://github.com/luisg6z/Lingo.git
cd Lingo
# Instalar dependencias utilizando uv
uv sync
```

### 4. Configuración Física
- Configura el **videobeam** apuntando hacia un área de trabajo despejada (como una mesa blanca o de color uniforme).
- Monta el **Kinect** de manera fija sobre la superficie (aprox a 2.4 metros de altura o distancias recomendadas por lente) asegurando una vista clara y casi perpendicular al área de juego.

---

## ▶️ Ejecución de Lingo

1. **Calibrar el área (Primero o al mover equipos)**:
   ```bash
   uv run calibrate_area_mejorado.py
   ```
   *Asegúrate de no obstruir la superficie mientras recopila los 500 frames de profundidad base.*

2. **Ejecutar Menú y Juegos**:
   ```bash
   uv run main.py
   ```
3. El videobeam mostrará la interfaz interactiva. Puedes interactuar tocando las tarjetas de los juegos proyectados sobre la mesa.

---

## 💻 Tecnologías Utilizadas

- **Python**: Lenguaje principal de desarrollo.
- **OpenCV**: Visión por computadora, filtros de imagen, detección de contornos, homografía.
- **OpenNI2**: SDK subyacente interactuando con las cámaras y profundidad del Kinect.
- **NumPy**: Operaciones matriciales rápidas para la manipulación de imágenes y cálculos de centroides.

---

## 🤝 Contribuciones
Si deseas aportar, arreglar un bug o añadir un nuevo juego educativo, eres libre de proponer un **Pull Request** o contactar al equipo desarrollador.

---

**¡Disfruta construyendo una nueva forma de jugar con Lingo!** 🎮✨
