# Explicación del Sistema de Proyección e Interacción Táctil

## Descripción General
Este documento explica cómo funciona el sistema de proyección interactiva que utiliza un sensor Kinect para detectar toques sobre una superficie y proyectar contenido visual (cards de juegos) que responden a la interacción del usuario.

---

## 1. FASE DE CALIBRACIÓN (`calibrate_area_mejorado.py`)

### 1.1 Objetivo
La calibración establece la correspondencia entre:
- **Coordenadas de la cámara** (640x480): donde ve el Kinect
- **Coordenadas de proyección** (1280x800): donde se proyecta el videobeam

### 1.2 Proceso de Calibración

#### Paso 1: Proyección de Marcadores de Referencia
```python
def proyectar_cuadrados(view_width, view_height):
```
- Se proyectan **dos cuadrados blancos** en la superficie:
  - Uno en la esquina inferior izquierda
  - Otro en la esquina superior derecha
- Estos cuadrados sirven como puntos de referencia conocidos

#### Paso 2: Detección de los Marcadores
```python
def detectar_cuadrados(frame):
```
- La cámara RGB del Kinect captura la superficie
- Se detectan los cuadrados blancos usando:
  - Umbralización (threshold) para aislar objetos blancos
  - Detección de contornos
  - Filtrado por área y forma (solo cuadrados)

#### Paso 3: Cálculo de la Matriz de Homografía
```python
homography_matrix = cv2.getPerspectiveTransform(pts_camara, pts_proyeccion)
```
- **Puntos de la cámara**: coordenadas donde se detectaron los cuadrados en la imagen RGB
- **Puntos de proyección**: coordenadas donde se proyectaron los cuadrados
- La **matriz de homografía** permite transformar cualquier punto de la cámara al espacio de proyección

**¿Por qué homografía?**
- Corrige la perspectiva (la mesa está en ángulo, no perpendicular)
- Permite mapeo preciso punto a punto
- Es más preciso que un escalado lineal simple

#### Paso 4: Definición del Área de Trabajo (ROI - Region of Interest)
```python
xw_min, xw_max_escalado = # coordenadas en la cámara
yw_min_escalado, yw_max = # coordenadas en la cámara
xv_min, xv_max = # coordenadas en la proyección
yv_min, yv_max = # coordenadas en la proyección
```
- Se define una región rectangular donde funcionará la interacción
- Se aplica un factor de escala (1.16x y 1.12y) para compensar distorsiones

#### Paso 5: Cálculo del Mapa de Profundidad (dmax_map)
```python
def calculate_dmax(device, calibrated_area, ...):
```
- Se capturan **500 frames** de la cámara de profundidad
- Para cada píxel en el área calibrada, se calcula la **moda** (valor más frecuente) de profundidad
- Esto representa la profundidad de la superficie cuando **no hay toques**

**Resultado**: Un mapa 2D donde cada píxel tiene el valor de profundidad de la mesa sin toques.

#### Paso 6: Guardado de Configuración
- Se guardan en `config/ultima_configuracion_coordenadas.json`:
  - Coordenadas de la ROI (en cámara y proyección)
  - Matriz de homografía
- Se guarda `dmax_map` en `config/dmax_map.txt`

---

## 2. FASE DE DETECCIÓN DE TOQUES (`calibrate_area_mejorado.py` y `main.py`)

### 2.1 Principio de Funcionamiento
Cuando alguien toca la superficie:
1. La profundidad en ese punto **disminuye** (el objeto está más cerca de la cámara)
2. El valor de profundidad queda **entre dmin y dmax**
3. Esto crea una "coraza" de detección de toques

### 2.2 Proceso de Detección

#### Paso 1: Captura de Datos
```python
depth_frame = depth_stream.read_frame()
depth_data = np.frombuffer(...).reshape(480, 640)
depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
```
- Se lee el frame de profundidad actual
- Se extrae solo la ROI (área de trabajo)

#### Paso 2: Filtrado de Vibraciones
```python
roi_diff = cv2.absdiff(depth_roi, previous_roi)
vibration_mask = cv2.threshold(roi_diff, vibration_threshold, 255, ...)
depth_roi[vibration_mask > 0] = previous_roi[vibration_mask > 0]
```
- Se compara con el frame anterior
- Si hay cambios bruscos (vibraciones), se usa el valor anterior
- Esto elimina falsos positivos por vibraciones

#### Paso 3: Creación de la Máscara de Toques
```python
touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
```
- **dmin_map = dmax_map - 10**: Profundidad mínima para considerar un toque
- **dmax_map**: Profundidad de la superficie sin toques
- Solo los píxeles dentro de este rango se marcan como toques

#### Paso 4: Filtrado Morfológico
```python
touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
touch_mask_filtered = cv2.morphologyEx(..., cv2.MORPH_OPEN, kernel)
```
- **Filtro mediano**: Elimina ruido puntual
- **Apertura morfológica**: Elimina pequeños grupos de píxeles (ruido)

#### Paso 5: Detección de Componentes Conectados
```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_final, connectivity=8)
```
- Se identifican regiones conectadas (cada toque es una región)
- Se calcula el **centroide** de cada toque (punto central)

#### Paso 6: Centroide Mejorado (Opcional - en calibrate_area_mejorado.py)
```python
# Centroide ponderado por profundidad
depth_inverse = (dmax_map - depth_roi)  # Más cerca = mayor peso
x_touch = 0.75 * cx_weighted + 0.25 * cx_standard
```
- El centroide se calcula ponderando por profundidad
- Las áreas más cercanas (centro del toque) tienen más peso
- Esto mejora la precisión del punto de toque

---

## 3. FASE DE TRANSFORMACIÓN DE COORDENADAS

### 3.1 Mapeo de Coordenadas
```python
# Opción 1: Usando homografía (más preciso)
point_camara = np.array([[[x_camara, y_camara]]], dtype=np.float32)
point_proyeccion = cv2.perspectiveTransform(point_camara, homography_matrix)
x_viewport, y_viewport = point_proyeccion[0][0]

# Opción 2: Mapeo lineal (fallback)
sx = float(xv_max - xv_min) / depth_roi.shape[1]
sy = float(yv_max - yv_min) / depth_roi.shape[0]
x_viewport = int(xv_min + (x_touch * sx))
y_viewport = int(yv_min + (y_touch * sy))
```

**Proceso**:
1. **Coordenadas relativas en ROI**: `(x_touch, y_touch)` respecto a la ROI
2. **Coordenadas absolutas en cámara**: `x_camara = xw_min + x_touch`
3. **Coordenadas en proyección**: Se transforman usando homografía o mapeo lineal
4. **Coordenadas finales**: `(x_viewport, y_viewport)` en espacio de proyección (0-1280, 0-800)

---

## 4. FASE DE PROYECCIÓN E INTERACCIÓN CON CARDS (`main.py`)

### 4.1 Proyección del Menú Principal

#### Inicialización
```python
videobeam_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
```
- Se crea una imagen negra de 1280x800 píxeles
- Esta imagen se proyecta en la pantalla/videobeam

#### Dibujo del Fondo y Logo
```python
# Degradado de colores
for y in range(view_height):
    ratio = y / view_height
    r = int(255 * (0.3 + 0.4 * ratio))
    g = int(200 * (0.5 + 0.3 * ratio))
    b = int(255 * (0.8 - 0.3 * ratio))
    videobeam_screen[y, :] = [b, g, r]
```
- Se dibuja un degradado de colores pastel
- Se superpone el logo centrado en la parte superior

#### Dibujo de las Cards
```python
def draw_game_cards(screen, game_positions, elevated_card=None):
```

**Estructura de cada Card**:
- **Posición**: `(x, y)` en coordenadas de proyección
- **Tamaño**: 320x400 píxeles
- **Contenido**: 
  - Icono emoji
  - Nombre del juego
  - Descripción breve
  - Color distintivo

**Efecto de Elevación**:
- Cuando una card está seleccionada (`elevated_card`):
  - Se mueve 20 píxeles hacia arriba
  - Se escala al 105% del tamaño original
  - Se hace más brillante (15% más)
  - La sombra se hace más grande
  - El borde se hace más grueso

#### Mostrar en Pantalla
```python
cv2.namedWindow("Menú de Juegos", cv2.WINDOW_NORMAL)
cv2.moveWindow("Menú de Juegos", 1920, 0)  # Mover a segundo monitor
cv2.setWindowProperty("Menú de Juegos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Menú de Juegos", videobeam_screen)
```
- La ventana se coloca en el segundo monitor (posición 1920, 0)
- Se pone en pantalla completa para la proyección

---

### 4.2 Detección de Toques en Cards

#### Bucle Principal
```python
while True:
    # 1. Capturar frame de profundidad
    depth_frame = depth_stream.read_frame()
    depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
    
    # 2. Crear máscara de toques (igual que en calibración)
    touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map)
    
    # 3. Encontrar contornos
    contours, _ = cv2.findContours(touch_mask, ...)
    
    # 4. Para cada contorno
    for contour in contours:
        # Calcular centroide
        M = cv2.moments(contour)
        cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        
        # Transformar a coordenadas de proyección
        x_touch = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
        y_touch = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
        
        # Verificar si está sobre una card
        juego_seleccionado = detectar_juego_seleccionado(x_touch, y_touch, game_positions)
```

#### Función de Detección de Card
```python
def detectar_juego_seleccionado(x_touch, y_touch, game_positions):
    for nombre, pos in game_positions.items():
        x, y = pos['x'], pos['y']
        w, h = pos['width'], pos['height']
        if x <= x_touch <= x + w and y <= y_touch <= y + h:
            return nombre
    return None
```

**Lógica**:
- Se verifica si el punto `(x_touch, y_touch)` está dentro del rectángulo de alguna card
- Si está dentro, se retorna el nombre del juego

---

### 4.3 Efectos Visuales al Tocar una Card

#### Animación de Elevación
```python
if juego_seleccionado:
    # Animación de 10 frames
    for frame_num in range(10):
        # Recrear fondo
        temp_screen = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        # ... dibujar fondo y logo ...
        
        # Elevar la card seleccionada
        if frame_num >= 5:
            draw_game_cards(temp_screen, game_positions, elevated_card=juego_seleccionado)
        else:
            draw_game_cards(temp_screen, game_positions)
        
        cv2.imshow("Menú de Juegos", temp_screen)
        cv2.waitKey(30)  # 30ms entre frames = ~33 FPS
```

**Efecto**:
1. **Frames 0-4**: Cards normales
2. **Frames 5-9**: Card seleccionada se eleva gradualmente
3. La card mantiene su posición elevada por 200ms adicionales

#### Mensaje de Confirmación
```python
mostrar_mensaje_juego(juego_seleccionado)
```
- Muestra un overlay oscuro semi-transparente
- Muestra el nombre del juego seleccionado
- Permanece 2 segundos

#### Restauración del Menú
- Se vuelve a dibujar el menú completo
- Todas las cards vuelven a su estado normal
- El sistema está listo para otra selección

---

## 5. FLUJO COMPLETO DEL SISTEMA

```
1. CALIBRACIÓN (calibrate_area_mejorado.py)
   ├─ Proyectar cuadrados blancos
   ├─ Detectar cuadrados en cámara RGB
   ├─ Calcular homografía
   ├─ Calcular dmax_map (500 frames)
   └─ Guardar configuración

2. INICIALIZACIÓN (main.py)
   ├─ Cargar configuración guardada
   ├─ Cargar dmax_map
   ├─ Iniciar streams RGB y profundidad
   └─ Crear ventana de proyección

3. BUCLE DE INTERACCIÓN
   ├─ Capturar frame de profundidad
   ├─ Extraer ROI
   ├─ Detectar toques (comparar con dmax_map)
   ├─ Filtrar ruido (morfología, vibraciones)
   ├─ Calcular centroides
   ├─ Transformar coordenadas (cámara → proyección)
   ├─ Verificar si toca una card
   ├─ Si toca card:
   │  ├─ Animar elevación
   │  ├─ Mostrar mensaje
   │  └─ Ejecutar acción (iniciar juego, etc.)
   └─ Actualizar proyección
```

---

## 6. CONCEPTOS CLAVE PARA LA DEFENSA

### 6.1 ¿Por qué usar profundidad y no solo RGB?
- **Ventaja**: Funciona independientemente del color de la superficie
- **Ventaja**: No se confunde con sombras o iluminación
- **Ventaja**: Detecta cualquier objeto que se acerque a la superficie

### 6.2 ¿Por qué dmax_map en lugar de un valor fijo?
- La profundidad varía por posición en la superficie
- Un valor fijo solo funcionaría si la superficie fuera perfectamente plana
- El mapa permite calibración precisa para cada píxel

### 6.3 ¿Por qué homografía en lugar de mapeo lineal?
- **Perspectiva**: La cámara ve la superficie en ángulo
- **Distorsión**: Las líneas paralelas en la realidad no lo son en la imagen
- **Precisión**: La homografía corrige estos efectos

### 6.4 ¿Cómo se evitan falsos positivos?
- **Filtro de vibraciones**: Ignora cambios bruscos
- **Filtro morfológico**: Elimina ruido pequeño
- **Umbral de área**: Ignora toques muy pequeños
- **Historial temporal**: Requiere que el toque persista varios frames

---

## 7. TECNOLOGÍAS UTILIZADAS

- **OpenNI2**: API para acceder al sensor Kinect
- **OpenCV**: Procesamiento de imágenes y visión por computadora
- **NumPy**: Operaciones matemáticas y manejo de arrays
- **Python**: Lenguaje de programación principal

---

## 8. PUNTOS CLAVE PARA LA DEFENSA

1. **Calibración robusta**: El sistema se adapta a cualquier configuración física
2. **Detección precisa**: Usa múltiples técnicas de filtrado para evitar errores
3. **Transformación geométrica**: Homografía para mapeo preciso
4. **Feedback visual**: El usuario ve inmediatamente el efecto de sus acciones
5. **Modularidad**: La calibración es independiente del menú/juegos

---

## Preguntas Frecuentes para la Defensa

**P: ¿Qué pasa si la iluminación cambia?**
R: No afecta porque usa profundidad, no color. Solo afecta si hay cambios físicos en la superficie.

**P: ¿Funciona con múltiples toques simultáneos?**
R: Sí, cada componente conectado se detecta por separado. Puede manejar varios toques.

**P: ¿Por qué 500 frames para calcular dmax_map?**
R: Para tener una muestra estadística robusta. Con 500 frames se captura la variación normal y se calcula la moda con confianza.

**P: ¿Se puede recalibrar sin reiniciar?**
R: Actualmente no, pero se podría implementar fácilmente guardando una nueva configuración.

**P: ¿Qué precisión tiene el sistema?**
R: Depende de la calidad de la calibración y la estabilidad de la superficie. Típicamente ±5-10 píxeles en la proyección.

