# Proceso de Calibración Matemática - MagicboARd

Este documento detalla los fundamentos matemáticos y algoritmos utilizados para la calibración y detección de toques en el sistema MagicboARd.

---

## 1. Transformación de Coordenadas: Homografía

El núcleo matemático del sistema es la **Transformación de Perspectiva** o **Homografía**. Dado que la cámara Kinect y el videoproyector no están alineados en el mismo eje óptico, la imagen de la mesa vista por la cámara sufre una distorsión de perspectiva.

### 1.1 La Matriz de Homografía ($H$)
La homografía es una matriz de $3 \times 3$ que relaciona puntos entre dos planos: el plano de la imagen de la cámara $(x_c, y_c)$ y el plano de proyección $(x_p, y_p)$.

$$
\begin{bmatrix} 
x'_p \\ 
y'_p \\ 
w 
\end{bmatrix} = 
\begin{bmatrix} 
h_{11} & h_{12} & h_{13} \\ 
h_{21} & h_{22} & h_{23} \\ 
h_{31} & h_{32} & h_{33} 
\end{bmatrix} 
\begin{bmatrix} 
x_c \\ 
y_c \\ 
1 
\end{bmatrix}
$$

Donde:
- $(x_c, y_c)$ son las coordenadas en el sensor RGB (640x480).
- $(x_p, y_p)$ se obtienen como: $x_p = \frac{x'_p}{w}$ y $y_p = \frac{y'_p}{w}$.
- $H$ es la matriz con 8 grados de libertad (normalizada $h_{33} = 1$).

### 1.2 Cálculo de la Matriz
Se utiliza la función `cv2.getPerspectiveTransform(pts_camara, pts_proyeccion)` que resuelve un sistema de ecuaciones lineales si se conocen al menos **4 puntos correspondientes**. En nuestro sistema, usamos los centros de los cuadrados proyectados para definir los límites del área de trabajo.

---

## 2. Detección de Toques mediante Profundidad

El sistema utiliza la cámara de tiempo de vuelo (ToF) del Kinect para medir la distancia entre el sensor y la superficie.

### 2.1 Fase de Muestreo (Estadística)
Durante la calibración, se capturan $N=500$ frames. Para cada píxel $(i, j)$ en la ROI (Region of Interest), se calcula la **Moda** de la profundidad para filtrar el ruido del sensor:

$$ d_{max}(i, j) = \text{mode}(\{d_1, d_2, ..., d_n\}) $$

Este $d_{max}$ representa el "suelo" o superficie de la mesa en estado de reposo.

### 2.2 Algoritmo de Detección
Un toque se detecta cuando la profundidad actual $d_{curr}$ es menor que $d_{max}$ (indicando un objeto sobre la mesa) pero mayor que un umbral de seguridad $d_{min}$:

$$ \text{Touch}(i, j) = \begin{cases} 
1 & \text{si } d_{max}(i, j) - \Delta_{high} < d_{curr}(i, j) < d_{max}(i, j) - \Delta_{low} \\
0 & \text{en otro caso}
\end{cases} $$

Donde:
- $\Delta_{low} \approx 4mm$ (evita falsos positivos por ruido).
- $\Delta_{high} \approx 10mm$ (grosor máximo de un toque para ignorar manos/brazos).

---

## 3. Optimización del Centroide

Para mejorar la precisión del "Click", no solo calculamos el centro geométrico del área tocada, sino un **Centroide Ponderado por Profundidad**.

### 3.1 Centroide Estándar (Momentos)
Basado en los momentos de la máscara binaria:
$$ C_x = \frac{M_{10}}{M_{00}}, \quad C_y = \frac{M_{01}}{M_{00}} $$

### 3.2 Centroide Ponderado
Asignamos mayor peso a los píxeles que están más cerca del sensor (donde la presión o el contacto es más franco):

$$ W(i, j) = \max(0, d_{max}(i, j) - d_{curr}(i, j)) $$
$$ C_{weighted} = \frac{\sum (i, j) \cdot W(i, j)}{\sum W(i, j)} $$

Finalmente, el punto de toque final es una interpolación:
$$ P_{final} = 0.7 \cdot C_{weighted} + 0.3 \cdot C_{standard} $$

---

## 4. Resumen de Flujo Matemático

1.  **Detección de Marcadores**: Umbralización y momentos para hallar $(x_c, y_c)$ de los patrones proyectados.
2.  **Solución del Modelo Geométrico**: Generación de la matriz $H$.
3.  **Filtrado Espacial**: Aplicación de la ROI y redimensionamiento de matrices para sincronizar RGB y Profundidad.
4.  **Mapeo Directo**: Aplicación de la transformación de perspectiva para cada toque detectado:
    $$ P_{projector} = H \cdot P_{camera} $$
