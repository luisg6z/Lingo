# Explicación de Calibración: 2 Cuadrados vs. 4 Cuadrados

Este documento detalla la lógica matemática detrás de la elección de usar dos puntos de referencia en lugar de cuatro en el sistema de calibración de Lingo.

---

## 1. El Dilema Matemático

Para realizar una **Transformación de Perspectiva (Homografía)** real, se necesitan matemáticamente **4 puntos** independientes. Esto se debe a que la matriz de homografía tiene 8 grados de libertad, y cada punto aporta 2 ecuaciones ($x$ e $y$).

### ¿Por qué el proyecto usa solo 2?
En el archivo `src/core/calibrate_area.py`, el sistema proyecta solo dos cuadrados (Inferior Izquierdo y Superior Derecho). Esto se hace por **usabilidad**: es mucho más sencillo para el usuario asegurar que dos puntos estén dentro del campo de visión que alinear cuatro esquinas perfectamente.

---

## 2. La "Simulación" de Esquinas

Aunque solo se miden 2 puntos, el código genera artificialmente los otros 2 para poder llamar a la función `cv2.getPerspectiveTransform()`. 

```python
# El código toma los límites de los 2 puntos medidos
xw_min, xw_max = sorted([puntos_camara[1][0], puntos_camara[0][0]])
yw_min, yw_max = sorted([puntos_camara[0][1], puntos_camara[1][1]])

# Genera un rectángulo "perfecto" basado en esos límites
pts_camara = np.array([
    [xw_min, yw_min_escalado],           # Superior Izquierda (Calculada)
    [xw_max_escalado, yw_min_escalado],  # Superior Derecha (Medida)
    [xw_max_escalado, yw_max],           # Inferior Derecha (Calculada)
    [xw_min, yw_max]                     # Inferior Izquierda (Medida)
])
```

---

## 3. Consecuencias Técnicas

### Transformación Afín vs. Homografía
Al derivar las esquinas de esta manera, la transformación resultante deja de ser una "Homografía de Perspectiva" pura y se convierte en una **Transformación de Escala y Traslación (Afín)**.

*   **Pérdida de Corrección de Inclinación**: Si la cámara Kinect está inclinada respecto a la mesa, el área se verá como un trapecio. El método de 2 puntos **no puede ver esta deformación** y asumirá que la mesa es un rectángulo perfecto.
*   **Precisión**: La precisión será óptima en el centro, pero puede haber un pequeño "desplazamiento" (drift) en las esquinas si la cámara no está perfectamente nivelada.

---

## 4. Resumen Comparativo

| Característica | 2 Cuadrados (Actual) | 4 Cuadrados (Ideal) |
| :--- | :--- | :--- |
| **Complejidad de Usuario** | Baja (Muy fácil) | Media (Requiere alineación) |
| **Tipo de Transformación** | Escala / Lineal | Perspectiva Real |
| **Corrección de Keystone** | No | Sí |
| **Robustez ante Inclinación** | Baja | Alta |

---

## 5. Conclusión

El sistema actual prioriza la **experiencia del usuario**. Para que la calibración de 2 puntos sea efectiva, el sensor Kinect debe estar lo más nivelado posible y centrado sobre el área de proyección. Los "factores de escala" (`1.16` y `1.12`) presentes en el código actúan como compensadores para ajustar el área de profundidad a la proyección de forma empírica.
