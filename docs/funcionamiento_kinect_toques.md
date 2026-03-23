# Funcionamiento del Sensor Kinect y Detección de Toques Espaciales

## 1. Fundamentación Teórica
El mecanismo de interacción táctil del presente proyecto se fundamenta en los principios de visión por computadora mediante el uso de sensores de profundidad. Este enfoque teórico y arquitectónico toma inspiración directa del trabajo de Andrew D. Wilson titulado **"Using a Depth Camera as a Touch Sensor"** [1], publicado en la conferencia *ACM International Conference on Interactive Tabletops and Surfaces* (ITS 2010).

En dicho documento, Wilson plantea el uso combinado de una cámara de profundidad y un proyector para transformar superficies físicas ordinarias e irregulares en superficies interactivas, prescindiendo de la necesidad de instrumentar la mesa con sensores capacitivos o resistivos independientes. Esto se logra mediante el modelado analítico continuo de un mapa de topografía del entorno estático y la configuración de una capa o "corredor" interactivo directamente encima de este nivel base. Cuando el sensor de profundidad advierte la intrusión de un objeto físico (o las manos del usuario) que se sobrepone a dicha zona delimitada, el suceso es interceptado en forma de "clúster" o parche de colisión y decodificado como información de toque en el espacio tridimensional. El presente sistema homologa y optimiza esta metodología para posibilitar una pizarra interactiva fluida y adaptable a distintas mesas.

## 2. Integración del Sensor Kinect en la Arquitectura
De acuerdo a la implementación estructural examinada en el módulo de calibración (`src/core/calibrate_area.py`), las capacidades del dispositivo sensórico Kinect son instrumentalizadas a su nivel inferior mediante el uso de la biblioteca y abstracción **OpenNI2**. La recolección de eventos físicos depende esencialmente de dos flujos simultáneos de adquisición óptica:

1. **Flujo de Color RGB:** 
   Es empleado con propósitos de calibración, validación geométrica inicial y correspondencia visual. Inicialmente, el sistema proyecta en la superficie mediante el _videobeam_ patrones blancos de referencia (cuadrados) a las periferias de la mesa. La cámara captura estos marcadores aplicando algoritmos de umbralización cruzada (`cv2.adaptiveThreshold`) acoplados a heurísticas estructuradas dimensionales (porcentaje de rellenado, factor de forma rectilíneo, límite de proporciones, momentos rotacionales espaciales), ubicando los vértices interactables.
   
2. **Flujo Infrarrojo (Cámara de Profundidad):**
   Constituye el núcleo de la sensórica inmersiva. El sistema captura una densa malla topográfica estructurando alturas volumétricas. Este flujo sirve para componer tanto la matriz "muerta" o de fondo, como para supervisar el ingreso de comandos táciles iterativamente. Para coordinar la disparidad de visuales, ambas mallas se emparejan derivando a una **Matriz de Homografía** matemática (`cv2.getPerspectiveTransform`), que proyecta con altísima fidelidad las transformaciones escalares y rotacionales entre los píxeles captados y la matriz de la proyección virtual en pantalla.

## 3. Algoritmo de Detección de Toques Interactivo
La transición heurística y algorítmica de la información bruta de la cámara hacia una pulsación digital validada, envuelve varias fases sucesivas de procesamiento para aislar imperfecciones o perturbaciones ambientales:

### 3.1 Construcción del Referente Basal (`dmax_map`)
Como prolegómeno a la detección, resulta necesario estimar un contexto exento de interacciones humanas temporales. Evaluando un acumulado de cientos de fotogramas estáticos con la zona despejada, el sistema procesa la moda estadística frecuencial de profundidad en cada uno de los sectores de la ROI (Región de Interés). Esta superposición matricial de frecuencias resulta en el mapa general base o `dmax_map`, fijando distancias precisas, a nivel dimensional de los píxeles, de la separación existente entre el equipo emisor espacial infrarrojo y la mesa per se.

### 3.2 La Coraza o Capa Volumétrica Virtual
Determinado el suelo, todo objeto interactivo que pretenda clasificarse como táctil debe aproximarse en la estrecha banda volumétrica paralela a dicho piso. Se establecen umbrales de contención rígidos reconfigurando la holgura en milímetros:
- El umbral inferior estricto (`dmax_map`), desplazado ligeramente desde la matriz original original reduciendo márgenes (por ej. `- 4`) para compensar reflexiones y ruído base.
- El umbral de altura límite o cénit detectivo delimitado en un relativo (`dmin_map`) posicionado convencionalmente debajo del máximo reajustado (profundidad inferior nominal).

En tiempo de ejecución continua, toda lectura arrojada por el estativo infrarrojo que perfore esa cuña estricta delimitada `[dmin_map, dmax_map]` engendrará una máscara booleana (`touch_mask`) representativa abstracta de la colisión física.

### 3.3 Filtros de Persistencia Temporal y Suavizado
Con miras a erradicar falsos positivos emanados de la atenuación infrarroja esporádica inherente a los sensores ópticos de consumo comercial, la careta oclógica transita los siguientes filtros:
* **Filtros Espaciales Morfológicos y Medianos:** Empleando librerías matriciales, se inyectan filtrados de mediana (`cv2.medianBlur`) y se operan limpiezas morfológicas de clausura y apertura. Esta desinfección difumina motas satelitales inconexas del rebote infrarrojo, reconstituyendo geometrías dactilares desarticuladas.
* **Persistencia Cíclica (Histórica):** Absteniéndose de computar decisiones prematuras en microcolisiones incidentales, las máscaras depuradas se resguardan en colas secuenciales de memoria temporal (`touch_history`). Ocurriendo entonces que aquel grupo conexo que sobreviva con durabilidad temporal preconfigurada sumariando acumulador afirmativo (`accumulated_mask`), superaría el descarte, avalando su legitimidad ininterrumpida frente a los parpadeos inescrutables.

### 3.4 Geometría y Centroide Ponderado por Profundidad
En su estancia final, el conjunto binario decantado es sujeto de un escáner de componentes adyacentes conexos (`cv2.connectedComponentsWithStats`) para cuantificar los cuerpos invasores, comprobando normativas restrictivas de envergadura milimétrica.

A los elementos válidos se les deduce su baricentro. En el algoritmo destaca la aplicación de un momento geométrico compuesto dependiente en proporciones y peso de la elevación (distancia):
1. Porcentaje de preponderancia geométrica tradicional estándar (x/y planar).
2. Un sesgo intensivo o gravitatorio explícitamente derivado por factor de altura (`depth-weighted centroid`); las cuotas espaciales topográficas de mayor intromisión en el corredor acarrearían un anclaje superior en la designación final del "clic", previendo paralajes de visualización en yemas de dedo en ángulo.

Con ambas estimaciones fusionadas linealmente, el vector 2D en plano de cámara es deformado usando la matriz de homografía antedicha. El remate es una entrega milimetricamente simétrica, traduciendo ese asalto estocástico del infrarrojo sobre la mesa, a una representación lógica asestada en el píxel diana predeterminado del proyector en vivo, cerrando el bucle del registro tácitamente táctil en la máquina local.

---

**Referencias Documentales**

[1] Wilson, A. D. (2010). *Using a depth camera as a touch sensor*. En Proceedings of the ACM International Conference on Interactive Tabletops and Surfaces (ITS '10). Association for Computing Machinery, New York, NY, USA, 69–72. DOI: https://doi.org/10.1145/1936652.1936665
