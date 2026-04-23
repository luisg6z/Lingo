# Documentación Ténica y de Arquitectura: Lingo

Como parte integral de la ingeniería de software y el desarrollo de **Lingo** (entorno de realidad aumentada espacial), se confeccionó un compendio sólido de documentación orientado a asegurar la mantenibilidad a largo plazo, escalabilidad del código y facilitar la comprensión técnica de los subsistemas del proyecto a nivel de trabajo de grado. 

Toda la documentación se ha estandarizado en formato *Markdown (.md)* y *Mermaid (.mmd)*, y se divide en cuatro categorías principales que abarcan desde manuales de usuario hasta la fundamentación matemática que sustenta la visión por computadora.

---

## 1. Documentación Base y Manual de Usuario
El núcleo de la comprensión inicial para instaladores, usuarios o futuros desarrolladores radica en las guías iniciales del repositorio:
- **`README.md`**: Funciona como el punto de entrada principal. Expone el resumen funcional del sistema, un recuento de las tecnologías utilizadas, hardware vital requerido y, lo más importante, la secuencia detallada de instalación de dependencias mediada por `uv`, así como los comandos de arranque para la interfaz de juegos.
- **`EXPLICACION_SISTEMA.md`**: Un archivo robusto a nivel de repositorio que ilustra las distintas fases transversales del proyecto desde una óptica de desarrollo: desde la base sobre la que funciona el sistema de proyección, hasta cómo se interpreta la información táctil proveniente del Kinect y los bucles utilizados en los minijuegos.

---

## 2. Especificaciones Funcionales Subyacentes
El sistema Lingo contiene lógicas que requieren explicaciones detalladas y aisladas del funcionamiento principal. Estos comportamientos fueron documentados individualmente en la carpeta `docs/`.
- **`funcionamiento_kinect_toques.md`**: Describe de forma detallada cómo el análisis del mapa de profundidad captura los cambios e intermitencias en la altura usando morfoglogía y transformaciones a través de la librería OpenCV, con el objetivo de descartar ruidos en la calibración y crear la matriz de colisión en tiempo real.
- **Documentación de Interfaces Interactivas**: Abarca el desglose del funcionamiento lógico e instrucciones detrás de algunos minijuegos, como se detalla de la literatura sobre las terapias adaptadas al software:
  - `funcionamiento_absurdos_visuales.md`: Documento dedicado al minijuego "Absurdos Visuales".
  - `funcionamiento_juego_historia.md`: Explicación referida al flujo principal interactivo del juego "Historia".

---

## 3. Fundamentos Teóricos y Matemáticos
Por las características de la realidad espacial y uso del sensor RGB-D en un plano, parte esencial del proyecto es el cruce de datos y geometría proyectiva:
- **`calibracion_matematica.md`**: Se encarga de detallar, desde una perspectiva puramente lógico-matemática, cómo se lleva a cabo el mapeado y la matriz de *homografía*. Muestra cómo los toques obtenidos del espacio de la cámara referencial (`dmax_map` y ROI) logran trasladarse en forma precisa al plano cartesiano del Videobeam corrigiendo cualquier distorsión de perspectiva asimétrica.

---

## 4. Modelado y Arquitectura UML
Para representar estructural y comportamentalmente el flujo algorítmico global, la base de datos de actividades y la organización del código, se elaboraron diversos diagramas implementados mediante *Mermaid* integrados al control de versiones.

- **Modelo C4 (`docs/c4_model/`)**: Se representó el sistema desde distintas capas de abstracción (Contexto, Contenedores, Componentes, Código). Esto permite que tanto personal no-técnico (ej. psicopedagogos) como desarrolladores comprendan todos los actores involucrados, y la interrelación de procesos internos.
- **Lógica UML de Flujos y Estados**:
  - `diagrama_casos_uso.mmd`: Modela los eventos interactivos que disparan tanto terapeutas como los niños (actores primarios de la aplicación).
  - `diagrama_clases_sistema.mmd`: Estructuración y organización de la base de código implementando orientación asimilada a clases.
  - `diagrama_secuencia_sistema.mmd`: Muestra la interacción a través del tiempo entre el hardware interno y la capa de software a partir de que un niño ejecuta una acción física sobre la mesa.
  - `diagrama_estados_sistema.mmd` y `diagrama_flujo_clasificacion.mmd`: Documentan los ciclos de vida y transiciones (ciclo iterativo Menú > Calibración > Juego > Resultados) del entorno preescolar en general y sub-modelos específicos.

---

## 5. Estructura del Código y Organización del Repositorio (`src/`)
Para garantizar la escalabilidad, separación de responsabilidades y la mantenibilidad del código asimilable a plataformas complejas, el repositorio centraliza su lógica y bucles base dentro del directorio `src/`. Esta modularización refleja los patrones de diseño abordados en el desarrollo del proyecto:

- **`src/`**: Directorio raíz con el paquete principal de ejecución de Lingo.
  - **`main.py`**: Interfaz técnica y script principal del entorno. Es el encargado de orquestar el inicio del sistema, proyección del sistema de cartas principal (UI) y la delegación de eventos y flujos de entrada.
  - **`core/`**: Aloja las operaciones de procesamiento complejas que son agnósticas al contexto de un juego. Contiene submódulos para el procesamiento geométrico-matemático (Homografía), el motor del sensor RGB-D y las lecturas de los contornos para la colisión háptica virtual.
  - **`features/`**: Concentra las implementaciones aisladas de los distintos minijuegos (ej. juego de historias, clasificación, absurdos visuales). Este encapsulamiento permite a los mantenedores futuros integrar o modificar minijuegos sin alterar la base matriz (*core*) del software.
  - **`components/`**: Contiene la definición y construcción sistemática de la interfaz visual. Incluye elementos gráficos modulares (cards interactivas, botones proyectados o notificaciones) que se dibujan uniformemente.
  - **`config/`**: Responsable de la inicialización de variables base, de la gestión local de las calibraciones y las persistencias de umbrales morfológicos y de profundidad obtenidos por el sistema.
