# Calibración táctil y homografía

## Scripts

- `src/core/calibrate_area.py`: dos marcadores; homografía a partir de un rectángulo axis-aligned ampliado en profundidad.
- `src/core/calibrate_area_four_points.py`: cuatro marcadores en las esquinas; homografía con correspondencias reales (centroides) y, en el JSON, `calibration_quad_camera` / `calibration_quad_projection` (orden TL, TR, BR, BL).

Ambos escriben `ultima_configuracion_coordenadas.json` y `dmax_map.txt` vía `get_coordenadas_path()` / `get_dmax_map_path()`.

## Mapeo en tiempo de ejecución (importante)

La matriz `homography_matrix` transforma **coordenadas absolutas** del plano depth/color Kinect (640×480, misma convención que al calibrar, con flip horizontal) a **píxeles del proyector** (viewport 1280×800).

Usa `src.core.calibration.map_depth_roi_to_viewport(cx_roi, cy_roi, coordenadas, view_width, view_height)` para mapear un punto **relativo a la ROI** de profundidad al viewport: si hay `homography_matrix`, aplica `perspectiveTransform`; si no, escalado lineal sobre `xv_*` / `yw_*`.

**Integrado en:** menú principal y opciones de clasificación (`menu.py`), flujo principal de absurdos visuales (`absurdos-visuales/game.py`). Otras pantallas (historia, clasificación en `levels.py`, etc.) y `detection_logic.map_coordinates` pueden seguir con escalado lineal hasta migrarlas al mismo helper.

## Campos opcionales en el JSON

- `calibration_quad_camera`: cuatro pares `[x, y]` en cámara (orden TL, TR, BR, BL).
- `calibration_quad_projection`: cuatro pares en proyector, mismo orden.
- `calibration_method`: por ejemplo `"four_point_squares"` cuando aplica.

Los consumidores que solo lean `xw_*`, `yv_*` y `homography_matrix` siguen siendo válidos; los cuadriláteros sirven para depuración o para recomputar H si se desea.
