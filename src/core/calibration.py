"""
Calibration utilities for MagicboARd
"""
import os

import cv2
import numpy as np

# Perfil por defecto: (resta sobre mapa crudo para dmax, separación dmax-dmin)
_TOUCH_BAND_DEFAULTS = {
    "menu": (5, 7),
    "clasificacion": (7, 50),
}


def get_src_config_dir():
    """Directorio canónico: .../src/config (independiente del CWD)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"))


def get_dmax_map_path():
    return os.path.join(get_src_config_dir(), "dmax_map.txt")


def get_coordenadas_path():
    return os.path.join(get_src_config_dir(), "ultima_configuracion_coordenadas.json")


def homography_matrix_from_config(coordenadas):
    """3x3 float32 o None si no hay matriz guardada."""
    hm = coordenadas.get("homography_matrix")
    if hm is None:
        return None
    return np.array(hm, dtype=np.float32)


def map_depth_roi_to_viewport(
    cx_roi,
    cy_roi,
    coordenadas,
    view_width=1280,
    view_height=800,
):
    """
    Pasa un punto de la ROI de profundidad (origen arriba-izquierda del recorte) a píxeles del viewport.

    Si existe homography_matrix, usa coordenadas absolutas cámara 640x480 (misma convención que la calibración)
    y cv2.perspectiveTransform. Si no, usa el escalado lineal histórico sobre xv_/yv_ y xw_/yw_.
    """
    xv_min = coordenadas["xv_min"]
    xv_max = coordenadas["xv_max"]
    yv_min = coordenadas["yv_min"]
    yv_max = coordenadas["yv_max"]
    xw_min = coordenadas["xw_min"]
    xw_max = coordenadas["xw_max"]
    yw_min = coordenadas["yw_min"]
    yw_max = coordenadas["yw_max"]
    w_roi = xw_max - xw_min
    h_roi = yw_max - yw_min

    H = homography_matrix_from_config(coordenadas)
    if H is not None and w_roi > 0 and h_roi > 0:
        x_cam = float(cx_roi) + float(xw_min)
        y_cam = float(cy_roi) + float(yw_min)
        pt = np.array([[[x_cam, y_cam]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        x_touch = int(round(float(out[0, 0, 0])))
        y_touch = int(round(float(out[0, 0, 1])))
    else:
        if w_roi <= 0 or h_roi <= 0:
            return 0, 0
        x_touch = int(xv_min + float(cx_roi) * (xv_max - xv_min) / w_roi)
        y_touch = int(yv_min + float(cy_roi) * (yv_max - yv_min) / h_roi)

    x_touch = int(np.clip(x_touch, 0, max(0, view_width - 1)))
    y_touch = int(np.clip(y_touch, 0, max(0, view_height - 1)))
    return x_touch, y_touch


def map_camera_to_viewport(x_cam, y_cam, coordenadas, view_width=1280, view_height=800):
    """
    Pasa un píxel del frame de color Kinect 640×480 (ya volteado como en el juego)
    al viewport lógico. Reutiliza la misma homografía o escalado que
    ``map_depth_roi_to_viewport`` para alinear overlays (p. ej. YOLO) con los toques.
    """
    xw_min = float(coordenadas["xw_min"])
    yw_min = float(coordenadas["yw_min"])
    return map_depth_roi_to_viewport(
        float(x_cam) - xw_min,
        float(y_cam) - yw_min,
        coordenadas,
        view_width=view_width,
        view_height=view_height,
    )


def load_and_validate_dmax_map(coordenadas):
    """
    Carga y valida el archivo dmax_map.txt contra las coordenadas proporcionadas.

    Args:
        coordenadas: Diccionario con las coordenadas de calibración

    Returns:
        tuple: (dmax_map_reshaped, w, h) si es exitoso, (None, None, None) si hay error
    """
    config_path = get_dmax_map_path()

    try:
        dmax_map = np.loadtxt(config_path, dtype=np.uint16)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{config_path}'.")
        print("Por favor, ejecuta la calibración primero.")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: El archivo 'dmax_map.txt' tiene un formato inválido: {e}")
        return None, None, None

    xw_min = coordenadas.get("xw_min", 0)
    xw_max = coordenadas.get("xw_max", 0)
    yw_min = coordenadas.get("yw_min", 0)
    yw_max = coordenadas.get("yw_max", 0)

    w = xw_max - xw_min
    h = yw_max - yw_min

    expected_size = w * h
    actual_size = dmax_map.size

    if actual_size != expected_size:
        print("=" * 70)
        print("ERROR: Incompatibilidad de dimensiones entre dmax_map.txt y coordenadas")
        print("=" * 70)
        print(f"Dimensiones esperadas: {w} x {h} = {expected_size} elementos")
        print(f"Dimensiones del archivo: {actual_size} elementos")

        posibles_dimensiones = []
        for i in range(300, 500):
            if actual_size % i == 0:
                j = actual_size // i
                posibles_dimensiones.append((i, j))

        if posibles_dimensiones:
            print("\nDimensiones posibles del archivo:")
            for dim in posibles_dimensiones[:3]:
                print(f"  - {dim[0]} x {dim[1]} o {dim[1]} x {dim[0]}")

        print("\nSOLUCIÓN:")
        print("1. Ejecuta la calibración para regenerar dmax_map.txt con las dimensiones correctas:")
        print("   python src/core/calibrate_area.py")
        print(f"\n2. O ajusta las coordenadas en '{get_coordenadas_path()}'")
        print("   para que coincidan con las dimensiones del archivo actual.")
        print("=" * 70)
        return None, None, None

    try:
        dmax_map_reshaped = dmax_map.reshape((h, w))
        return dmax_map_reshaped, w, h
    except ValueError as e:
        print(f"ERROR al hacer reshape del dmax_map: {e}")
        return None, None, None


def load_touch_depth_maps(coordenadas, band_profile="menu"):
    """
    Carga dmax_map y construye dmax/dmin para detección de toques.

    Opciones en coordenadas (JSON), todas opcionales:
      - surface_depth_offset: suma global (mm) a dmax y dmin tras la banda; corrige mesa más alta/baja.
      - touch_dmax_subtract: sustituye la resta por defecto del perfil.
      - touch_dmin_span: sustituye la separación (dmax - dmin) por defecto del perfil.

    Args:
        coordenadas: dict de calibración (mismas claves que el JSON).
        band_profile: "menu" | "clasificacion"

    Returns:
        (dmax_map, dmin_map) como int32, o (None, None) si falla la carga.
    """
    dmax_raw, _w, _h = load_and_validate_dmax_map(coordenadas)
    if dmax_raw is None:
        return None, None

    sub, span = _TOUCH_BAND_DEFAULTS.get(band_profile, _TOUCH_BAND_DEFAULTS["menu"])
    if "touch_dmax_subtract" in coordenadas:
        sub = int(coordenadas["touch_dmax_subtract"])
    if "touch_dmin_span" in coordenadas:
        span = int(coordenadas["touch_dmin_span"])

    offset = int(coordenadas.get("surface_depth_offset", 0))

    raw = dmax_raw.astype(np.int32)
    dmax = raw - sub + offset
    dmin = dmax - span
    return dmax, dmin


def capture_dmax_map(
    device,
    calibrated_area,
    num_frames=500,
    min_depth=500,
    max_depth=4000,
    save_path=None,
    progress=None,
):
    """
    Muestrea la profundidad en la ROI y genera el mapa modal (superficie vacía).

    Args:
        device: dispositivo OpenNI2 ya abierto.
        calibrated_area: (x, y, w, h) en coordenadas de depth 640x480 flipped.
        num_frames: frames a acumular.
        min_depth, max_depth: rango válido (mm).
        save_path: si se indica (o por defecto get_dmax_map_path()), guarda el .txt plano.
        progress: iterable opcional (ej. tqdm(range(n))) para barra; si None, usa range(num_frames).

    Returns:
        ndarray (h, w) uint16 profundidad modal por píxel.
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    x, y, w, h = calibrated_area
    depth_stream = device.create_depth_stream()
    depth_stream.start()

    depth_accum = np.zeros((h, w, max_depth - min_depth + 1), dtype=int)

    iterator = progress if progress is not None else (
        _tqdm(range(num_frames), desc="Numero de frames", unit="frames")
        if _tqdm is not None else range(num_frames)
    )

    for _ in iterator:
        frame = depth_stream.read_frame()
        depth_data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[y : y + h, x : x + w]

        valid_mask = (depth_roi >= min_depth) & (depth_roi <= max_depth)
        valid_depth = depth_roi[valid_mask] - min_depth
        indices = np.where(valid_mask)
        depth_accum[indices[0], indices[1], valid_depth] += 1

    depth_stream.stop()

    dmax_map = np.argmax(depth_accum, axis=2) + min_depth

    out_path = save_path if save_path is not None else get_dmax_map_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, dmax_map.flatten(), fmt="%d")

    return dmax_map


if __name__ == "__main__":
    """
    Misma calibración que ``calibrate_area.py`` (mismo OpenNI + ventanas).

    Se ejecuta ese archivo con ``runpy`` para que su ``if __name__ == "__main__"``
    corra de verdad; así no hay divergencia con ``initialize_openni2`` ni doble
    carga del módulo ``calibration`` como ``__main__`` vs ``src.core.calibration``.
    """
    import runpy
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_here, "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    _calibrate_area = os.path.join(_here, "calibrate_area.py")
    # Misma interfaz de línea de órdenes que calibrate_area.py
    sys.argv = [os.path.basename(_calibrate_area)] + sys.argv[1:]
    runpy.run_path(_calibrate_area, run_name="__main__")
