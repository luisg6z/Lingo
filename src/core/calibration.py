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
