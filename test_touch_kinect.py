# -*- coding: utf-8 -*-
"""
Test script to visualize Kinect touch detection with exact coordinates.
Uses the same calibration as the main app (src/config/ultima_configuracion_coordenadas.json + dmax_map.txt).
Shows where each touch is detected in both ROI (window) and viewport (projection) coordinates.

Run after calibrating:  python test_touch_kinect.py   or   uv run test_touch_kinect.py
Press 'q' to quit.
"""

import cv2
import numpy as np
import json
import os
import sys
from openni import openni2

_project_root = os.path.abspath(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.calibration import get_coordenadas_path, load_touch_depth_maps

# Viewport (projection) size - must match your calibration
VIEW_WIDTH = 1280
VIEW_HEIGHT = 800

# Min/max contour area to count as a touch (reject noise and huge blobs)
MIN_TOUCH_AREA = 100
MAX_TOUCH_AREA = 50000


def window_to_viewport(cx, cy, xw_min, xw_max, yw_min, yw_max, xv_min, xv_max, yv_min, yv_max):
    """Map (cx, cy) in depth ROI to viewport (x, y)."""
    x_viewport = int(xv_min + (cx) * (xv_max - xv_min) / (xw_max - xw_min))
    y_viewport = int(yv_min + (cy) * (yv_max - yv_min) / (yw_max - yw_min))
    return x_viewport, y_viewport


def run_touch_test(device):
    config_path = get_coordenadas_path()
    if not os.path.exists(config_path):
        print(f"ERROR: No se encontró '{config_path}'. Ejecuta la calibración primero.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dmax_map, dmin_map = load_touch_depth_maps(config, band_profile="menu")
    if dmax_map is None:
        return

    xw_min = config["xw_min"]
    xw_max = config["xw_max"]
    yw_min = config["yw_min"]
    yw_max = config["yw_max"]
    xv_min = config["xv_min"]
    xv_max = config["xv_max"]
    yv_min = config["yv_min"]
    yv_max = config["yv_max"]
    homography_matrix = config.get("homography_matrix")
    if homography_matrix is not None:
        homography_matrix = np.array(homography_matrix, dtype=np.float32)

    depth_stream = device.create_depth_stream()
    depth_stream.start()

    # Windows: projection (with grid + touches + coords), touch mask, optional depth ROI
    cv2.namedWindow("Touch test - Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Touch test - Proyeccion", 1920, 0)
    cv2.setWindowProperty("Touch test - Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow("Mascara de Toque", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI Profundidad (normalizado)", cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("TEST DE TOQUES KINECT")
    print("=" * 60)
    print("Toca el área calibrada. Se mostrarán las coordenadas exactas.")
    print("  - Proyección: coordenadas viewport (x, y) en píxeles de proyección")
    print("  - Consola: viewport (x, y) y ROI (cx, cy)")
    print("Presiona 'q' en cualquier ventana para salir.")
    print("=" * 60)

    while True:
        depth_frame = depth_stream.read_frame()
        if depth_frame is None:
            continue

        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        depth_data = cv2.flip(depth_data, 1)
        depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]

        if depth_roi.shape != dmax_map.shape:
            print("ERROR: dimensiones ROI vs dmax_map no coinciden.")
            break

        # Touch mask (same pipeline as main.py: strict threshold + morphology)
        touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
        touch_mask = cv2.medianBlur(touch_mask, ksize=5)
        touch_mask = cv2.GaussianBlur(touch_mask, (7, 7), 0)
        touch_mask_lowpass = cv2.boxFilter(touch_mask, ddepth=-1, ksize=(3, 3))
        _, touch_mask_final = cv2.threshold(touch_mask_lowpass, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_OPEN, kernel)
        touch_mask_final = cv2.morphologyEx(touch_mask_final, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(touch_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build projection image: dark background, calibrated area with grid, touches with exact coords
        proyeccion = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)
        proyeccion[:] = (30, 30, 30)

        # Calibrated area rectangle
        cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

        # Grid inside calibrated area (every 80 px for readability)
        step = 80
        for xx in range(xv_min, xv_max + 1, step):
            cv2.line(proyeccion, (xx, yv_min), (xx, yv_max), (60, 60, 60), 1)
        for yy in range(yv_min, yv_max + 1, step):
            cv2.line(proyeccion, (xv_min, yy), (xv_max, yy), (60, 60, 60), 1)

        # Instructions
        cv2.putText(proyeccion, "Toca el area verde. Coordenadas viewport (x, y) se muestran en cada toque.",
                    (xv_min, yv_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(proyeccion, "q = salir", (xv_min, yv_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        touch_points_viewport = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (MIN_TOUCH_AREA <= area <= MAX_TOUCH_AREA):
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Map to viewport
            if homography_matrix is not None:
                x_cam = xw_min + cx
                y_cam = yw_min + cy
                pt_cam = np.array([[[x_cam, y_cam]]], dtype=np.float32)
                pt_view = cv2.perspectiveTransform(pt_cam, homography_matrix)
                x_viewport = int(round(pt_view[0][0][0]))
                y_viewport = int(round(pt_view[0][0][1]))
            else:
                x_viewport, y_viewport = window_to_viewport(
                    cx, cy, xw_min, xw_max, yw_min, yw_max, xv_min, xv_max, yv_min, yv_max
                )

            x_viewport = np.clip(x_viewport, 0, VIEW_WIDTH - 1)
            y_viewport = np.clip(y_viewport, 0, VIEW_HEIGHT - 1)
            touch_points_viewport.append((x_viewport, y_viewport, cx, cy, area))

        for (x_viewport, y_viewport, cx, cy, area) in touch_points_viewport:
            # Crosshair + circle at touch
            cv2.circle(proyeccion, (x_viewport, y_viewport), 12, (0, 0, 255), 2)
            cv2.circle(proyeccion, (x_viewport, y_viewport), 3, (0, 0, 255), -1)
            cv2.line(proyeccion, (x_viewport - 18, y_viewport), (x_viewport + 18, y_viewport), (0, 200, 255), 2)
            cv2.line(proyeccion, (x_viewport, y_viewport - 18), (x_viewport, y_viewport + 18), (0, 200, 255), 2)

            # Exact coordinates text (viewport)
            coord_text = f"({x_viewport}, {y_viewport})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thick = 2
            (tw, th), _ = cv2.getTextSize(coord_text, font, scale, thick)
            # Background for readability
            pad = 4
            cv2.rectangle(
                proyeccion,
                (x_viewport + 20, y_viewport - th - pad),
                (x_viewport + 20 + tw + pad * 2, y_viewport + pad),
                (0, 0, 0),
                -1,
            )
            cv2.putText(proyeccion, coord_text, (x_viewport + 22, y_viewport - 5), font, scale, (0, 255, 0), thick)

            # Optional: small ROI coords below
            roi_text = f"ROI:({cx},{cy})"
            (rw, rh), _ = cv2.getTextSize(roi_text, font, 0.45, 1)
            cv2.rectangle(
                proyeccion,
                (x_viewport + 20, y_viewport + 5),
                (x_viewport + 22 + rw + 4, y_viewport + 10 + rh + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(proyeccion, roi_text, (x_viewport + 22, y_viewport + 18), font, 0.45, (200, 200, 0), 1)

            print(f"  Toque viewport ({x_viewport}, {y_viewport})  |  ROI ({cx}, {cy})  area={area}")

        # Depth ROI normalized for display (optional debug)
        depth_vis = depth_roi.copy().astype(np.float32)
        valid = (depth_vis > 0) & (depth_vis < 4000)
        if np.any(valid):
            d_min, d_max = depth_vis[valid].min(), depth_vis[valid].max()
            if d_max > d_min:
                depth_vis = (depth_vis - d_min) / (d_max - d_min)
        depth_vis = np.clip(depth_vis, 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Touch test - Proyeccion", proyeccion)
        cv2.imshow("Mascara de Toque", touch_mask_final)
        cv2.imshow("ROI Profundidad (normalizado)", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    depth_stream.stop()
    cv2.destroyAllWindows()
    print("Test de toques finalizado.")


def main():
    openni2_paths = [
        os.environ.get("OPENNI2_PATH"),
        os.path.join(os.environ.get("OPENNI2_PATH", ""), "Redist"),
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
    ]
    program_files = os.environ.get("ProgramFiles", "")
    if program_files:
        openni2_paths.extend([
            os.path.join(program_files, "OpenNI2", "Redist"),
            os.path.join(program_files, "OpenNI2"),
        ])
    openni2_paths = list(dict.fromkeys([p for p in openni2_paths if p and os.path.exists(p)]))

    if not openni2_paths:
        for path in ["C:/Program Files/OpenNI2/Redist", "C:/Program Files (x86)/OpenNI2/Redist"]:
            if os.path.exists(path):
                openni2_paths.append(path)
                break

    openni2_initialized = False
    for path in openni2_paths:
        try:
            openni2.initialize(path)
            openni2_initialized = True
            print(f"OpenNI2 inicializado desde: {path}")
            break
        except Exception:
            continue

    if not openni2_initialized:
        print("ERROR: No se pudo cargar OpenNI2. Instala el SDK o configura OPENNI2_PATH.")
        return 1

    try:
        device = openni2.Device.open_any()
        run_touch_test(device)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        openni2.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
