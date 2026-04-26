"""
Calibración mesa / Kinect con 4 marcadores y homografía por correspondencias reales.

Proyecta cuatro cuadrados blancos (TL, TR, BR, BL), detecta sus centroides en color
y resuelve la asignación cámara↔proyector probando las 24 permutaciones.

Mapeo en runtime: usar src.core.calibration.map_depth_roi_to_viewport (menú y absurdos
visuales ya lo usan cuando hay homography_matrix). Ver docs/calibracion_touch_homografia.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import permutations

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
from tqdm import tqdm
from openni import openni2
import customtkinter as ctk

from src.core.calibration import capture_dmax_map, get_coordenadas_path, get_dmax_map_path
from src.core.calibrate_area import run_solo_profundidad


def mostrar_mensaje(proyeccion, texto, xv_min, yv_min, xv_max, yv_max):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(texto, font, 2, 4)[0]
    text_x = xv_min + (xv_max - xv_min - text_size[0]) // 2
    text_y = yv_min + (yv_max - yv_min + text_size[1]) // 2
    cv2.putText(proyeccion, texto, (text_x, text_y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)


def calculate_dmax(device, calibrated_area, xv_min, yv_min, xv_max, yv_max, num_frames=500):
    x, y, w, h = calibrated_area
    print(f"{w} * {h} = {w * h} ")

    proyeccion = np.zeros((800, 1280, 3), dtype=np.uint8)
    mostrar_mensaje(proyeccion, "Calibrando...", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1)

    dmax_map = capture_dmax_map(
        device,
        calibrated_area,
        num_frames=num_frames,
        save_path=get_dmax_map_path(),
        progress=tqdm(range(num_frames), desc="Numero de frames", unit="frames"),
    )

    cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 0, 0), -1)
    mostrar_mensaje(proyeccion, "Calibracion Completada", xv_min, yv_min, xv_max, yv_max)
    cv2.imshow("Proyeccion", proyeccion)
    cv2.waitKey(1000)

    return dmax_map


def proyectar_cuatro_cuadrados(view_width: int, view_height: int):
    """
    Cuatro cuadrados blancos cerca de las esquinas del área útil.
    Orden de centroides en proyección (para homografía): TL, TR, BR, BL (sentido horario).
    """
    proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    S = 50
    mh = 100
    mv = 100

    # (x0, y0) esquina superior izquierda de cada cuadrado
    tl = (mh, mv)
    tr = (view_width - mh - S, mv)
    br = (view_width - mh - S, view_height - mv - S)
    bl = (mh, view_height - mv - S)

    for (x0, y0) in (tl, tr, br, bl):
        cv2.rectangle(
            proyeccion,
            (x0, y0),
            (x0 + S, y0 + S),
            (255, 255, 255),
            -1,
        )

    def center(xy):
        x0, y0 = xy
        return (x0 + S // 2, y0 + S // 2)

    centers_proj = np.array(
        [center(tl), center(tr), center(br), center(bl)],
        dtype=np.float32,
    )

    # Bbox del rectángulo que envuelve los cuatro cuadrados (referencia viewport / fallback lineal)
    xs = [tl[0], tr[0], br[0], bl[0]]
    ys = [tl[1], tr[1], br[1], bl[1]]
    xv_min = min(xs)
    xv_max = max(x + S for x in xs)
    yv_min = min(ys)
    yv_max = max(y + S for y in ys)

    return proyeccion, centers_proj, int(xv_min), int(xv_max), int(yv_min), int(yv_max), S


def detectar_cuadrados_hasta_n(frame, max_n: int = 4, min_dist_duplicado: float = 40.0):
    """Misma lógica que calibrate_area.detectar_cuadrados pero devuelve hasta max_n contornos."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cuadrados_encontrados = []

    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, thresh_fixed1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, thresh_fixed2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, thresh_fixed3 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    for thresh in [thresh_adapt, thresh_fixed1, thresh_fixed2, thresh_fixed3]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if 3 <= len(approx) <= 5:
                        rect_area = w * h
                        extent = float(area) / rect_area if rect_area > 0 else 0
                        if extent > 0.7:
                            es_duplicado = False
                            for cuadrado_existente, _ in cuadrados_encontrados:
                                M_existente = cv2.moments(cuadrado_existente)
                                M_nuevo = cv2.moments(cnt)
                                if M_existente["m00"] > 0 and M_nuevo["m00"] > 0:
                                    cx_e = M_existente["m10"] / M_existente["m00"]
                                    cy_e = M_existente["m01"] / M_existente["m00"]
                                    cx_n = M_nuevo["m10"] / M_nuevo["m00"]
                                    cy_n = M_nuevo["m01"] / M_nuevo["m00"]
                                    distancia = np.hypot(cx_e - cx_n, cy_e - cy_n)
                                    if distancia < min_dist_duplicado:
                                        es_duplicado = True
                                        break
                            if not es_duplicado:
                                cuadrados_encontrados.append((cnt, area))

    cuadrados_encontrados = sorted(cuadrados_encontrados, key=lambda x: x[1], reverse=True)
    return [c[0] for c in cuadrados_encontrados[:max_n]]


def contorno_a_centroide_float(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos en imagen (y hacia abajo) como TL, TR, BR, BL para un cuadrilátero tipo rectángulo.
    Heurística estándar (suma x+y y diferencia x-y); bajo fuerte perspectiva puede fallar, por eso
    se combina con permutaciones completas.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    tl_i = int(np.argmin(s))
    br_i = int(np.argmax(s))
    remaining = [i for i in range(4) if i not in (tl_i, br_i)]
    if len(remaining) != 2:
        return pts.copy()
    dif = pts[remaining, 0] - pts[remaining, 1]
    tr_i = remaining[int(np.argmax(dif))]
    bl_i = remaining[int(np.argmin(dif))]
    return np.array([pts[tl_i], pts[tr_i], pts[br_i], pts[bl_i]], dtype=np.float32)


def _homography_reproj_sum_err(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    warped = cv2.perspectiveTransform(src.reshape(1, 4, 2), H).reshape(4, 2)
    return float(np.linalg.norm(warped - dst, axis=1).sum())


def best_homography_from_permutations(
    pts_camara_detectados: list[np.ndarray],
    pts_proyeccion_dst: np.ndarray,
    max_total_error: float = 80.0,
):
    """
    pts_proyeccion_dst: (4,2) float32, orden TL, TR, BR, BL (horario en pantalla).
    Prueba orden geométrico + rotaciones, su reflejo cíclico, y las 24 permutaciones; elige menor error.
    """
    if len(pts_camara_detectados) != 4:
        return None, None, float("inf")

    det = np.stack(pts_camara_detectados, axis=0).astype(np.float32)
    dst = np.asarray(pts_proyeccion_dst, dtype=np.float32)
    best_err = float("inf")
    best_H = None
    best_src = None

    def consider(src_candidate: np.ndarray):
        nonlocal best_err, best_H, best_src
        try:
            H = cv2.getPerspectiveTransform(src_candidate, dst)
        except cv2.error:
            return
        err = _homography_reproj_sum_err(H, src_candidate, dst)
        if err < best_err:
            best_err = err
            best_H = H
            best_src = src_candidate.copy()

    src_geo = order_quad_tl_tr_br_bl(det)
    for k in range(4):
        consider(np.roll(src_geo, k, axis=0))
    src_rev = src_geo[::-1].copy()
    for k in range(4):
        consider(np.roll(src_rev, k, axis=0))

    for perm in permutations(range(4)):
        src = np.array([det[i] for i in perm], dtype=np.float32)
        consider(src)

    if best_H is None or best_err > max_total_error:
        return None, None, best_err

    return best_H, best_src, best_err


def depth_roi_from_quad(pts_camara: np.ndarray, margin: int = 18):
    """Bounding axis-aligned de los 4 puntos cámara, recortado a 640x480."""
    xs = pts_camara[:, 0]
    ys = pts_camara[:, 1]
    xw_min = int(max(0, np.floor(xs.min() - margin)))
    xw_max = int(min(640, np.ceil(xs.max() + margin)))
    yw_min = int(max(0, np.floor(ys.min() - margin)))
    yw_max = int(min(480, np.ceil(ys.max() + margin)))
    if xw_max <= xw_min or yw_max <= yw_min:
        return None
    return xw_min, xw_max, yw_min, yw_max


def calibrar_mesa_y_detectar_toques(device):
    view_width = 1280
    view_height = 800

    cv2.namedWindow("Proyeccion", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Proyeccion", 1920, 0)
    cv2.setWindowProperty("Proyeccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not device.has_sensor(openni2.SENSOR_COLOR):
        print("=" * 60)
        print("ERROR: El dispositivo no tiene sensor de color disponible")
        print("=" * 60)
        return

    color_stream = None
    try:
        color_stream = device.create_color_stream()
        color_stream.start()
        print("✓ Stream de color iniciado correctamente")
    except Exception as e:
        print("=" * 60)
        print("ERROR: No se pudo iniciar el stream de color")
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        return

    try:
        depth_stream = device.create_depth_stream()
        depth_stream.start()
        print("✓ Stream de profundidad iniciado correctamente")
    except Exception as e:
        print("=" * 60)
        print("ERROR: No se pudo iniciar el stream de profundidad")
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        if color_stream is not None:
            try:
                color_stream.stop()
            except Exception:
                pass
        return

    depth_frame = depth_stream.read_frame()
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
    depth_data = cv2.flip(depth_data, 1)

    homography_matrix = None
    xw_min = xw_max = yw_min = yw_max = None
    xv_min = xv_max = yv_min = yv_max = None
    pts_camara_final = None
    pts_proyeccion_final = None
    calibracion_completada = False

    while True:
        proyeccion, centers_proj, xv_min, xv_max, yv_min, yv_max, _S = proyectar_cuatro_cuadrados(
            view_width, view_height
        )
        cv2.imshow("Proyeccion", proyeccion)
        cv2.waitKey(2000)

        frame = color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=frame_data)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 1)
        cuadrados_detectados = detectar_cuadrados_hasta_n(frame_bgr, max_n=4)

        frame_debug = frame_bgr.copy()
        for i, cuadrado in enumerate(cuadrados_detectados):
            cv2.drawContours(frame_debug, [cuadrado], -1, (0, 255, 0), 2)
            c = contorno_a_centroide_float(cuadrado)
            if c is not None:
                cv2.circle(frame_debug, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame_debug,
                    f"{i+1}",
                    (int(c[0]) - 8, int(c[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        info_text = f"Cuadrados detectados: {len(cuadrados_detectados)}/4"
        cv2.putText(frame_debug, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if len(cuadrados_detectados) < 4:
            cv2.putText(
                frame_debug,
                "Visibles los 4 cuadrados blancos en las esquinas",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            print(f"Esperando cuadrados... Detectados: {len(cuadrados_detectados)}/4")

        cv2.imshow("Camara", frame_debug)

        if len(cuadrados_detectados) >= 4:
            puntos = []
            for cuadrado in cuadrados_detectados[:4]:
                c = contorno_a_centroide_float(cuadrado)
                if c is not None:
                    puntos.append(c)

            if len(puntos) == 4:
                H, src_ord, err = best_homography_from_permutations(puntos, centers_proj)
                if H is not None:
                    homography_matrix = H
                    pts_camara_final = src_ord
                    pts_proyeccion_final = centers_proj.copy()
                    roi_depth = depth_roi_from_quad(src_ord)
                    if roi_depth is None:
                        print("[ERROR] ROI de profundidad inválida")
                    else:
                        xw_min, xw_max, yw_min, yw_max = roi_depth
                        print(f"[OK] Homografía 4 puntos, error total reproyección ≈ {err:.2f} px")
                        for i, name in enumerate(["TL", "TR", "BR", "BL"]):
                            print(f"    {name} cam={src_ord[i]} -> proj={centers_proj[i]}")

                        cv2.polylines(
                            proyeccion,
                            [centers_proj.astype(np.int32).reshape(1, 4, 2)],
                            True,
                            (0, 255, 0),
                            2,
                        )
                        cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
                        cv2.imshow("Proyeccion", proyeccion)

                        cv2.polylines(
                            frame_bgr,
                            [src_ord.astype(np.int32).reshape(1, 4, 2)],
                            True,
                            (255, 0, 0),
                            2,
                        )
                        cv2.imshow("Camara", frame_bgr)
                        print("Calibración completada (4 puntos).")
                        calibracion_completada = True
                        cv2.waitKey(5000)
                        break
                else:
                    print(
                        f"[INFO] Permutaciones no convergen (error {err:.1f}); "
                        "ajusta iluminación o alinea la cámara con los cuatro marcadores."
                    )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        depth_stream.stop()
    except Exception:
        pass

    if not calibracion_completada:
        print("=" * 60)
        print("ERROR: La calibración no se completó")
        print("=" * 60)
        print("Se necesitan exactamente 4 cuadrados detectados y un error de reproyección bajo.")
        print("=" * 60)
        color_stream.stop()
        try:
            depth_stream.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        return

    dmax_map = calculate_dmax(
        device,
        (xw_min, yw_min, xw_max - xw_min, yw_max - yw_min),
        xv_min,
        yv_min,
        xv_max,
        yv_max,
    )

    calibration_quad_camera = pts_camara_final.astype(float).tolist()
    calibration_quad_projection = pts_proyeccion_final.astype(float).tolist()

    coordenadas = {
        "calibration_method": "four_point_squares",
        "xv_min": xv_min,
        "xv_max": xv_max,
        "yv_min": yv_min,
        "yv_max": yv_max,
        "xw_min": xw_min,
        "xw_max": xw_max,
        "yw_min": yw_min,
        "yw_max": yw_max,
        "homography_matrix": homography_matrix.tolist() if homography_matrix is not None else None,
        "calibration_quad_camera": calibration_quad_camera,
        "calibration_quad_projection": calibration_quad_projection,
    }

    cfg_path = get_coordenadas_path()
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as file:
        json.dump(coordenadas, file, indent=4)
    print("✓ Coordenadas guardadas exitosamente.")

    if xw_min is None:
        color_stream.stop()
        cv2.destroyAllWindows()
        return

    print("Iniciando detección de toques...")
    user_response = None

    def on_yes():
        nonlocal user_response
        user_response = "Sí"
        app.destroy()

    def on_no():
        nonlocal user_response
        user_response = "No"
        app.destroy()

    ctk.set_appearance_mode("Dark")
    app = ctk.CTk()
    app.title("Verificación de Detección")
    app.geometry("400x200")
    try:
        from src.core.font_utils import get_ubuntu_font_path_for_customtkinter

        ubuntu_font = get_ubuntu_font_path_for_customtkinter()
        font_family = ubuntu_font if ubuntu_font else "Arial"
    except Exception:
        font_family = "Arial"
    label = ctk.CTkLabel(
        app, text="¿Desea proceder con la detección de toques?", font=(font_family, 16)
    )
    label.pack(pady=20)
    button_frame = ctk.CTkFrame(app, fg_color="transparent")
    button_frame.pack(pady=10)
    ctk.CTkButton(button_frame, text="Sí", command=on_yes).pack(side="left", padx=10)
    ctk.CTkButton(button_frame, text="No", command=on_no).pack(side="left", padx=10)
    app.mainloop()

    if user_response == "Sí":
        if not isinstance(dmax_map, np.ndarray):
            print("[ERROR] dmax_map no es un array válido")
            color_stream.stop()
            try:
                depth_stream.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()
            return

        dmax_map = dmax_map.astype(np.int16) - 4
        dmin_map = dmax_map.astype(np.int16) - 10

        roi_height = yw_max - yw_min
        roi_width = xw_max - xw_min
        if dmax_map.shape != (roi_height, roi_width):
            print(f"[ADVERTENCIA] Dimensiones no coinciden: dmax_map={dmax_map.shape}, ROI=({roi_height}, {roi_width})")
            print("[INFO] Redimensionando dmax_map...")
            dmax_map = (
                cv2.resize(dmax_map.astype(np.float32), (roi_width, roi_height), interpolation=cv2.INTER_NEAREST).astype(
                    np.int16
                )
            )
            dmin_map = (
                cv2.resize(dmin_map.astype(np.float32), (roi_width, roi_height), interpolation=cv2.INTER_NEAREST).astype(
                    np.int16
                )
            )

        try:
            depth_stream.start()
        except Exception:
            pass

        print("Iniciando detección de toques...")
        print(f"  ROI: xw_min={xw_min}, xw_max={xw_max}, yw_min={yw_min}, yw_max={yw_max}")
        previous_roi = None
        touch_history = []
        vibration_threshold = 20
        touch_duration_threshold = 2
        max_history_frames = 5

        proyeccion = np.zeros((view_height, view_width, 3), dtype=np.uint8)
        cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)
        y_text = 30
        for texto in [
            "Toca la mesa para ver los puntos",
            "Presiona 'q' para salir",
        ]:
            cv2.putText(
                proyeccion,
                texto,
                (xv_min + 10, yv_min + y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_text += 30

        while True:
            depth_frame = depth_stream.read_frame()
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
            depth_data = cv2.flip(depth_data, 1)

            try:
                color_frame = color_stream.read_frame()
                frame_data = color_frame.get_buffer_as_uint8()
                frame_array = np.ndarray(
                    (color_frame.height, color_frame.width, 3), dtype=np.uint8, buffer=frame_data
                )
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.flip(frame_bgr, 1)
            except Exception:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

            cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 0, 0), -1)
            cv2.rectangle(proyeccion, (xv_min, yv_min), (xv_max, yv_max), (0, 255, 0), 2)

            depth_roi = depth_data[yw_min:yw_max, xw_min:xw_max]
            frame_roi = frame_bgr[yw_min:yw_max, xw_min:xw_max]

            if previous_roi is not None:
                roi_diff = cv2.absdiff(depth_roi, previous_roi)
                vibration_mask = cv2.threshold(roi_diff, vibration_threshold, 255, cv2.THRESH_BINARY)[1]
                vibration_mask = cv2.medianBlur(vibration_mask, ksize=3)
                depth_roi[vibration_mask > 200] = previous_roi[vibration_mask > 200]

            previous_roi = depth_roi.copy()

            if depth_roi.shape != dmax_map.shape:
                print(f"[ERROR] Dimensiones no coinciden: depth_roi={depth_roi.shape}, dmax_map={dmax_map.shape}")
                break

            touch_mask = np.logical_and(depth_roi > dmin_map, depth_roi < dmax_map).astype(np.uint8) * 255
            touch_mask_filtered = cv2.medianBlur(touch_mask, ksize=3)
            kernel = np.ones((2, 2), np.uint8)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_OPEN, kernel)
            touch_mask_filtered = cv2.morphologyEx(touch_mask_filtered, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask_filtered, connectivity=8)
            min_size = 50
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    touch_mask_filtered[labels == i] = 0

            touch_history.append(touch_mask_filtered)
            if len(touch_history) > max_history_frames:
                touch_history.pop(0)

            accumulated_mask = np.sum(touch_history, axis=0)
            accumulated_mask = np.clip(accumulated_mask, 0, 255).astype(np.uint8)
            threshold_value = max(1, touch_duration_threshold) * 255
            _, final_touch_mask = cv2.threshold(accumulated_mask, threshold_value, 255, cv2.THRESH_BINARY)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_touch_mask, connectivity=8)
            if num_labels <= 1:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    touch_mask_filtered, connectivity=8
                )
                if num_labels <= 1:
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(touch_mask, connectivity=8)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    mask_component = (labels == i).astype(np.uint8)
                    M = cv2.moments(mask_component)
                    if M["m00"] > 0:
                        cx_standard = M["m10"] / M["m00"]
                        cy_standard = M["m01"] / M["m00"]
                        depth_inverse = (dmax_map - depth_roi).astype(np.float32)
                        depth_inverse[depth_inverse < 0] = 0
                        depth_weighted = mask_component.astype(np.float32) * depth_inverse
                        M_weighted = cv2.moments(depth_weighted)
                        if M_weighted["m00"] > 0:
                            cx_weighted = M_weighted["m10"] / M_weighted["m00"]
                            cy_weighted = M_weighted["m01"] / M_weighted["m00"]
                            x_touch = int(0.7 * cx_weighted + 0.3 * cx_standard)
                            y_touch = int(0.7 * cy_weighted + 0.3 * cy_standard)
                        else:
                            x_touch = int(cx_standard)
                            y_touch = int(cy_standard)
                    else:
                        centroid = centroids[i]
                        x_touch, y_touch = int(centroid[0]), int(centroid[1])

                    x_camara = xw_min + x_touch
                    y_camara = yw_min + y_touch

                    if homography_matrix is not None:
                        point_camara = np.array([[[x_camara, y_camara]]], dtype=np.float32)
                        point_proyeccion = cv2.perspectiveTransform(point_camara, homography_matrix)
                        x_viewport = int(point_proyeccion[0][0][0])
                        y_viewport = int(point_proyeccion[0][0][1])
                    else:
                        sx = float(xv_max - xv_min) / depth_roi.shape[1]
                        sy = float(yv_max - yv_min) / depth_roi.shape[0]
                        x_viewport = int(xv_min + (x_touch * sx))
                        y_viewport = int(yv_min + (y_touch * sy))

                    x_viewport = int(np.clip(x_viewport, 0, view_width - 1))
                    y_viewport = int(np.clip(y_viewport, 0, view_height - 1))

                    cv2.circle(frame_roi, (x_touch, y_touch), 8, (255, 0, 0), -1)
                    cv2.circle(frame_roi, (x_touch, y_touch), 10, (255, 255, 255), 2)
                    cv2.circle(proyeccion, (x_viewport, y_viewport), 10, (0, 0, 255), -1)
                    cv2.circle(proyeccion, (x_viewport, y_viewport), 12, (255, 255, 255), 2)
                    coord_text = f"({x_viewport}, {y_viewport})"
                    text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        proyeccion,
                        (x_viewport + 15, y_viewport - 5),
                        (x_viewport + 15 + text_size[0] + 5, y_viewport + text_size[1] + 5),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        proyeccion,
                        coord_text,
                        (x_viewport + 18, y_viewport + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    print(f"[TOQUE] Coordenadas: ({x_viewport}, {y_viewport})")

            cv2.putText(
                proyeccion,
                f"Toques: {num_labels - 1}",
                (xv_min + 10, yv_max - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            full_mask = np.zeros_like(depth_data, dtype=np.uint8)
            full_mask[yw_min:yw_max, xw_min:xw_max] = final_touch_mask
            full_mask_filtered = np.zeros_like(depth_data, dtype=np.uint8)
            full_mask_filtered[yw_min:yw_max, xw_min:xw_max] = touch_mask_filtered

            cv2.imshow("Proyeccion", proyeccion)
            cv2.imshow("Mascara de Toque", full_mask)
            cv2.imshow("Mascara Filtrada", full_mask_filtered)
            cv2.imshow("Camara", frame_bgr)
            cv2.imshow("ROI", frame_roi)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    color_stream.stop()
    depth_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibración 4 puntos mesa / Kinect")
    parser.add_argument(
        "--solo-profundidad",
        action="store_true",
        help="Solo regenera dmax_map.txt con la ROI del JSON (misma opción que calibrate_area).",
    )
    args = parser.parse_args()

    openni2_paths = []
    env_path = os.environ.get("OPENNI2_PATH")
    if env_path:
        openni2_paths.append(env_path)
        openni2_paths.append(os.path.join(env_path, "Redist"))
    openni2_paths.extend(
        [
            "C:/Program Files/OpenNI2/Redist",
            "C:/Program Files/OpenNI2",
        ]
    )
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    if program_files:
        openni2_paths.append(os.path.join(program_files, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files, "OpenNI2"))
        openni2_paths.append(os.path.join(program_files, "OpenNI2", "Driver"))
    if program_files_x86:
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2", "Redist"))
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2"))
        openni2_paths.append(os.path.join(program_files_x86, "OpenNI2", "Driver"))
    openni2_paths.extend(
        [
            "C:/Program Files (x86)/OpenNI2/Redist",
            "C:/Program Files (x86)/OpenNI2",
            "C:/OpenNI2/Redist",
            "C:/OpenNI2",
            "D:/Program Files/OpenNI2/Redist",
            "D:/Program Files/OpenNI2",
            "D:/Program Files (x86)/OpenNI2/Redist",
            "D:/Program Files (x86)/OpenNI2",
            "D:/OpenNI2/Redist",
            "D:/OpenNI2",
        ]
    )
    openni2_paths = list(dict.fromkeys([p for p in openni2_paths if p]))

    openni2_initialized = False
    last_exception = None
    for path in openni2_paths:
        if os.path.exists(path):
            try:
                openni2.initialize(path)
                openni2_initialized = True
                print(f"✓ OpenNI2 inicializado desde: {path}")
                break
            except Exception as e:
                last_exception = e
                continue

    if not openni2_initialized:
        print("ERROR: No se pudo inicializar OpenNI2. Ver OPENNI2_PATH y SDK.")
        if last_exception:
            print(last_exception)
        exit(1)

    try:
        device = openni2.Device.open_any()
        device_info = device.get_device_info()
        print(f"✓ Dispositivo: {device_info.name.decode('utf-8') if device_info.name else 'Desconocido'}")
    except Exception as e:
        print(f"ERROR abriendo dispositivo: {e}")
        openni2.unload()
        exit(1)

    try:
        if args.solo_profundidad:
            run_solo_profundidad(device)
        else:
            calibrar_mesa_y_detectar_toques(device)
    except KeyboardInterrupt:
        print("\n\nCalibración cancelada por el usuario.")
    except Exception as e:
        print(f"\nERROR durante la calibración: {e}")
        import traceback

        traceback.print_exc()
    finally:
        openni2.unload()
