"""
Calibration utilities for MagicboARd
"""
import numpy as np
import os


def load_and_validate_dmax_map(coordenadas):
    """
    Carga y valida el archivo dmax_map.txt contra las coordenadas proporcionadas.
    
    Args:
        coordenadas: Diccionario con las coordenadas de calibración
    
    Returns:
        tuple: (dmax_map_reshaped, w, h) si es exitoso, (None, None, None) si hay error
    """
    # Get the path relative to src directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "dmax_map.txt")
    
    try:
        # Cargar el archivo dmax_map.txt
        dmax_map = np.loadtxt(config_path, dtype=np.uint16)
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'src/config/dmax_map.txt'.")
        print("Por favor, ejecuta la calibración primero.")
        return None, None, None
    except ValueError as e:
        print(f"ERROR: El archivo 'dmax_map.txt' tiene un formato inválido: {e}")
        return None, None, None
    
    # Calcular dimensiones esperadas
    xw_min = coordenadas.get("xw_min", 0)
    xw_max = coordenadas.get("xw_max", 0)
    yw_min = coordenadas.get("yw_min", 0)
    yw_max = coordenadas.get("yw_max", 0)
    
    w = xw_max - xw_min
    h = yw_max - yw_min
    
    expected_size = w * h
    actual_size = dmax_map.size
    
    # Validar dimensiones
    if actual_size != expected_size:
        print("=" * 70)
        print("ERROR: Incompatibilidad de dimensiones entre dmax_map.txt y coordenadas")
        print("=" * 70)
        print(f"Dimensiones esperadas: {w} x {h} = {expected_size} elementos")
        print(f"Dimensiones del archivo: {actual_size} elementos")
        
        # Intentar encontrar dimensiones posibles
        posibles_dimensiones = []
        for i in range(300, 500):
            if actual_size % i == 0:
                j = actual_size // i
                posibles_dimensiones.append((i, j))
        
        if posibles_dimensiones:
            print(f"\nDimensiones posibles del archivo:")
            for dim in posibles_dimensiones[:3]:
                print(f"  - {dim[0]} x {dim[1]} o {dim[1]} x {dim[0]}")
        
        print("\nSOLUCIÓN:")
        print("1. Ejecuta la calibración para regenerar dmax_map.txt con las dimensiones correctas:")
        print("   python src/core/calibrate_area.py")
        print("\n2. O ajusta las coordenadas en 'src/config/ultima_configuracion_coordenadas.json'")
        print("   para que coincidan con las dimensiones del archivo actual.")
        print("=" * 70)
        return None, None, None
    
    try:
        # Hacer reshape
        dmax_map_reshaped = dmax_map.reshape((h, w))
        return dmax_map_reshaped, w, h
    except ValueError as e:
        print(f"ERROR al hacer reshape del dmax_map: {e}")
        return None, None, None

