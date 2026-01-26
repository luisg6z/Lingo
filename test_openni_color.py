# -*- coding: utf-8 -*-
import sys
import io

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openni import openni2

try:
    # Usa exactamente la ruta que te funciona
    openni2.initialize("C:/Program Files/OpenNI2/Redist")
    print("[OK] OpenNI2 inicializado correctamente")
    
    dev = openni2.Device.open_any()ERROR durante la calibración: name 'touch_mask_final' is not defined
Traceback (most recent call last):
  File "C:\Users\CHARBEL\Desktop\Lingo\calibrate_area.py", line 585, in <module>
    calibrar_mesa_y_detectar_toques(device)
  File "C:\Users\CHARBEL\Desktop\Lingo\calibrate_area.py", line 451, in calibrar_mesa_y_detectar_toques
    full_mask[yw_min_escalado:yw_max, xw_min:xw_max_escalado] = touch_mask_final
NameError: name 'touch_mask_final' is not defined
    print("[OK] Dispositivo abierto")
    
    # Verificar sensores disponibles
    color_available = dev.has_sensor(openni2.SENSOR_COLOR)
    depth_available = dev.has_sensor(openni2.SENSOR_DEPTH)
    
    print("=" * 60)
    print("RESULTADO:")
    print(f"Color disponible: {color_available}")
    print(f"Depth disponible: {depth_available}")
    print("=" * 60)
    
    if color_available:
        print("\n[OK] El driver SI soporta color")
        print("El problema puede ser que otro programa este usando el Kinect")
    else:
        print("\n[ERROR] El driver NO soporta color")
        print("Necesitas instalar un driver compatible con color para Kinect v1")
    
    openni2.unload()
    
except Exception as e:
    print("=" * 60)
    print("ERROR al ejecutar el test:")
    print(f"{e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
