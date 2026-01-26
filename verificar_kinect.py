# -*- coding: utf-8 -*-
"""
Script para verificar si el Kinect está disponible antes de ejecutar calibrate_area.py
"""
import sys
import io

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import os
from openni import openni2

def verificar_procesos_kinect():
    """Verifica si hay procesos comunes de Kinect ejecutándose"""
    procesos_kinect = [
        "KinectStudio",
        "KinectStudio.exe",
        "KinectExplorer",
        "KinectExplorer.exe",
        "KinectDeveloperKit-Browser",  # Kinect Developer Browser
        "KinectDeveloperKit-Browser.exe",
        "KinectForWindows",
        "KinectForWindows.exe",
        "KSStudio",
        "KSStudio.exe",
    ]
    
    print("Verificando procesos de Kinect...")
    procesos_encontrados = []
    
    try:
        # Ejecutar tasklist para ver procesos
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        procesos = result.stdout.lower()
        
        for proceso in procesos_kinect:
            if proceso.lower() in procesos:
                procesos_encontrados.append(proceso)
    except:
        pass
    
    return procesos_encontrados

def main():
    print("=" * 60)
    print("VERIFICACION DE KINECT")
    print("=" * 60)
    
    # 1. Verificar procesos
    procesos = verificar_procesos_kinect()
    if procesos:
        print("\n[ADVERTENCIA] Se encontraron procesos de Kinect ejecutandose:")
        for p in procesos:
            print(f"  - {p}")
        print("\nCIERRA estos procesos antes de ejecutar calibrate_area.py")
        print("Puedes cerrarlos desde el Administrador de tareas (Ctrl+Shift+Esc)")
    else:
        print("\n[OK] No se encontraron procesos de Kinect ejecutandose")
    
    # 2. Verificar OpenNI2
    print("\n" + "=" * 60)
    print("VERIFICANDO OPENNI2 Y DISPOSITIVO")
    print("=" * 60)
    
    # Buscar OpenNI2 en múltiples ubicaciones
    openni2_paths = [
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
        os.path.join(os.environ.get("ProgramFiles", ""), "OpenNI2", "Redist"),
    ]
    
    openni2_inicializado = False
    for path in openni2_paths:
        if os.path.exists(path):
            try:
                openni2.initialize(path)
                openni2_inicializado = True
                print(f"[OK] OpenNI2 inicializado desde: {path}")
                break
            except:
                continue
    
    if not openni2_inicializado:
        print("[ERROR] No se pudo inicializar OpenNI2")
        return
    
    try:
        
        dev = openni2.Device.open_any()
        print("[OK] Dispositivo encontrado")
        
        color_available = dev.has_sensor(openni2.SENSOR_COLOR)
        depth_available = dev.has_sensor(openni2.SENSOR_DEPTH)
        
        print(f"\nColor disponible: {color_available}")
        print(f"Depth disponible: {depth_available}")
        
        if color_available and depth_available:
            print("\n[OK] El dispositivo tiene sensores disponibles")
            
            # Intentar iniciar los streams realmente para verificar que están libres
            print("\nProbando acceso a los streams...")
            try:
                color_stream = dev.create_color_stream()
                color_stream.start()
                print("[OK] Stream de COLOR iniciado correctamente")
                color_stream.stop()
            except Exception as e:
                print(f"[ERROR] No se pudo iniciar stream de COLOR: {e}")
                print("        Esto significa que otro programa está usando el stream de color")
                color_available = False
            
            try:
                depth_stream = dev.create_depth_stream()
                depth_stream.start()
                print("[OK] Stream de PROFUNDIDAD iniciado correctamente")
                depth_stream.stop()
            except Exception as e:
                print(f"[ERROR] No se pudo iniciar stream de PROFUNDIDAD: {e}")
                print("        Esto significa que otro programa está usando el stream de profundidad")
                depth_available = False
            
            if color_available and depth_available:
                print("\n[OK] El dispositivo esta completamente listo para usar")
                if procesos:
                    print("\n[IMPORTANTE] Pero primero CIERRA los procesos de Kinect")
                else:
                    print("\n[OK] Puedes ejecutar calibrate_area.py ahora")
            else:
                print("\n[ERROR] El dispositivo NO está disponible")
                print("\nSOLUCIONES:")
                print("1. Cierra TODAS las aplicaciones que usan el Kinect")
                print("2. Ejecuta: python cerrar_kinect_studio.py")
                print("3. Desconecta y reconecta el cable USB del Kinect")
                print("4. Espera 5 segundos después de cerrar las aplicaciones")
                print("5. Si persiste, reinicia tu computadora")
        else:
            print("\n[ERROR] El dispositivo no tiene todos los sensores disponibles")
        
        openni2.unload()
        
    except Exception as e:
        print(f"\n[ERROR] No se pudo verificar el dispositivo: {e}")
        print("\nAsegurate de que:")
        print("1. El Kinect este conectado")
        print("2. OpenNI2 este instalado correctamente")
        print("3. No haya otros programas usando el Kinect")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

