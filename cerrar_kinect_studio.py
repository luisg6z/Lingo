# -*- coding: utf-8 -*-
"""
Script para cerrar automáticamente procesos de Kinect que puedan estar bloqueando el dispositivo
"""
import sys
import io
import subprocess
import time

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def cerrar_procesos_kinect():
    """Cierra procesos de Kinect que puedan estar bloqueando el dispositivo"""
    procesos_kinect = [
        "KinectStudio",
        "KinectStudio.exe",
        "KinectExplorer",
        "KinectExplorer.exe",
        "KinectDeveloperKit-Browser",  # Kinect Developer Browser
        "KinectDeveloperKit-Browser.exe",
        "KinectForWindows",  # Otros procesos comunes
        "KinectForWindows.exe",
        "KSStudio",  # Versiones alternativas
        "KSStudio.exe",
        "ColorBasics",  # Aplicación Color Basics
        "ColorBasics.exe",
        "DepthBasics",  # Otras aplicaciones básicas
        "DepthBasics.exe",
        "BodyBasics",
        "BodyBasics.exe",
    ]
    
    print("=" * 60)
    print("CERRANDO PROCESOS DE KINECT")
    print("=" * 60)
    
    # Primero, buscar procesos que contengan "kinect" en el nombre (insensible a mayúsculas)
    print("\nBuscando procesos relacionados con Kinect...")
    procesos_encontrados = []
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        lines = result.stdout.split('\n')
        for line in lines:
            if 'kinect' in line.lower():
                # Extraer el nombre del proceso (primera columna)
                parts = line.split()
                if parts:
                    proceso_name = parts[0]
                    if proceso_name not in procesos_encontrados:
                        procesos_encontrados.append(proceso_name)
                        print(f"[ENCONTRADO] {proceso_name}")
    except Exception as e:
        print(f"[INFO] No se pudieron listar procesos: {e}")
    
    # Combinar procesos conocidos con los encontrados
    todos_los_procesos = list(set(procesos_kinect + procesos_encontrados))
    
    procesos_cerrados = []
    
    for proceso in todos_los_procesos:
        try:
            # Intentar cerrar el proceso usando taskkill
            result = subprocess.run(
                ['taskkill', '/F', '/IM', proceso, '/T'],
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                print(f"[OK] Proceso cerrado: {proceso}")
                procesos_cerrados.append(proceso)
            elif "no se encuentra" in result.stderr.lower() or "not found" in result.stderr.lower():
                # El proceso no está ejecutándose, está bien
                pass
            else:
                # Intentar sin /T (sin forzar procesos hijos)
                try:
                    result2 = subprocess.run(
                        ['taskkill', '/F', '/IM', proceso],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        encoding='utf-8',
                        errors='ignore'
                    )
                    if result2.returncode == 0:
                        print(f"[OK] Proceso cerrado: {proceso}")
                        procesos_cerrados.append(proceso)
                except:
                    pass
        except subprocess.TimeoutExpired:
            print(f"[ADVERTENCIA] Timeout al intentar cerrar {proceso}")
        except Exception as e:
            print(f"[INFO] {proceso}: {str(e)[:50]}")
    
    if procesos_cerrados:
        print(f"\n[OK] Se cerraron {len(procesos_cerrados)} proceso(s)")
        print("[INFO] Espera 3 segundos antes de ejecutar tu script...")
        time.sleep(3)
    else:
        print("\n[INFO] No se encontraron procesos de Kinect ejecutándose")
    
    print("=" * 60)
    return len(procesos_cerrados) > 0

if __name__ == "__main__":
    cerrar_procesos_kinect()

