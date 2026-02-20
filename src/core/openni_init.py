"""
OpenNI2 initialization utilities for MagicboARd
"""
import os
from openni import openni2


def initialize_openni2():
    """
    Inicializa OpenNI2 buscando en múltiples ubicaciones comunes.
    
    Returns:
        tuple: (device, success) donde device es el dispositivo OpenNI2 o None si falla
    """
    # Inicializar OpenNI2 - Buscar en múltiples ubicaciones comunes
    openni2_paths = []
    
    # 1. Primero verificar si hay una variable de entorno configurada
    env_path = os.environ.get("OPENNI2_PATH")
    if env_path:
        openni2_paths.append(env_path)
        openni2_paths.append(os.path.join(env_path, "Redist"))
    
    # 2. Agregar la ruta más común primero (prioridad)
    openni2_paths.extend([
        "C:/Program Files/OpenNI2/Redist",
        "C:/Program Files/OpenNI2",
    ])
    
    # 3. Agregar ubicaciones comunes en Windows usando variables de entorno
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
    
    # 4. Agregar otras rutas absolutas comunes
    openni2_paths.extend([
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
    ])
    
    # Eliminar duplicados y rutas vacías
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
        print("=" * 60)
        print("ERROR: No se pudo encontrar OpenNI2 SDK")
        print("=" * 60)
        print("\nSOLUCIONES:")
        print("\n1. Instala OpenNI2 SDK desde:")
        print("   https://structure.io/openni")
        print("   Descarga: OpenNI 2 SDK for Windows")
        print("\n2. O configura la variable de entorno OPENNI2_PATH:")
        print("   setx OPENNI2_PATH \"C:\\ruta\\a\\OpenNI2\\Redist\"")
        print("\n3. O especifica la ruta manualmente editando src/core/openni_init.py")
        print("   (agrega tu ruta al inicio de la lista openni2_paths)")
        print("\nUbicaciones buscadas:")
        for path in openni2_paths[:10]:  # Mostrar solo las primeras 10
            status = "✓ Existe" if os.path.exists(path) else "✗ No existe"
            print(f"  {status}: {path}")
        if len(openni2_paths) > 10:
            print(f"  ... y {len(openni2_paths) - 10} ubicaciones más")
        if last_exception:
            print(f"\nÚltimo error: {last_exception}")
        print("=" * 60)
        return None, False
    
    device = openni2.Device.open_any()
    return device, True

