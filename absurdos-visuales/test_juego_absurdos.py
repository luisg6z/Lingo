"""
Test file para probar el juego de absurdos visuales.
Muestra una imagen y permite probar la evaluación de respuestas usando sentence-transformers.

INSTRUCCIONES:
1. Asegúrate de tener instalado sentence-transformers:
   uv pip install sentence-transformers
   o
   pip install sentence-transformers

2. Ejecuta el test:
   python absurdos-visuales/test_juego_absurdos.py

3. El test mostrará:
   - Una imagen del absurdo seleccionado
   - Pruebas automáticas con diferentes respuestas
   - Modo interactivo para probar tus propias respuestas

4. El modelo se descargará automáticamente la primera vez (puede tardar unos minutos)
"""

import cv2
import numpy as np
import os
import json
import random

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers no está instalado.")
    print("Por favor instálalo con: uv pip install sentence-transformers")
    print("O con: pip install sentence-transformers")
    exit(1)


def test_juego_absurdos():
    """
    Función de prueba que muestra una imagen y permite probar respuestas.
    """
    print("=" * 60)
    print("TEST DEL JUEGO DE ABSURDOS VISUALES")
    print("=" * 60)
    
    # Cargar configuración de absurdos
    json_path = "absurdos-visuales/absurdos_logicos/config/absurdos.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        absurdos_list = datos.get("absurdos", [])
        if not absurdos_list:
            print("Error: No se encontraron absurdos en el archivo JSON")
            return
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {json_path} no es un JSON válido")
        return
    
    # Seleccionar un absurdo (o usar el primero para testing)
    absurdo_actual = absurdos_list[0]  # Usar el primero para testing consistente
    imagen_nombre = absurdo_actual.get("imagen", "")
    
    print(f"\nAbsurdo seleccionado:")
    print(f"  ID: {absurdo_actual.get('id')}")
    print(f"  Imagen: {imagen_nombre}")
    print(f"  Descripción: {absurdo_actual.get('descripcion')}")
    print(f"  Objeto: {absurdo_actual.get('objeto')}")
    print(f"  Propiedad incorrecta: {absurdo_actual.get('propiedad_incorrecta')}")
    print(f"  Propiedades correctas: {absurdo_actual.get('propiedades_correctas')}")
    
    # Construir ruta de imagen
    assets_path = "absurdos-visuales/absurdos_logicos/assets"
    ruta_imagen = os.path.join(assets_path, imagen_nombre)
    
    # Función para normalizar nombres de archivo
    def normalizar_nombre(nombre):
        nombre_sin_ext = os.path.splitext(nombre)[0]
        nombre_normalizado = nombre_sin_ext.lower().replace("-", "").replace("_", "").replace(" ", "")
        return nombre_normalizado
    
    # Intentar encontrar la imagen con matching flexible
    if not os.path.exists(ruta_imagen):
        imagen_nombre_alt = imagen_nombre.replace("_", "-")
        ruta_imagen = os.path.join(assets_path, imagen_nombre_alt)
    
    if not os.path.exists(ruta_imagen):
        imagen_normalizada = normalizar_nombre(imagen_nombre)
        mejor_coincidencia = None
        
        if os.path.exists(assets_path):
            archivos_disponibles = os.listdir(assets_path)
            for archivo in archivos_disponibles:
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    archivo_normalizado = normalizar_nombre(archivo)
                    if imagen_normalizada == archivo_normalizado:
                        mejor_coincidencia = archivo
                        break
        
        if mejor_coincidencia:
            ruta_imagen = os.path.join(assets_path, mejor_coincidencia)
            print(f"Imagen encontrada por coincidencia: {mejor_coincidencia}")
    
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró la imagen {imagen_nombre} en {assets_path}")
        return
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    print(f"\n✓ Imagen cargada: {ruta_imagen}")
    
    # Mostrar imagen en una ventana
    window_name = "Test - Juego de Absurdos Visuales"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Redimensionar imagen para mostrar (mantener aspect ratio)
    display_width = 800
    img_height, img_width = imagen.shape[:2]
    aspect_ratio = img_width / img_height
    
    if aspect_ratio > 1:
        new_width = display_width
        new_height = int(display_width / aspect_ratio)
    else:
        new_height = display_width
        new_width = int(display_width * aspect_ratio)
    
    imagen_display = cv2.resize(imagen, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convertir a BGR si tiene canal alfa
    if len(imagen_display.shape) == 3 and imagen_display.shape[2] == 4:
        # Crear fondo blanco
        fondo = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        alpha = imagen_display[:, :, 3] / 255.0
        # OpenCV lee en BGR, así que usar los canales BGR directamente
        for c in range(3):
            fondo[:, :, c] = (alpha * imagen_display[:, :, c] + (1 - alpha) * fondo[:, :, c])
        imagen_display = fondo
    # Si ya es BGR (3 canales), no necesita conversión
    
    cv2.imshow(window_name, imagen_display)
    print(f"\n✓ Ventana de imagen abierta. Presiona cualquier tecla para continuar...")
    cv2.waitKey(0)
    
    # Cargar modelo de sentence-transformers
    print("\n" + "=" * 60)
    print("CARGANDO MODELO DE SENTENCE-TRANSFORMERS")
    print("=" * 60)
    model = None
    try:
        print("Cargando modelo 'paraphrase-multilingual-MiniLM-L12-v2'...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✓ Modelo cargado correctamente")
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        return
    
    # Cargar respuestas correctas desde JSON
    respuestas_json_path = "absurdos-visuales/absurdos_logicos/config/respuestas_correctas.json"
    respuestas_correctas = []
    try:
        with open(respuestas_json_path, "r", encoding="utf-8") as f:
            datos_respuestas = json.load(f)
            absurdo_id = absurdo_actual.get("id")
            imagen_actual = absurdo_actual.get("imagen", "")
            for respuesta_item in datos_respuestas.get("respuestas", []):
                if respuesta_item.get("id") == absurdo_id or respuesta_item.get("imagen") == imagen_actual:
                    respuestas_correctas = respuesta_item.get("respuestas_correctas", [])
                    break
        print(f"✓ Cargadas {len(respuestas_correctas)} respuestas correctas")
    except Exception as e:
        print(f"✗ Error al cargar respuestas correctas: {e}")
        return
    
    if not respuestas_correctas:
        print("✗ No se encontraron respuestas correctas")
        return
    
    print("\nRespuestas correctas de referencia:")
    for i, respuesta in enumerate(respuestas_correctas, 1):
        print(f"  {i}. {respuesta}")
    
    # Función para calcular similitud
    def calcular_similitud(texto_usuario, respuestas_correctas, model):
        """Calcula la similitud entre el texto del usuario y las respuestas correctas"""
        try:
            # Crear embeddings
            texto_usuario_embedding = model.encode([texto_usuario])[0]
            respuestas_embeddings = model.encode(respuestas_correctas)
            
            # Calcular similitud coseno
            def cosine_similarity_numpy(vec1, vec2_array):
                dot_products = np.dot(vec2_array, vec1)
                norm1 = np.linalg.norm(vec1)
                norms2 = np.linalg.norm(vec2_array, axis=1)
                return dot_products / (norm1 * norms2)
            
            similarities = cosine_similarity_numpy(texto_usuario_embedding, respuestas_embeddings)
            return similarities
        except Exception as e:
            print(f"Error al calcular similitud: {e}")
            return None
    
    # Probar con diferentes respuestas
    print("\n" + "=" * 60)
    print("PRUEBAS DE EVALUACIÓN DE RESPUESTAS")
    print("=" * 60)
    
    SIMILARITY_THRESHOLD = 0.7
    
    # Respuestas de prueba
    respuestas_prueba = [
        # Respuestas correctas (deberían pasar)
        "la manzana no es azul, debe ser roja",
        "está mal porque las manzanas son rojas",
        "las manzanas no son azules",
        # Respuestas parcialmente correctas
        "la manzana es azul y eso está mal",
        "manzana azul incorrecto",
        # Respuestas incorrectas
        "veo una manzana",
        "es una fruta",
        "me gusta la manzana",
    ]
    
    print(f"\nUmbral de similitud: {SIMILARITY_THRESHOLD}")
    print("\nProbando respuestas...\n")
    
    for i, respuesta_prueba in enumerate(respuestas_prueba, 1):
        print(f"\n{'─' * 60}")
        print(f"Prueba {i}: \"{respuesta_prueba}\"")
        print(f"{'─' * 60}")
        
        similarities = calcular_similitud(respuesta_prueba, respuestas_correctas, model)
        
        if similarities is not None:
            max_similarity = float(np.max(similarities))
            mejor_match_idx = int(np.argmax(similarities))
            mejor_match = respuestas_correctas[mejor_match_idx]
            
            print(f"Similitud máxima: {max_similarity:.4f}")
            print(f"Mejor coincidencia: \"{mejor_match}\"")
            
            # Mostrar top 3 similitudes
            top_indices = np.argsort(similarities)[::-1][:3]
            print("\nTop 3 similitudes:")
            for idx in top_indices:
                print(f"  {similarities[idx]:.4f} - \"{respuestas_correctas[idx]}\"")
            
            # Evaluar
            es_correcta = max_similarity >= SIMILARITY_THRESHOLD
            resultado = "✓ CORRECTO" if es_correcta else "✗ INCORRECTO"
            print(f"\nResultado: {resultado} (umbral: {SIMILARITY_THRESHOLD})")
        else:
            print("✗ Error al calcular similitud")
    
    # Modo interactivo
    print("\n" + "=" * 60)
    print("MODO INTERACTIVO")
    print("=" * 60)
    print("\nEscribe respuestas para probar (o 'salir' para terminar):")
    
    while True:
        respuesta_usuario = input("\nTu respuesta: ").strip()
        
        if respuesta_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
            break
        
        if not respuesta_usuario:
            continue
        
        print(f"\nEvaluando: \"{respuesta_usuario}\"")
        similarities = calcular_similitud(respuesta_usuario, respuestas_correctas, model)
        
        if similarities is not None:
            max_similarity = float(np.max(similarities))
            mejor_match_idx = int(np.argmax(similarities))
            mejor_match = respuestas_correctas[mejor_match_idx]
            
            print(f"Similitud máxima: {max_similarity:.4f}")
            print(f"Mejor coincidencia: \"{mejor_match}\"")
            
            es_correcta = max_similarity >= SIMILARITY_THRESHOLD
            resultado = "✓ CORRECTO" if es_correcta else "✗ INCORRECTO"
            print(f"Resultado: {resultado}")
        else:
            print("✗ Error al calcular similitud")
    
    # Cerrar ventana
    cv2.destroyAllWindows()
    print("\n✓ Test completado")


if __name__ == "__main__":
    test_juego_absurdos()

