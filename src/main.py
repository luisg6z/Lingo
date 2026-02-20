"""
MagicboARd - Main entry point
"""
import cv2
import os
import sys
import time
from openni import openni2

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core utilities
from src.core.openni_init import initialize_openni2

# Import menu feature
from src.features.menu.menu import mostrar_menu_juegos


def load_sentence_transformer_model():
    """
    Load the sentence-transformer model for the absurdos game.
    
    Returns:
        SentenceTransformer model or None if loading fails
    """
    print("\n" + "=" * 60)
    print("CARGANDO MODELO DE SENTENCE-TRANSFORMERS")
    print("=" * 60)
    try:
        from sentence_transformers import SentenceTransformer
        print("Cargando modelo 'paraphrase-multilingual-MiniLM-L12-v2'...")
        print("(Esto puede tardar unos minutos la primera vez que se descarga)")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ Modelo de sentence-transformers cargado correctamente")
                return model
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"✗ Intento {attempt + 1} fallido: {e}")
                    print(f"  Reintentando en {wait_time} segundos...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Error al cargar modelo después de {max_retries} intentos: {e}")
                    print("  El juego de absurdos puede no funcionar correctamente.")
                    print("  Verifica tu conexión a internet e intenta nuevamente.")
                    return None
    except ImportError:
        print("✗ sentence-transformers no está instalado.")
        print("  Instálalo con: uv pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"✗ Error inesperado al cargar modelo: {e}")
        return None
    
    print("=" * 60 + "\n")
    return None


if __name__ == "__main__":
    # Initialize OpenNI2
    device, success = initialize_openni2()
    if not success:
        print("Failed to initialize OpenNI2. Exiting.")
        sys.exit(1)
    
    # Load sentence-transformer model
    sentence_transformer_model = load_sentence_transformer_model()
    
    # Show the game menu
    print("=" * 60)
    print("Iniciando MagicboARd - Menú de Juegos")
    print("=" * 60)
    print("\nPresiona 'q' en la ventana del menú para salir")
    print("Toca las cards para seleccionar un juego\n")
    
    try:
        mostrar_menu_juegos(device, sentence_transformer_model)
    except KeyboardInterrupt:
        print("\n\nInterrupción del usuario. Cerrando...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        openni2.unload()
        cv2.destroyAllWindows()
        print("\n¡Hasta luego!")

