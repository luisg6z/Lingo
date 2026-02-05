"""
Script para convertir LingoBien.svg a LingoBien.png
Ejecuta este script para crear la versión PNG del SVG
"""

import os
import sys

def convert_svg_to_png():
    svg_path = "images/LingoBien.svg"
    png_path = "images/LingoBien.png"
    
    if not os.path.exists(svg_path):
        print(f"Error: No se encontró el archivo {svg_path}")
        return False
    
    try:
        import cairosvg
        print(f"Convirtiendo {svg_path} a {png_path}...")
        
        # Convertir SVG a PNG con tamaño 800x800 (alta resolución)
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=800, output_height=800)
        
        if os.path.exists(png_path):
            print(f"✓ Conversión exitosa: {png_path} creado")
            return True
        else:
            print("✗ Error: El archivo PNG no se creó")
            return False
            
    except ImportError:
        print("Error: cairosvg no está instalado")
        print("Instálalo con: pip install cairosvg")
        return False
    except Exception as e:
        print(f"Error al convertir: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_svg_to_png()
    sys.exit(0 if success else 1)

