"""
juego-historia views: selection and voice flow.
"""
from .selection_dificultad import mostrar_seleccion_dificultad
from .selection_historias import mostrar_seleccion_historias
from .selection_acciones import mostrar_seleccion_acciones
from .selection_lugares import mostrar_seleccion_lugares
from .vista_final import mostrar_vista_final
from .voice_flow import run_historia_voice_flow

__all__ = [
    "mostrar_seleccion_dificultad",
    "mostrar_seleccion_historias",
    "mostrar_seleccion_acciones",
    "mostrar_seleccion_lugares",
    "mostrar_vista_final",
    "run_historia_voice_flow",
]

