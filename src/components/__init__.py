"""
UI Components for MagicboARd games.
Reusable button components for consistent UI across games.
"""
from .rectangular_button import draw_rectangular_button, is_point_in_rectangular_button
from .circular_button import draw_circular_button, is_point_in_circular_button

__all__ = [
    'draw_rectangular_button',
    'is_point_in_rectangular_button',
    'draw_circular_button',
    'is_point_in_circular_button',
]

