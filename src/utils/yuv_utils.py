"""Утилиты для работы с YUV форматом."""

import numpy as np
from typing import Dict, Tuple, Optional

from .constants import Y_BLACK, UV_NEUTRAL


def create_y_plane(height: int, width: int, value: int = Y_BLACK) -> np.ndarray:
    """
    Создает Y-плоскость с заданным значением.
    
    Args:
        height: Высота плоскости
        width: Ширина плоскости
        value: Значение для заполнения (по умолчанию черный)
        
    Returns:
        numpy.ndarray: Y-плоскость
    """
    return np.full((height, width), value, dtype=np.uint8)


def create_uv_plane(height: int, width: int, value: int = UV_NEUTRAL) -> np.ndarray:
    """
    Создает U или V плоскость с заданным значением.
    
    Args:
        height: Высота плоскости
        width: Ширина плоскости
        value: Значение для заполнения (по умолчанию нейтральный)
        
    Returns:
        numpy.ndarray: U или V плоскость
    """
    return np.full((height, width), value, dtype=np.uint8)


def create_yuv_buffer(height: int, width: int, chroma_subsampling: str = "422"):
    """
    Создает буфер кадра в формате YUV с нейтрально-серым фоном.
    
    Args:
        height: Высота кадра
        width: Ширина кадра
        chroma_subsampling: Формат цветовой субдискретизации ("420", "422", "444")
        
    Returns:
        Dict[str, np.ndarray]: Словарь с Y, U и V плоскостями
    """
    # Y-плоскость (полное разрешение)
    y_plane = create_y_plane(height, width)
    
    # U и V плоскости в зависимости от субдискретизации
    if chroma_subsampling == "420":
        u_plane = create_uv_plane(height // 2, width // 2)
        v_plane = create_uv_plane(height // 2, width // 2)
    elif chroma_subsampling == "422":
        u_plane = create_uv_plane(height, width // 2)
        v_plane = create_uv_plane(height, width // 2)
    elif chroma_subsampling == "444":
        u_plane = create_uv_plane(height, width)
        v_plane = create_uv_plane(height, width)
    else:
        raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_subsampling}")
    
    return {'Y': y_plane, 'U': u_plane, 'V': v_plane}