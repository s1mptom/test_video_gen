"""Утилиты для работы с YUV форматом."""

import numpy as np
from typing import Dict, Tuple, Optional

from .constants import (
    Y_BLACK_LIMITED, UV_NEUTRAL_LIMITED,
    Y_BLACK_FULL, UV_NEUTRAL_FULL,
    ColorRange, ChromaFormat
)


def create_y_plane(
    height: int, 
    width: int, 
    value: int = None, 
    color_range: str = ColorRange.LIMITED
) -> np.ndarray:
    """
    Создает Y-плоскость с заданным значением.
    
    Args:
        height: Высота плоскости
        width: Ширина плоскости
        value: Значение для заполнения (по умолчанию черный в выбранном диапазоне)
        color_range: Цветовой диапазон ('limited' или 'full')
        
    Returns:
        numpy.ndarray: Y-плоскость
    """
    if value is None:
        value = Y_BLACK_FULL if color_range == ColorRange.FULL else Y_BLACK_LIMITED
    
    return np.full((height, width), value, dtype=np.uint8)


def create_uv_plane(
    height: int, 
    width: int, 
    value: int = None,
    color_range: str = ColorRange.LIMITED
) -> np.ndarray:
    """
    Создает U или V плоскость с заданным значением.
    
    Args:
        height: Высота плоскости
        width: Ширина плоскости
        value: Значение для заполнения (по умолчанию нейтральный)
        color_range: Цветовой диапазон ('limited' или 'full')
        
    Returns:
        numpy.ndarray: U или V плоскость
    """
    if value is None:
        value = UV_NEUTRAL_FULL if color_range == ColorRange.FULL else UV_NEUTRAL_LIMITED
    
    return np.full((height, width), value, dtype=np.uint8)


def create_yuv_buffer(
    height: int, 
    width: int, 
    chroma_subsampling: str = ChromaFormat.YUV_422,
    color_range: str = ColorRange.LIMITED
) -> Dict[str, np.ndarray]:
    """
    Создает буфер кадра в формате YUV с нейтрально-серым фоном.
    
    Args:
        height: Высота кадра
        width: Ширина кадра
        chroma_subsampling: Формат цветовой субдискретизации ("420", "422", "444")
        color_range: Цветовой диапазон ('limited' или 'full')
        
    Returns:
        Dict[str, np.ndarray]: Словарь с Y, U и V плоскостями
    """
    # Y-плоскость (полное разрешение)
    y_plane = create_y_plane(height, width, color_range=color_range)
    
    # U и V плоскости в зависимости от субдискретизации
    if chroma_subsampling == ChromaFormat.YUV_420:
        u_plane = create_uv_plane(height // 2, width // 2, color_range=color_range)
        v_plane = create_uv_plane(height // 2, width // 2, color_range=color_range)
    elif chroma_subsampling == ChromaFormat.YUV_422:
        u_plane = create_uv_plane(height, width // 2, color_range=color_range)
        v_plane = create_uv_plane(height, width // 2, color_range=color_range)
    elif chroma_subsampling == ChromaFormat.YUV_444:
        u_plane = create_uv_plane(height, width, color_range=color_range)
        v_plane = create_uv_plane(height, width, color_range=color_range)
    else:
        raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_subsampling}")
    
    return {'Y': y_plane, 'U': u_plane, 'V': v_plane}


def get_chroma_dimensions(
    height: int, 
    width: int, 
    chroma_subsampling: str
) -> Tuple[int, int]:
    """
    Возвращает размеры хроматических (U, V) плоскостей в зависимости от формата.
    
    Args:
        height: Высота исходного кадра
        width: Ширина исходного кадра
        chroma_subsampling: Формат цветовой субдискретизации
    
    Returns:
        Tuple[int, int]: Высота и ширина хроматических плоскостей
    """
    if chroma_subsampling == ChromaFormat.YUV_420:
        return height // 2, width // 2
    elif chroma_subsampling == ChromaFormat.YUV_422:
        return height, width // 2
    elif chroma_subsampling == ChromaFormat.YUV_444:
        return height, width
    else:
        raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_subsampling}")