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


def create_yuv_buffer(
    height: int, 
    width: int, 
    chroma_subsampling: str = "420"
) -> Dict[str, np.ndarray]:
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


def get_chroma_dimensions(
    height: int, 
    width: int, 
    chroma_subsampling: str = "420"
) -> Tuple[int, int]:
    """
    Возвращает размеры цветоразностных плоскостей для заданной субдискретизации.
    
    Args:
        height: Высота исходного изображения
        width: Ширина исходного изображения
        chroma_subsampling: Формат цветовой субдискретизации ("420", "422", "444")
        
    Returns:
        Tuple[int, int]: (высота, ширина) цветоразностных плоскостей
    """
    if chroma_subsampling == "420":
        return height // 2, width // 2
    elif chroma_subsampling == "422":
        return height, width // 2
    elif chroma_subsampling == "444":
        return height, width
    else:
        raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_subsampling}")


def fill_rect_yuv(
    frame: Dict[str, np.ndarray],
    y_range: Tuple[int, int],
    x_range: Tuple[int, int],
    y_value: int,
    u_value: Optional[int] = None,
    v_value: Optional[int] = None,
    chroma_subsampling: str = "420"
) -> None:
    """
    Заполняет прямоугольную область в YUV буфере заданными значениями.
    
    Args:
        frame: YUV буфер кадра
        y_range: Диапазон координат по вертикали (y1, y2)
        x_range: Диапазон координат по горизонтали (x1, x2)
        y_value: Значение для Y-плоскости
        u_value: Значение для U-плоскости (если None, то UV_NEUTRAL)
        v_value: Значение для V-плоскости (если None, то UV_NEUTRAL)
        chroma_subsampling: Формат цветовой субдискретизации
    """
    y1, y2 = y_range
    x1, x2 = x_range
    
    # Заполняем Y-плоскость
    frame['Y'][y1:y2, x1:x2] = y_value
    
    # Значения по умолчанию для UV
    u_value = UV_NEUTRAL if u_value is None else u_value
    v_value = UV_NEUTRAL if v_value is None else v_value
    
    # Вычисляем координаты для цветоразностных плоскостей
    if chroma_subsampling == "420":
        y1_uv, y2_uv = y1 // 2, (y2 + 1) // 2
        x1_uv, x2_uv = x1 // 2, (x2 + 1) // 2
    elif chroma_subsampling == "422":
        y1_uv, y2_uv = y1, y2
        x1_uv, x2_uv = x1 // 2, (x2 + 1) // 2
    elif chroma_subsampling == "444":
        y1_uv, y2_uv = y1, y2
        x1_uv, x2_uv = x1, x2
    else:
        raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_subsampling}")
    
    # Заполняем U и V плоскости
    frame['U'][y1_uv:y2_uv, x1_uv:x2_uv] = u_value
    frame['V'][y1_uv:y2_uv, x1_uv:x2_uv] = v_value