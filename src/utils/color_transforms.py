"""Функции для преобразования цветовых пространств."""

import numpy as np
from typing import Tuple

from .constants import (
    Y_MIN_LIMITED, Y_MAX_LIMITED, Y_RANGE_LIMITED, 
    UV_NEUTRAL_LIMITED, UV_RANGE_LIMITED,
    Y_MIN_FULL, Y_MAX_FULL, Y_RANGE_FULL, 
    UV_NEUTRAL_FULL, UV_RANGE_FULL,
    ColorRange, get_yuv_constants
)


def rgb_to_yuv_bt709(
    rgb_array: np.ndarray, 
    color_range: str = ColorRange.LIMITED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразует массив RGB значений в YUV (BT.709).
    
    Args:
        rgb_array: numpy.ndarray формы (N, 3) с RGB значениями [0-255]
        color_range: Цветовой диапазон ('limited' или 'full')
        
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Y, U, V компоненты
    """
    # Получаем константы в зависимости от выбранного диапазона
    yuv_const = get_yuv_constants(color_range)
    y_min = yuv_const["Y_MIN"]
    y_range = yuv_const["Y_RANGE"]
    uv_neutral = yuv_const["UV_NEUTRAL"]
    uv_range = yuv_const["UV_RANGE"]
    
    # Нормализуем к [0, 1]
    r_norm = rgb_array[:, 0] / 255.0
    g_norm = rgb_array[:, 1] / 255.0
    b_norm = rgb_array[:, 2] / 255.0
    
    # Матрица преобразования BT.709
    y = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm
    u = -0.1146 * r_norm - 0.3854 * g_norm + 0.5000 * b_norm
    v = 0.5000 * r_norm - 0.4542 * g_norm - 0.0458 * b_norm
    
    # Применяем выбранный диапазон (вычисления в float)
    y_float = y_min + y * y_range
    u_float = uv_neutral + u * uv_range
    v_float = uv_neutral + v * uv_range

    # Ограничиваем float значения стандартными диапазонами,
    # Клиппим U и V, чтобы избежать максимального значения (UV_MAX)
    y_clipped = np.clip(y_float, yuv_const["Y_MIN"], yuv_const["Y_MAX"])
    u_clipped = np.clip(u_float, yuv_const["UV_MIN"], yuv_const["UV_MAX"])
    v_clipped = np.clip(v_float, yuv_const["UV_MIN"], yuv_const["UV_MAX"])
    
    # Округляем и конвертируем в uint8 ПОСЛЕ клиппинга
    y_values = np.round(y_clipped).astype(np.uint8)
    u_values = np.round(u_clipped).astype(np.uint8)
    v_values = np.round(v_clipped).astype(np.uint8)

    return y_values, u_values, v_values


def yuv_to_rgb_bt709(
    y: np.ndarray, 
    u: np.ndarray, 
    v: np.ndarray, 
    color_range: str = ColorRange.LIMITED
) -> np.ndarray:
    """
    Преобразует YUV (BT.709) в RGB.
    
    Args:
        y: numpy.ndarray с Y компонентой
        u: numpy.ndarray с U компонентой
        v: numpy.ndarray с V компонентой
        color_range: Цветовой диапазон ('limited' или 'full')
        
    Returns:
        numpy.ndarray: RGB представление [0-255]
    """
    # Получаем константы в зависимости от выбранного диапазона
    yuv_const = get_yuv_constants(color_range)
    y_min = yuv_const["Y_MIN"]
    y_range = yuv_const["Y_RANGE"]
    uv_neutral = yuv_const["UV_NEUTRAL"]
    uv_range = yuv_const["UV_RANGE"]
    
    # Нормализация YUV к [0, 1] для Y и [-0.5, 0.5] для U/V
    y_norm = (y.astype(np.float64) - y_min) / y_range
    u_norm = (u.astype(np.float64) - uv_neutral) / uv_range 
    v_norm = (v.astype(np.float64) - uv_neutral) / uv_range
    
    # Матрица преобразования BT.709
    r = y_norm + 1.5748 * v_norm
    g = y_norm - 0.1873 * u_norm - 0.4681 * v_norm
    b = y_norm + 1.8556 * u_norm
    
    # Клиппинг и преобразование в [0, 255]
    rgb = np.stack([
        np.clip(r * 255, 0, 255),
        np.clip(g * 255, 0, 255),
        np.clip(b * 255, 0, 255)
    ], axis=-1).astype(np.uint8)
    
    return rgb