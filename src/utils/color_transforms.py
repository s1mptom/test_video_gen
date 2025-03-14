"""Функции для преобразования цветовых пространств."""

import numpy as np
from typing import Tuple

from .constants import Y_MIN, Y_RANGE, UV_NEUTRAL, UV_RANGE


def rgb_to_yuv_bt709(rgb_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразует массив RGB значений в YUV (BT.709) в видеодиапазоне.
    
    Args:
        rgb_array: numpy.ndarray формы (N, 3) с RGB значениями [0-255]
        
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Y, U, V компоненты [16-235, 16-240, 16-240]
    """
    # Нормализуем к [0, 1]
    r_norm = rgb_array[:, 0] / 255.0
    g_norm = rgb_array[:, 1] / 255.0
    b_norm = rgb_array[:, 2] / 255.0
    
    # Матрица преобразования BT.709
    y = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm
    u = -0.1146 * r_norm - 0.3854 * g_norm + 0.5000 * b_norm
    v = 0.5000 * r_norm - 0.4542 * g_norm - 0.0458 * b_norm
    
    # Применяем ограниченный диапазон для видео
    y_values = np.round(Y_MIN + y * Y_RANGE).astype(np.uint8)
    u_values = np.round(UV_NEUTRAL + u * (UV_RANGE // 2)).astype(np.uint8)
    v_values = np.round(UV_NEUTRAL + v * (UV_RANGE // 2)).astype(np.uint8)
    
    return y_values, u_values, v_values


def yuv_to_rgb_bt709(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Преобразует YUV (BT.709) в RGB в полном диапазоне.
    
    Args:
        y: numpy.ndarray с Y компонентой [16-235]
        u: numpy.ndarray с U компонентой [16-240]
        v: numpy.ndarray с V компонентой [16-240]
        
    Returns:
        numpy.ndarray: RGB представление [0-255]
    """
    # Нормализация YUV к [0, 1]
    y_norm = (y.astype(np.float32) - Y_MIN) / Y_RANGE
    u_norm = (u.astype(np.float32) - UV_NEUTRAL) / (UV_RANGE // 2)
    v_norm = (v.astype(np.float32) - UV_NEUTRAL) / (UV_RANGE // 2)
    
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