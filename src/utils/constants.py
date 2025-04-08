"""Константы для проекта генерации цветовых паттернов."""

from typing import List, Tuple, Dict

# Цветовой диапазон (range)
class ColorRange:
    LIMITED = "limited"
    FULL = "full"

# Форматы цветовой субдискретизации
class ChromaFormat:
    YUV_420 = "420"
    YUV_422 = "422"
    YUV_444 = "444"

# Цветовые константы YUV для LIMITED range
Y_BLACK_LIMITED = 16
Y_WHITE_LIMITED = 235
UV_NEUTRAL_LIMITED = 128

# Цветовые константы YUV для FULL range
Y_BLACK_FULL = 0
Y_WHITE_FULL = 255
UV_NEUTRAL_FULL = 128

# По умолчанию используем LIMITED range
Y_BLACK = Y_BLACK_LIMITED
Y_WHITE = Y_WHITE_LIMITED
UV_NEUTRAL = UV_NEUTRAL_LIMITED

# Цветовые ограничения для видео (BT.709) в LIMITED range
Y_MIN_LIMITED = 16
Y_MAX_LIMITED = 235
Y_RANGE_LIMITED = Y_MAX_LIMITED - Y_MIN_LIMITED  # 219
UV_MIN_LIMITED = 16
UV_MAX_LIMITED = 240
UV_RANGE_LIMITED = 224

# Цветовые ограничения для видео (BT.709) в FULL range
Y_MIN_FULL = 0
Y_MAX_FULL = 255
Y_RANGE_FULL = Y_MAX_FULL - Y_MIN_FULL  # 255
UV_MIN_FULL = 0
UV_MAX_FULL = 255
UV_RANGE_FULL = 255

# По умолчанию используем LIMITED range
Y_MIN = Y_MIN_LIMITED
Y_MAX = Y_MAX_LIMITED
Y_RANGE = Y_RANGE_LIMITED
UV_MIN = UV_MIN_LIMITED
UV_MAX = UV_MAX_LIMITED
UV_RANGE = UV_RANGE_LIMITED

# Функции для получения констант в зависимости от выбранного диапазона
def get_yuv_constants(color_range: str = ColorRange.LIMITED) -> Dict[str, int]:
    """
    Возвращает константы YUV в зависимости от выбранного диапазона.
    
    Args:
        color_range: Цветовой диапазон (limited или full)
        
    Returns:
        Dict[str, int]: Словарь с константами
    """
    if color_range == ColorRange.FULL:
        return {
            "Y_BLACK": Y_BLACK_FULL,
            "Y_WHITE": Y_WHITE_FULL,
            "UV_NEUTRAL": UV_NEUTRAL_FULL,
            "Y_MIN": Y_MIN_FULL,
            "Y_MAX": Y_MAX_FULL,
            "Y_RANGE": Y_RANGE_FULL,
            "UV_MIN": UV_MIN_FULL,
            "UV_MAX": UV_MAX_FULL,
            "UV_RANGE": UV_RANGE_FULL,
        }
    else:
        return {
            "Y_BLACK": Y_BLACK_LIMITED,
            "Y_WHITE": Y_WHITE_LIMITED,
            "UV_NEUTRAL": UV_NEUTRAL_LIMITED,
            "Y_MIN": Y_MIN_LIMITED,
            "Y_MAX": Y_MAX_LIMITED,
            "Y_RANGE": Y_RANGE_LIMITED,
            "UV_MIN": UV_MIN_LIMITED,
            "UV_MAX": UV_MAX_LIMITED,
            "UV_RANGE": UV_RANGE_LIMITED,
        }

# Количество бит для кодирования номера паттерна
PATTERN_NUMBER_BITS = 12

# Маркер для идентификации паттерна
MARKER_PATCHES = 20  # 2(якорь) + 12(номер) + 4(контр. сумма) + 2(якорь)

# Калибровочные цвета (RGB)
CALIBRATION_COLORS: List[Tuple[int, int, int]] = [
    # Белый и градации серого от светлого к темному
    (255, 255, 255),  # Белый (100%)
    (224, 224, 224),  # Серый (90%)
    (192, 192, 192),  # Серый (75%)
    (160, 160, 160),  # Серый (60%)
    (128, 128, 128),  # Серый (50%)
    (96, 96, 96),     # Серый (40%)
    (64, 64, 64),     # Серый (25%)
    (32, 32, 32),     # Серый (10%)
    (0, 0, 0),        # Черный (0%)
    
    # Основные цвета
    (255, 0, 0),      # Красный (100%)
    (0, 255, 0),      # Зеленый (100%)
    (0, 0, 255),      # Синий (100%)
    (255, 255, 0),    # Желтый
    (0, 255, 255),    # Голубой
    (255, 0, 255),    # Пурпурный
    
    # Промежуточные значения для RGB
    (128, 0, 0),      # Темно-красный (50%)
    (0, 128, 0),      # Темно-зеленый (50%)
    (0, 0, 128),      # Темно-синий (50%)
    (128, 128, 0),    # Темно-желтый
    (0, 128, 128),    # Темно-голубой
    (128, 0, 128),    # Темно-пурпурный
]

# Команды для внешних утилит
ENCODER_CMD = "x265"
MUXER_CMD = "MP4Box"
DECODER_CMD = "ffmpeg"