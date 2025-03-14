"""Константы для проекта генерации цветовых паттернов."""

from typing import List, Tuple

# Цветовые константы YUV
Y_BLACK = 16
Y_WHITE = 235
UV_NEUTRAL = 128

# Цветовые ограничения для видео (BT.709)
Y_MIN = 16
Y_MAX = 235
Y_RANGE = Y_MAX - Y_MIN  # 219
UV_MIN = 16
UV_MAX = 240
UV_RANGE = 224

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