"""Модуль для генерации цветовых паттернов в формате YUV."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .utils.constants import (
    Y_BLACK, Y_WHITE, UV_NEUTRAL, MARKER_PATCHES,
    PATTERN_NUMBER_BITS, CALIBRATION_COLORS
)
from .utils.color_transforms import rgb_to_yuv_bt709
from .utils.yuv_utils import create_yuv_buffer


@dataclass
class PatchCoordinates:
    """Координаты патча в YUV буфере."""
    y_range: Tuple[int, int]
    x_range: Tuple[int, int]
    y_uv_range: Tuple[int, int]
    x_uv_range: Tuple[int, int]
    is_tech: bool = False


class PatternGenerator:
    """Класс для генерации цветовых паттернов в формате YUV."""
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        patch_size: int, 
        patch_gap: int, 
        color_range_percent: float,
        bit_depth: int = 8,
        chroma_subsampling: str = "420"
    ):
        """
        Инициализирует генератор цветовых паттернов.
        
        Args:
            width: Ширина кадра
            height: Высота кадра
            patch_size: Размер одного патча
            patch_gap: Промежуток между патчами
            color_range_percent: Процент используемого цветового диапазона [0-100]
            bit_depth: Глубина цвета (по умолчанию 8 бит)
            chroma_subsampling: Формат цветовой субдискретизации
        """
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.patch_gap = patch_gap
        self.color_range_percent = color_range_percent
        self.bit_depth = bit_depth
        self.chroma_subsampling = chroma_subsampling
        self.max_value = (1 << bit_depth) - 1
        
        # Рассчитываем количество патчей
        self.patches_x = width // (patch_size + patch_gap)
        self.patches_y = height // (patch_size + patch_gap)
        
        # Резервируем последнюю строку для технической информации
        self.tech_row = self.patches_y - 1
        
        # Доступное количество патчей для цветов (за исключением технической строки)
        self.available_patches = self.patches_x * (self.patches_y - 1)
        
        # Общее количество патчей в кадре
        self.total_patches = self.patches_x * self.patches_y
        
        self._log_init_info()
        
        # Предварительно вычисляем координаты всех патчей
        self.patch_coords: List[PatchCoordinates] = []
        self.tech_patch_indices: List[int] = []
        self.marker_indices: List[int] = []
        self._initialize_patch_coordinates()
        
        # Генерируем цветовые наборы
        self.colors: List[Tuple[int, int, int]] = []
        self.patterns_count = 0
        self._generate_color_set()
        
        # Создаем маску патчей для валидации
        self.patches_mask = self._create_patches_mask()
    
    def _log_init_info(self) -> None:
        """Выводит информацию об инициализации генератора."""
        print(f"Размер кадра: {self.width}x{self.height}, "
              f"размер патча: {self.patch_size}x{self.patch_size}, "
              f"промежуток: {self.patch_gap}")
        print(f"Сетка патчей: {self.patches_x}x{self.patches_y} = {self.total_patches} патчей")
        print(f"Из них {self.available_patches} для цветовых образцов "
              f"и {self.patches_x} для технической строки")
    
    def _initialize_patch_coordinates(self) -> None:
        """Предварительно вычисляет координаты всех патчей."""
        for i in range(self.total_patches):
            patch_y = i // self.patches_x
            patch_x = i % self.patches_x
            
            x1 = patch_x * (self.patch_size + self.patch_gap)
            y1 = patch_y * (self.patch_size + self.patch_gap)
            
            x1_uv = x1 // 2
            y1_uv = y1 // 2
            patch_size_uv = self.patch_size // 2
            
            # Определяем, является ли патч техническим
            is_tech = (patch_y == self.tech_row)
            
            # Сохраняем координаты патча
            self.patch_coords.append(PatchCoordinates(
                y_range=(y1, y1 + self.patch_size),
                x_range=(x1, x1 + self.patch_size),
                y_uv_range=(y1_uv, y1_uv + patch_size_uv),
                x_uv_range=(x1_uv, x1_uv + patch_size_uv),
                is_tech=is_tech
            ))
            
            # Сохраняем индексы технических патчей
            if is_tech:
                self.tech_patch_indices.append(i)
                
                # Первые N патчей в технической строке резервируем под маркер
                if patch_x < MARKER_PATCHES:
                    self.marker_indices.append(i)
    
    def _generate_color_set(self) -> None:
        """Генерирует оптимизированный набор цветов."""
        # Определяем количество уровней для каждого канала
        levels = int(256 * self.color_range_percent / 100)
        if levels < 1:
            levels = 1
        
        print(f"Генерация цветовых комбинаций ({levels}^3)...")
        values = np.linspace(0, 255, levels, dtype=np.uint8)
                
        # Генерируем все комбинации с помощью numpy
        r, g, b = np.meshgrid(values, values, values, indexing='ij')
        r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()
        self.colors = list(zip(r_flat, g_flat, b_flat))
        
        # Если полученных цветов меньше, чем нужно для патчей, дублируем их
        if len(self.colors) < self.available_patches:
            self.colors = (self.colors * (self.available_patches // len(self.colors) + 1))[:self.available_patches]
        
        # Рассчитываем количество различных паттернов
        self.patterns_count = (len(self.colors) + self.available_patches - 1) // self.available_patches
        
        print(f"Сгенерировано {len(self.colors)} уникальных цветов с {levels} уровнями по каждому каналу")
        print(f"Будет создано {self.patterns_count} уникальных паттернов.")
    
    def _create_patches_mask(self) -> Dict[str, np.ndarray]:
        """
        Создает маску, указывающую расположение патчей (1 - патч, 0 - фон).
        
        Returns:
            Dict[str, np.ndarray]: Маски для Y, U и V плоскостей
        """
        mask_y = np.zeros((self.height, self.width), dtype=np.uint8)
        mask_uv = np.zeros((self.height // 2, self.width // 2), dtype=np.uint8)
        
        # Заполняем маску для всех патчей, включая технические
        for coords in self.patch_coords:
            y_range = coords.y_range
            x_range = coords.x_range
            y_uv_range = coords.y_uv_range
            x_uv_range = coords.x_uv_range
            
            mask_y[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 1
            mask_uv[y_uv_range[0]:y_uv_range[1], x_uv_range[0]:x_uv_range[1]] = 1
        
        return {'Y': mask_y, 'U': mask_uv, 'V': mask_uv}
    
    def generate_pattern_frame(self, pattern_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Генерирует один кадр с цветовыми патчами и маркером.
        
        Args:
            pattern_index: Индекс паттерна для генерации
            
        Returns:
            Dict[str, np.ndarray]: Буфер кадра с Y, U и V плоскостями
        """
        # Создаем буфер кадра
        frame = create_yuv_buffer(self.height, self.width, self.chroma_subsampling)
        
        # Определяем какие цвета использовать для этого паттерна
        start_idx = pattern_index * self.available_patches
        end_idx = min(start_idx + self.available_patches, len(self.colors))
        colors = self.colors[start_idx:end_idx]
        
        # Дополняем недостающими цветами из начала списка если нужно
        if len(colors) < self.available_patches:
            colors = colors + self.colors[:self.available_patches-len(colors)]
        
        # Преобразуем все цвета в YUV за один раз с помощью векторизации
        patches_count = min(len(colors), self.available_patches)
        rgb_array = np.array(colors[:patches_count])
        
        # Быстрое преобразование RGB в YUV для всех цветов сразу
        y_values, u_values, v_values = rgb_to_yuv_bt709(rgb_array)
        
        # Заполняем цветовые патчи (исключая техническую строку)
        color_idx = 0
        for i, coords in enumerate(self.patch_coords):
            # Пропускаем технические патчи
            if coords.is_tech:
                continue
                
            # Проверяем, есть ли еще цвета для заполнения
            if color_idx < patches_count:
                y_range = coords.y_range
                x_range = coords.x_range
                y_uv_range = coords.y_uv_range
                x_uv_range = coords.x_uv_range
                
                # Заполняем Y-плоскость
                frame['Y'][y_range[0]:y_range[1], x_range[0]:x_range[1]] = y_values[color_idx]
                
                # Заполняем U и V плоскости
                frame['U'][y_uv_range[0]:y_uv_range[1], x_uv_range[0]:x_uv_range[1]] = u_values[color_idx]
                frame['V'][y_uv_range[0]:y_uv_range[1], x_uv_range[0]:x_uv_range[1]] = v_values[color_idx]
                
                color_idx += 1
        
        # Добавляем маркер паттерна в техническую строку
        self._add_pattern_marker(frame, pattern_index)
        
        # Добавляем калибровочные цвета в оставшуюся часть технической строки
        self._add_calibration_colors(frame)
        
        return frame
    
    def _add_pattern_marker(self, frame: Dict[str, np.ndarray], pattern_index: int) -> None:
        """
        Добавляет маркер для идентификации паттерна в техническую строку.
        
        Args:
            frame: Буфер кадра
            pattern_index: Индекс паттерна
        """
        # 1. Начальная якорная метка (2 патча)
        self._draw_anchor_start(frame)
        
        # 2. Двоичное представление номера паттерна (12 патчей)
        self._draw_pattern_number(frame, pattern_index)
        
        # 3. Контрольная сумма (4 патча)
        self._draw_checksum(frame, pattern_index)
        
        # 4. Конечная якорная метка (2 патча)
        self._draw_anchor_end(frame)
    
    def _add_calibration_colors(self, frame: Dict[str, np.ndarray]) -> None:
        """
        Добавляет эталонные цвета для калибровки в техническую строку.
        
        Args:
            frame: Буфер кадра
        """
        # Начинаем с позиции после маркера
        start_idx = MARKER_PATCHES
        
        # Преобразуем RGB цвета в YUV
        rgb_array = np.array(CALIBRATION_COLORS)
        y_values, u_values, v_values = rgb_to_yuv_bt709(rgb_array)
        
        # Рисуем калибровочные цвета
        for i, color_idx in enumerate(range(start_idx, min(start_idx + len(CALIBRATION_COLORS), self.patches_x))):
            patch_idx = self.tech_patch_indices[color_idx]
            coords = self.patch_coords[patch_idx]
            
            # Заполняем Y-плоскость
            frame['Y'][coords.y_range[0]:coords.y_range[1], 
                     coords.x_range[0]:coords.x_range[1]] = y_values[i]
            
            # Заполняем U и V плоскости
            frame['U'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                     coords.x_uv_range[0]:coords.x_uv_range[1]] = u_values[i]
            frame['V'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                     coords.x_uv_range[0]:coords.x_uv_range[1]] = v_values[i]
        
        # Заполняем оставшиеся патчи в технической строке черным цветом
        remaining_start = start_idx + len(CALIBRATION_COLORS)
        for color_idx in range(remaining_start, self.patches_x):
            patch_idx = self.tech_patch_indices[color_idx]
            coords = self.patch_coords[patch_idx]
            
            # Черный цвет в Y, нейтральный в UV
            frame['Y'][coords.y_range[0]:coords.y_range[1], 
                     coords.x_range[0]:coords.x_range[1]] = Y_BLACK
            frame['U'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                     coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
            frame['V'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                     coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
    
    def _draw_pattern_number(self, frame: Dict[str, np.ndarray], pattern_index: int) -> str:
        """
        Рисует двоичное представление номера паттерна.
        
        Args:
            frame: Буфер кадра
            pattern_index: Номер паттерна для кодирования
            
        Returns:
            str: Двоичное представление номера паттерна
        """
        # Преобразуем номер паттерна в N-битное двоичное представление
        binary = format(pattern_index, f'0{PATTERN_NUMBER_BITS}b')
        
        # Рисуем каждый бит как отдельный патч (индексы 2-13)
        for bit_idx, bit in enumerate(binary):
            patch_idx = self.marker_indices[2 + bit_idx]  # Смещение на 2 для пропуска якорей
            coords = self.patch_coords[patch_idx]
            
            # Определяем цвет (белый для 1, черный для 0)
            # Используем экстремальные значения для максимального контраста
            color = Y_WHITE if bit == '1' else Y_BLACK
            
            # Заполняем Y-плоскость
            frame['Y'][coords.y_range[0]:coords.y_range[1], 
                    coords.x_range[0]:coords.x_range[1]] = color
            
            # Заполняем U и V плоскости (нейтральный серый)
            frame['U'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                    coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
            frame['V'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                    coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
        
        return binary
    
    def _draw_checksum(self, frame: Dict[str, np.ndarray], pattern_index: int) -> str:
        """
        Рисует контрольную сумму для номера паттерна.
        
        Args:
            frame: Буфер кадра
            pattern_index: Номер паттерна для вычисления контрольной суммы
            
        Returns:
            str: Двоичное представление контрольной суммы
        """
        # Преобразуем номер паттерна в N-битное двоичное представление
        binary = format(pattern_index, f'0{PATTERN_NUMBER_BITS}b')
        
        # Разбиваем на 4 части по 3 бита и вычисляем XOR каждой части
        checksum = 0
        for i in range(0, PATTERN_NUMBER_BITS, 3):
            end = min(i + 3, PATTERN_NUMBER_BITS)
            chunk = int(binary[i:end], 2)
            checksum ^= chunk
        
        # Преобразуем контрольную сумму в 4-битное представление
        checksum_binary = format(checksum, '04b')
        
        # Рисуем контрольную сумму (индексы 14-17)
        for bit_idx, bit in enumerate(checksum_binary):
            patch_idx = self.marker_indices[14 + bit_idx]
            coords = self.patch_coords[patch_idx]
            
            # Определяем цвет (белый для 1, черный для 0)
            # Используем экстремальные значения для максимального контраста
            color = Y_WHITE if bit == '1' else Y_BLACK
            
            # Заполняем Y-плоскость
            frame['Y'][coords.y_range[0]:coords.y_range[1], 
                    coords.x_range[0]:coords.x_range[1]] = color
            
            # Заполняем U и V плоскости (нейтральный серый)
            frame['U'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                    coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
            frame['V'][coords.y_uv_range[0]:coords.y_uv_range[1], 
                    coords.x_uv_range[0]:coords.x_uv_range[1]] = UV_NEUTRAL
        
        return checksum_binary
    
    def _draw_anchor_start(self, frame: Dict[str, np.ndarray]) -> None:
        """
        Рисует высококонтрастную начальную якорную метку.
        
        Args:
            frame: Буфер кадра
        """
        self._draw_checkered_pattern(frame, 0, 1, is_start=True)


    def _draw_anchor_end(self, frame: Dict[str, np.ndarray]) -> None:
        """
        Рисует высококонтрастную конечную якорную метку.
        
        Args:
            frame: Буфер кадра
        """
        self._draw_checkered_pattern(frame, 18, 19, is_start=False)


    def _draw_checkered_pattern(
        self, 
        frame: Dict[str, np.ndarray], 
        start_idx: int, 
        end_idx: int, 
        is_start: bool = True
    ) -> None:
        """
        Рисует высококонтрастный шахматный узор для якорных меток.
        
        Args:
            frame: Буфер кадра
            start_idx: Начальный индекс в списке маркерных патчей
            end_idx: Конечный индекс в списке маркерных патчей
            is_start: True для начальной метки, False для конечной
        """
        for i in range(start_idx, end_idx + 1):
            patch_idx = self.marker_indices[i]
            coords = self.patch_coords[patch_idx]
            
            y_range = coords.y_range
            x_range = coords.x_range
            y_uv_range = coords.y_uv_range
            x_uv_range = coords.x_uv_range
            
            # Разделяем патч на 4 квадранта с четкими границами
            half_size_y = (y_range[1] - y_range[0]) // 2
            half_size_x = (x_range[1] - x_range[0]) // 2
            half_size_uv_y = (y_uv_range[1] - y_uv_range[0]) // 2
            half_size_uv_x = (x_uv_range[1] - x_uv_range[0]) // 2
            
            # Координаты для 4 квадрантов (высокая точность)
            quadrants_y = [
                (y_range[0], y_range[0] + half_size_y, x_range[0], x_range[0] + half_size_x),  # верхний левый
                (y_range[0], y_range[0] + half_size_y, x_range[0] + half_size_x, x_range[1]),  # верхний правый
                (y_range[0] + half_size_y, y_range[1], x_range[0], x_range[0] + half_size_x),  # нижний левый
                (y_range[0] + half_size_y, y_range[1], x_range[0] + half_size_x, x_range[1])   # нижний правый
            ]
            
            quadrants_uv = [
                (y_uv_range[0], y_uv_range[0] + half_size_uv_y, x_uv_range[0], x_uv_range[0] + half_size_uv_x),
                (y_uv_range[0], y_uv_range[0] + half_size_uv_y, x_uv_range[0] + half_size_uv_x, x_uv_range[1]),
                (y_uv_range[0] + half_size_uv_y, y_uv_range[1], x_uv_range[0], x_uv_range[0] + half_size_uv_x),
                (y_uv_range[0] + half_size_uv_y, y_uv_range[1], x_uv_range[0] + half_size_uv_x, x_uv_range[1])
            ]
            
            # Определяем паттерн цветов
            # Начальная метка: 1-й патч (ЧБ/БЧ), 2-й патч (БЧ/ЧБ)
            # Конечная метка: 1-й патч (БЧ/ЧБ), 2-й патч (ЧБ/БЧ) - инверсия начальной
            if is_start:
                # Для начальной метки
                colors = [Y_BLACK, Y_WHITE] if i == start_idx else [Y_WHITE, Y_BLACK]
            else:
                # Для конечной метки (инверсия)
                colors = [Y_WHITE, Y_BLACK] if i == start_idx else [Y_BLACK, Y_WHITE]
            
            # Заполняем квадранты шахматным узором с максимальной контрастностью
            for q in range(4):
                y1, y2, x1, x2 = quadrants_y[q]
                yuv1, yuv2, xuv1, xuv2 = quadrants_uv[q]
                
                # Классический шахматный узор: верхний левый и нижний правый одного цвета,
                # верхний правый и нижний левый - другого цвета
                color = colors[0] if q in [0, 3] else colors[1]
                
                # Y-плоскость с максимальным контрастом
                frame['Y'][y1:y2, x1:x2] = color
                
                # U и V плоскости (нейтральный серый) для чистого ч/б
                frame['U'][yuv1:yuv2, xuv1:xuv2] = UV_NEUTRAL
                frame['V'][yuv1:yuv2, xuv1:xuv2] = UV_NEUTRAL
                
    def save_pattern_metadata(self, pattern_index: int, metadata_handler) -> None:
        """
        Сохраняет метаданные о паттерне для последующей валидации.
        
        Args:
            pattern_index: Индекс паттерна
            metadata_handler: Обработчик метаданных
        """
        # Определяем какие цвета используются для этого паттерна
        start_idx = pattern_index * self.available_patches
        end_idx = min(start_idx + self.available_patches, len(self.colors))
        colors = self.colors[start_idx:end_idx]
        
        # Дополняем недостающими цветами из начала списка если нужно
        if len(colors) < self.available_patches:
            colors = colors + self.colors[:self.available_patches-len(colors)]
        
        # Преобразуем все цвета в YUV за один раз с помощью векторизации
        patches_count = min(len(colors), self.available_patches)
        rgb_array = np.array(colors[:patches_count])
        
        # Быстрое преобразование RGB в YUV для всех цветов сразу
        y_values, u_values, v_values = rgb_to_yuv_bt709(rgb_array)
        
        # Сохраняем метаданные
        metadata_handler.save_pattern_metadata(
            pattern_index, colors, self.patch_coords,
            y_values, u_values, v_values
        )

    def _draw_pattern_marker(self, frame: Dict[str, np.ndarray], pattern_index: int) -> None:
        """
        Добавляет маркер для идентификации паттерна в техническую строку.
        
        Args:
            frame: Буфер кадра
            pattern_index: Индекс паттерна
        """
        print(f"Добавление маркера для паттерна {pattern_index}")
        
        # 1. Начальная якорная метка (2 патча)
        self._draw_anchor_start(frame)
        
        # 2. Двоичное представление номера паттерна (12 патчей)
        binary = self._draw_pattern_number(frame, pattern_index)
        
        # 3. Контрольная сумма (4 патча)
        checksum_binary = self._draw_checksum(frame, pattern_index)
        
        # 4. Конечная якорная метка (2 патча)
        self._draw_anchor_end(frame)
        
        print(f"  Маркер паттерна: бинарно {binary}, контр. сумма {checksum_binary}")
