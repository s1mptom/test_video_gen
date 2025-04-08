"""Модуль для валидации видео."""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, BinaryIO

from .video_processor import VideoProcessor
from .utils.constants import ChromaFormat, ColorRange


class ValidationProcessor:
    """Класс для валидации видео."""
    
    def __init__(self, debug_mode: bool = False, debug_dir: Optional[Path] = None):
        """
        Инициализирует процессор валидации.
        
        Args:
            debug_mode: Режим отладки
            debug_dir: Директория для отладочных файлов
        """
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
    
    def verify_frame(
        self, 
        expected: Dict[str, np.ndarray], 
        actual: Dict[str, np.ndarray], 
        patches_mask: Dict[str, np.ndarray],
        deviation: int = 4, 
        max_miss_percent: float = 0.002,
        chroma_format: str = ChromaFormat.YUV_422
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """
        Проверяет кадр с допустимым отклонением и процентом ошибок.
        
        Args:
            expected: Ожидаемый буфер кадра
            actual: Фактический буфер кадра
            patches_mask: Маска патчей (1 - патч, 0 - фон)
            deviation: Максимальное допустимое отклонение значений
            max_miss_percent: Максимальный допустимый процент ошибок
            chroma_format: Формат цветовой субдискретизации
            
        Returns:
            Tuple[bool, Dict[str, Dict[str, float]]]: 
                Результат проверки (True/False) и детали по каждой плоскости
        """
        results = {}
        
        # Для каждой плоскости
        for plane in ['Y', 'U', 'V']:
            expected_plane = expected[plane]
            actual_plane = actual[plane]
            mask = patches_mask[plane]
            
            # Проверяем размеры
            if expected_plane.shape != actual_plane.shape:
                # Пытаемся адаптировать маску, если формат изменился
                if plane in ['U', 'V'] and chroma_format in [ChromaFormat.YUV_420, ChromaFormat.YUV_422]:
                    # Масштабируем маску, если размеры не совпадают
                    mask = self._resize_mask(mask, actual_plane.shape)
                else:
                    raise ValueError(f"Размеры не совпадают для плоскости {plane}: ожидалось {expected_plane.shape}, "
                                  f"получено {actual_plane.shape}")
            
            # Расчет отклонения только для пикселей патчей (применяем маску)
            diff = np.abs(actual_plane.astype(int) - expected_plane.astype(int))
            
            # Применяем маску - учитываем только пиксели, где маска == 1
            if mask.shape != diff.shape:
                # Масштабируем маску, если размеры не совпадают
                mask = self._resize_mask(mask, diff.shape)
                
            masked_diff = diff * mask
            masked_total = np.sum(mask)  # Общее число пикселей в патчах
            
            # Количество пикселей в допустимом диапазоне
            valid_pixels = np.sum((masked_diff <= deviation) | (mask == 0))
            
            # Расчет процента ошибок (только для пикселей в патчах)
            # Избегаем деления на ноль
            if masked_total > 0:
                miss_percent = 1.0 - (valid_pixels / masked_total)
            else:
                miss_percent = 0.0
            
            results[plane] = {
                'miss_percent': miss_percent,
                'max_diff': np.max(masked_diff),
                'valid_ratio': valid_pixels / masked_total if masked_total > 0 else 1.0
            }
            
            # Проверка условия
            if miss_percent > max_miss_percent:
                print(f"Ошибка валидации для {plane}-плоскости:")
                print(f"  Процент ошибок: {miss_percent*100:.4f}%")
                print(f"  Максимальное отклонение: {np.max(masked_diff)}")
                
                # Находим координаты с максимальной ошибкой
                masked_diff_copy = masked_diff.copy()
                masked_diff_copy[mask == 0] = 0  # Игнорируем области вне патчей
                
                if np.max(masked_diff_copy) > 0:
                    max_err_pos = np.unravel_index(np.argmax(masked_diff_copy), masked_diff.shape)
                    print(f"  Позиция макс. ошибки: {max_err_pos}")
                    print(f"  Ожидаемое значение: {expected_plane[max_err_pos]}")
                    print(f"  Полученное значение: {actual_plane[max_err_pos]}")
                
                # Сохраняем отладочные изображения при ошибке
                if self.debug_mode and self.debug_dir:
                    self._save_debug_images(expected, actual, masked_diff, plane)
                
                return False, results
                
        return True, results
    
    def _resize_mask(self, mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Масштабирует маску до целевого размера.
        
        Args:
            mask: Исходная маска
            target_shape: Целевой размер
            
        Returns:
            np.ndarray: Масштабированная маска
        """
        import cv2
        return cv2.resize(mask, (target_shape[1], target_shape[0]), 
                         interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
    
    def _save_debug_images(
        self, 
        expected: Dict[str, np.ndarray], 
        actual: Dict[str, np.ndarray],
        masked_diff: np.ndarray,
        plane: str
    ) -> None:
        """
        Сохраняет отладочные изображения при ошибке валидации.
        
        Args:
            expected: Ожидаемый буфер кадра
            actual: Фактический буфер кадра
            masked_diff: Маскированная разница
            plane: Плоскость с ошибкой
        """
        video_processor = VideoProcessor()
        video_processor.save_debug_frame(expected, "expected_error", self.debug_dir)
        video_processor.save_debug_frame(actual, "actual_error", self.debug_dir)
        
        # Создаем и сохраняем визуализацию разницы
        diff_viz = np.zeros_like(expected[plane])
        diff_viz[masked_diff > 0] = 255  # Выделяем ошибки белым
        
        if plane == 'Y':
            diff_frame = {
                'Y': diff_viz, 
                'U': np.zeros_like(expected['U']), 
                'V': np.zeros_like(expected['V'])
            }
            video_processor.save_debug_frame(diff_frame, "diff_error", self.debug_dir)
    
    def validate(
        self, 
        validation_y4m: Path, 
        expected_frames: Dict[int, Dict[str, np.ndarray]], 
        pattern_count: int, 
        frames_per_pattern: int, 
        width: int, 
        height: int, 
        patches_mask: Dict[str, np.ndarray], 
        deviation: int = 4, 
        max_miss_percent: float = 0.002,
        intro_frames_count: int = 0,
        chroma_format: str = ChromaFormat.YUV_422,
        color_range: str = ColorRange.LIMITED
    ) -> bool:
        """
        Валидирует декодированное видео.
        
        Args:
            validation_y4m: Путь к Y4M файлу для валидации
            expected_frames: Словарь ожидаемых кадров
            pattern_count: Количество паттернов
            frames_per_pattern: Количество кадров на один паттерн
            width: Ширина кадра
            height: Высота кадра
            patches_mask: Маска патчей
            deviation: Максимальное допустимое отклонение значений
            max_miss_percent: Максимальный допустимый процент ошибок
            intro_frames_count: Количество вводных кадров для пропуска
            chroma_format: Формат цветовой субдискретизации
            color_range: Цветовой диапазон
            
        Returns:
            bool: Результат валидации (True - успешно, False - ошибка)
        """
        from tqdm import tqdm
        
        # Читаем заголовок Y4M для определения реальных параметров файла
        video_processor = VideoProcessor()
        file_params = video_processor.parse_y4m_header(validation_y4m)
        
        # Используем параметры из файла, если они отличаются от переданных
        actual_chroma_format = file_params.get('chroma_format', chroma_format)
        actual_color_range = file_params.get('color_range', color_range)
        actual_width = file_params.get('width', width)
        actual_height = file_params.get('height', height)
        
        if actual_width != width or actual_height != height:
            print(f"Предупреждение: Размеры в файле ({actual_width}x{actual_height}) "
                 f"отличаются от ожидаемых ({width}x{height})")
        
        if actual_chroma_format != chroma_format:
            print(f"Предупреждение: Формат в файле ({actual_chroma_format}) "
                 f"отличается от ожидаемого ({chroma_format})")
            
        if actual_color_range != color_range:
            print(f"Предупреждение: Цветовой диапазон в файле ({actual_color_range}) "
                 f"отличается от ожидаемого ({color_range})")
        
        with open(validation_y4m, 'rb') as f:
            # Пропускаем заголовок Y4M
            header = f.readline().decode('ascii')
                        
            # Пропускаем кадры вводной последовательности
            self._skip_intro_frames(f, video_processor, actual_width, actual_height, 
                                   intro_frames_count, actual_chroma_format)
            
            # Счетчики для статистики
            frames_checked = 0
            frames_valid = 0
            
            # Проходим по всем паттернам в том же порядке, что и при кодировании
            with tqdm(total=pattern_count, desc="Валидация паттернов") as pbar_patterns:
                for pattern_idx in range(pattern_count):
                    expected_frame = expected_frames[pattern_idx]
                    
                    # Для каждого кадра в этом паттерне
                    for frame_idx in range(frames_per_pattern):
                        # Читаем кадр с учетом формата
                        actual_frame = video_processor.read_y4m_frame(
                            f, actual_width, actual_height, actual_chroma_format)
                        
                        if actual_frame is None:
                            print(f"Ошибка чтения кадра (паттерн {pattern_idx}, кадр {frame_idx})")
                            continue
                        
                        # Сохраняем отладочные изображения
                        if self.debug_mode and self.debug_dir and frame_idx == 0:
                            video_processor.save_debug_frame(
                                actual_frame, f"decoded_pattern_{pattern_idx}", self.debug_dir)
                        
                        # Проверяем кадр с учетом формата
                        result, details = self.verify_frame(
                            expected_frame, actual_frame, patches_mask, 
                            deviation, max_miss_percent, actual_chroma_format)
                            
                        frames_checked += 1
                        
                        if result:
                            frames_valid += 1
                        else:
                            print(f"Ошибка валидации на паттерне {pattern_idx}, кадре {frame_idx}")
                            return False
                    
                    pbar_patterns.update(1)
            
            print(f"Валидация завершена: проверено {frames_checked} кадров, валидных {frames_valid}")
            return frames_valid == frames_checked
    
    def _skip_intro_frames(
        self, 
        file: BinaryIO, 
        video_processor: VideoProcessor, 
        width: int, 
        height: int, 
        intro_frames_count: int,
        chroma_format: str = ChromaFormat.YUV_422
    ) -> None:
        """
        Пропускает вводные кадры перед валидацией.
        
        Args:
            file: Файловый объект для чтения
            video_processor: Процессор видео
            width: Ширина кадра
            height: Высота кадра
            intro_frames_count: Количество кадров для пропуска
            chroma_format: Формат цветовой субдискретизации
        """
        if intro_frames_count > 0:
            print(f"Пропуск {intro_frames_count} вводных кадров...")
            for _ in range(intro_frames_count):
                _ = video_processor.read_y4m_frame(file, width, height, chroma_format)