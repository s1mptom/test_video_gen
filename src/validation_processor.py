"""Модуль для валидации видео."""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, BinaryIO

from .video_processor import VideoProcessor


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
        max_miss_percent: float = 0.002
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """
        Проверяет кадр с допустимым отклонением и процентом ошибок.
        
        Args:
            expected: Ожидаемый буфер кадра
            actual: Фактический буфер кадра
            patches_mask: Маска патчей (1 - патч, 0 - фон)
            deviation: Максимальное допустимое отклонение значений
            max_miss_percent: Максимальный допустимый процент ошибок
            
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
                raise ValueError(f"Размеры не совпадают: ожидалось {expected_plane.shape}, "
                               f"получено {actual_plane.shape}")
            
            # Расчет отклонения только для пикселей патчей (применяем маску)
            diff = np.abs(actual_plane.astype(int) - expected_plane.astype(int))
            
            # Применяем маску - учитываем только пиксели, где маска == 1
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
        intro_frames_count: int = 0
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
            
        Returns:
            bool: Результат валидации (True - успешно, False - ошибка)
        """
        from tqdm import tqdm
        
        # Читаем заголовок Y4M
        with open(validation_y4m, 'rb') as f:
            header = f.readline().decode('ascii')
            
            # Проверяем размеры
            width_match = re.search(r'W(\d+)', header)
            height_match = re.search(r'H(\d+)', header)
            
            if not width_match or not height_match:
                raise ValueError("Не удалось извлечь размеры из заголовка Y4M")
            
            width_val = int(width_match.group(1))
            height_val = int(height_match.group(1))
            
            if width_val != width or height_val != height:
                raise ValueError(f"Размеры не совпадают: ожидалось {width}x{height}, "
                               f"получено {width_val}x{height_val}")
            
            # Пропускаем кадры вводной последовательности
            video_processor = VideoProcessor()
            self._skip_intro_frames(f, video_processor, width_val, height_val, intro_frames_count)
            
            # Счетчики для статистики
            frames_checked = 0
            frames_valid = 0
            
            # Проходим по всем паттернам в том же порядке, что и при кодировании
            with tqdm(total=pattern_count, desc="Валидация паттернов") as pbar_patterns:
                for pattern_idx in range(pattern_count):
                    expected_frame = expected_frames[pattern_idx]
                    
                    # Для каждого кадра в этом паттерне
                    for frame_idx in range(frames_per_pattern):
                        # Читаем кадр
                        actual_frame = video_processor.read_y4m_frame(f, width_val, height_val)
                        if actual_frame is None:
                            print(f"Ошибка чтения кадра (паттерн {pattern_idx}, кадр {frame_idx})")
                            continue
                        
                        # Сохраняем отладочные изображения
                        if self.debug_mode and self.debug_dir and frame_idx == 0:
                            video_processor.save_debug_frame(
                                actual_frame, f"decoded_pattern_{pattern_idx}", self.debug_dir)
                        
                        # Проверяем кадр
                        result, details = self.verify_frame(
                            expected_frame, actual_frame, patches_mask, deviation, max_miss_percent)
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
        intro_frames_count: int
    ) -> None:
        """
        Пропускает вводные кадры перед валидацией.
        
        Args:
            file: Файловый объект для чтения
            video_processor: Процессор видео
            width: Ширина кадра
            height: Высота кадра
            intro_frames_count: Количество кадров для пропуска
        """
        if intro_frames_count > 0:
            print(f"Пропуск {intro_frames_count} вводных кадров...")
            for _ in range(intro_frames_count):
                _ = video_processor.read_y4m_frame(file, width, height)