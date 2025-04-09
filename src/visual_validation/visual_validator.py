"""Модуль для точной визуальной RGB-ориентированной валидации паттернов."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from src.pattern_metadata import PatternMetadataHandler
from src.utils.constants import (
    PATTERN_NUMBER_BITS, ColorRange, ChromaFormat, get_yuv_constants
)


class VisualValidationProcessor:
    """Класс для точной RGB-ориентированной визуальной валидации цветовых паттернов."""
    
    def __init__(
        self, 
        output_dir: str = "output", 
        debug_mode: bool = True, 
        debug_dir: Optional[Path] = None
    ):
        """
        Инициализирует процессор RGB визуальной валидации.
        
        Args:
            output_dir: Директория для выходных файлов
            debug_mode: Режим отладки
            debug_dir: Директория для отладочных файлов
        """
        self.output_dir = Path(output_dir)
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Создаем поддиректорию для маркеров
            self.marker_debug_dir = self.debug_dir / "rgb_markers"
            self.marker_debug_dir.mkdir(exist_ok=True)
        else:
            self.marker_debug_dir = None
        
        self.metadata_handler = PatternMetadataHandler(output_dir)
    
    def read_pattern_marker(
        self, 
        frame: np.ndarray, 
        pattern_generator,
        color_range: str = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Считывает маркер паттерна из технической строки в RGB кадре.
        
        Args:
            frame: RGB кадр
            pattern_generator: Генератор паттернов с информацией о маркерах
            color_range: Цветовой диапазон (если None, определяется из pattern_generator)
            
        Returns:
            Tuple[int, Dict[str, Any]]: Считанный номер паттерна (-1 в случае ошибки) и диагностика
        """
        # Определяем цветовой диапазон
        if color_range is None and hasattr(pattern_generator, 'color_range'):
            color_range = pattern_generator.color_range
        
        # Инициализация диагностических данных
        diagnostics = {
            "pattern_bits": [],
            "checksum_bits": [],
            "threshold": 0,
            "anchor_check": False
        }
        
        # Получаем правильные маркерные патчи из технической строки
        marker_patches = []
        for marker_idx in pattern_generator.marker_indices:
            marker_patches.append(pattern_generator.patch_coords[marker_idx])
        
        # Проверяем, что маркеры находятся в правильном месте (в технической строке)
        tech_row_y = pattern_generator.tech_row * (pattern_generator.patch_size + pattern_generator.patch_gap)
        if self.debug_mode:
            print(f"Technical row should start at Y: {tech_row_y}")
            first_patch = marker_patches[0]
            print(f"First marker patch Y range: {first_patch.y_range}")
            print(f"Technical row position check: {abs(first_patch.y_range[0] - tech_row_y) < 10}")
        
        # Проверка якорных маркеров (в RGB)
        anchor_start_valid = self._check_rgb_anchor_pattern(frame, marker_patches[0:2], is_start=True)
        anchor_end_valid = self._check_rgb_anchor_pattern(frame, marker_patches[-2:], is_start=False)
        
        if not anchor_start_valid or not anchor_end_valid:
            diagnostics["error"] = {
                "type": "anchor",
                "message": f"Invalid anchor markers. Start anchor valid: {anchor_start_valid}, End anchor valid: {anchor_end_valid}"
            }
            diagnostics["anchor_check"] = False
            
            # Сохраняем отладочное изображение
            if self.debug_mode and self.marker_debug_dir:
                self._save_rgb_marker_debug_image(
                    frame, marker_patches, "", "", 
                    "", diagnostics,
                    self.marker_debug_dir / f"marker_invalid_anchors.png"
                )
            
            return -1, diagnostics
        
        diagnostics["anchor_check"] = True
        
        # Извлекаем все патчи идентификатора и контрольной суммы
        id_patches = []
        for i in range(2, 14):  # Патчи идентификатора (2-13)
            patch = marker_patches[i - 0]  # -0 поскольку marker_patches уже содержит только маркерные патчи
            y_values = self._extract_gray_value(frame, patch.y_range[0], patch.y_range[1], 
                            patch.x_range[0], patch.x_range[1])
            id_patches.append(y_values)
        
        checksum_patches = []
        for i in range(14, 18):  # Патчи контрольной суммы (14-17)
            patch = marker_patches[i - 0]  # -0 поскольку marker_patches уже содержит только маркерные патчи
            y_values = self._extract_gray_value(frame, patch.y_range[0], patch.y_range[1], 
                            patch.x_range[0], patch.x_range[1])
            checksum_patches.append(y_values)
        
        # Используем фиксированный порог для бинаризации
        threshold = 128
        diagnostics["threshold"] = float(threshold)
        
        # Считываем биты идентификатора
        binary_str = ""
        for i, patch in enumerate(id_patches):
            mean_value = float(patch)
            bit = '1' if mean_value > threshold else '0'
            binary_str += bit
            
            diagnostics["pattern_bits"].append({
                "index": i + 2,
                "mean": mean_value,
                "bit": bit
            })
        
        # Проверка на полностью черный шаблон (все нули)
        if binary_str == "0" * len(binary_str) or binary_str == "000000000000":
            diagnostics["error"] = {
                "type": "all_zeros",
                "message": "Pattern is all zeros (completely black). This might be a lead-in frame or indicate improper detection."
            }
            return -1, diagnostics
        
        # Считываем биты контрольной суммы
        checksum_binary = ""
        for i, patch in enumerate(checksum_patches):
            mean_value = float(patch)
            bit = '1' if mean_value > threshold else '0'
            checksum_binary += bit
            
            diagnostics["checksum_bits"].append({
                "index": i + 14,
                "mean": mean_value,
                "bit": bit
            })
        
        # Вычисляем ожидаемую контрольную сумму точно так же, как в PatternGenerator
        expected_checksum = 0
        for i in range(0, len(binary_str), 3):
            end = min(i + 3, len(binary_str))
            chunk = int(binary_str[i:end], 2)
            expected_checksum ^= chunk
        
        expected_checksum_binary = format(expected_checksum, '04b')
        
        # Проверяем контрольную сумму
        if checksum_binary != expected_checksum_binary:
            error_msg = f"Ошибка контрольной суммы: ожидалось {expected_checksum_binary}, получено {checksum_binary}"
            print(error_msg)
            
            diagnostics["error"] = {
                "type": "checksum",
                "expected": expected_checksum_binary,
                "received": checksum_binary,
                "message": error_msg,
                "binary_str": binary_str
            }
            
            # Сохраняем отладочное изображение
            if self.debug_mode and self.marker_debug_dir:
                self._save_rgb_marker_debug_image(
                    frame, marker_patches, binary_str, checksum_binary, 
                    expected_checksum_binary, diagnostics,
                    self.marker_debug_dir / f"marker_error_{binary_str}.png"
                )
            
            return -1, diagnostics
        
        # Преобразуем двоичную строку в число
        try:
            pattern_idx = int(binary_str, 2)
            
            # Сохраняем отладочное изображение для успешных маркеров тоже
            if self.debug_mode and self.marker_debug_dir:
                self._save_rgb_marker_debug_image(
                    frame, marker_patches, binary_str, checksum_binary, 
                    expected_checksum_binary, diagnostics,
                    self.marker_debug_dir / f"marker_success_{pattern_idx}.png"
                )
                
            return pattern_idx, diagnostics
        except ValueError as e:
            error_msg = f"Ошибка преобразования строки '{binary_str}' в число: {str(e)}"
            print(error_msg)
            
            diagnostics["error"] = {
                "type": "conversion",
                "binary_string": binary_str,
                "message": error_msg
            }
            
            return -1, diagnostics
    
    def _extract_gray_value(self, frame, y1, y2, x1, x2):
        """
        Извлекает среднее значение яркости (серого) из RGB фрагмента.
        """
        patch = frame[y1:y2, x1:x2]
        # Преобразуем RGB в оттенки серого (средняя яркость)
        gray_value = np.mean(patch.mean(axis=2))
        return gray_value

    def _check_rgb_anchor_pattern(
        self, 
        frame: np.ndarray, 
        anchor_patches: List[Any], 
        is_start: bool = True
    ) -> bool:
        """
        Проверяет якорные маркеры на соответствие ожидаемому шаблону в RGB.
        
        Args:
            frame: RGB кадр
            anchor_patches: Список якорных патчей (2 штуки)
            is_start: True для начальных якорей, False для конечных
        
        Returns:
            bool: True если якорные маркеры валидны
        """
        if len(anchor_patches) != 2:
            print(f"Неверное количество якорных патчей: {len(anchor_patches)}")
            return False
        
        # Извлекаем значения из якорных патчей
        anchor_values = []
        
        for patch in anchor_patches:
            # Получаем RGB значения патча
            patch_rgb = frame[patch.y_range[0]:patch.y_range[1], 
                           patch.x_range[0]:patch.x_range[1]]
            
            # Делим патч на 4 квадранта
            h, w, _ = patch_rgb.shape
            half_h, half_w = h // 2, w // 2
            
            quadrants = [
                patch_rgb[:half_h, :half_w],       # верхний левый
                patch_rgb[:half_h, half_w:],       # верхний правый
                patch_rgb[half_h:, :half_w],       # нижний левый
                patch_rgb[half_h:, half_w:]        # нижний правый
            ]
            
            # Преобразуем в значения яркости
            quadrant_means = [np.mean(q.mean(axis=2)) for q in quadrants]
            anchor_values.append(quadrant_means)
        
        # Проверка паттерна якорей
        # Начальная метка: 1-й патч (ЧБ/БЧ), 2-й патч (БЧ/ЧБ)
        # Конечная метка: 1-й патч (БЧ/ЧБ), 2-й патч (ЧБ/БЧ)
        
        # Пороговое значение для определения черного и белого
        threshold = 128
        
        # Проверка контраста квадрантов
        valid = True
        
        for i, values in enumerate(anchor_values):
            # Для начальных якорей
            if is_start:
                if i == 0:  # Первый патч должен быть ЧБ/БЧ
                    valid = valid and (values[0] < threshold and values[3] < threshold)  # ЧЧ
                    valid = valid and (values[1] > threshold and values[2] > threshold)  # ББ
                else:  # Второй патч должен быть БЧ/ЧБ
                    valid = valid and (values[0] > threshold and values[3] > threshold)  # ББ
                    valid = valid and (values[1] < threshold and values[2] < threshold)  # ЧЧ
            else:  # Для конечных якорей (инверсия)
                if i == 0:  # Первый патч должен быть БЧ/ЧБ
                    valid = valid and (values[0] > threshold and values[3] > threshold)  # ББ
                    valid = valid and (values[1] < threshold and values[2] < threshold)  # ЧЧ
                else:  # Второй патч должен быть ЧБ/БЧ
                    valid = valid and (values[0] < threshold and values[3] < threshold)  # ЧЧ
                    valid = valid and (values[1] > threshold and values[2] > threshold)  # ББ
        
        # Проверка контраста между квадрантами (должна быть существенная разница)
        min_contrast = 50  # Минимальная разница между черным и белым значениями
        
        for values in anchor_values:
            black_values = [values[0], values[3]] if values[0] < threshold else [values[1], values[2]]
            white_values = [values[1], values[2]] if values[0] < threshold else [values[0], values[3]]
            
            avg_black = sum(black_values) / len(black_values)
            avg_white = sum(white_values) / len(white_values)
            
            valid = valid and (avg_white - avg_black > min_contrast)
        
        if not valid and self.debug_mode:
            print(f"Якорные маркеры не прошли проверку: {anchor_values}")
        
        return valid
        
    def _save_rgb_marker_debug_image(
        self, 
        frame: np.ndarray, 
        marker_patches: List[Any],
        binary_str: str,
        checksum_binary: str,
        expected_checksum: str,
        diagnostics: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Сохраняет подробное отладочное изображение маркера из RGB кадра.
        
        Args:
            frame: RGB кадр
            marker_patches: Список индексов маркерных патчей
            binary_str: Двоичная строка идентификатора
            checksum_binary: Двоичная строка контрольной суммы
            expected_checksum: Ожидаемая контрольная сумма
            diagnostics: Диагностические данные
            output_path: Путь для сохранения изображения
        """
        # Создаем копию кадра для рисования
        debug_frame = frame.copy()
        
        # Определяем область, содержащую все маркерные патчи
        tech_row_idx = marker_patches[0].y_range[0]  # Верхняя координата Y технической строки
        tech_row_height = marker_patches[0].y_range[1] - marker_patches[0].y_range[0]
        
        # Вырезаем регион, содержащий техническую строку, с небольшим запасом сверху и снизу
        padding = tech_row_height // 2
        h, w, _ = frame.shape
        tech_region = debug_frame[max(0, tech_row_idx - padding):min(h, tech_row_idx + tech_row_height + padding), :].copy()
        
        # Увеличиваем в 2 раза для лучшей видимости
        scale_factor = 2
        tech_region_large = cv2.resize(
            tech_region, 
            (tech_region.shape[1], tech_region.shape[0] * scale_factor),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Пороговое значение для определения бита
        threshold = 128
        
        # Рисуем информацию о маркерных патчах
        # Патчи идентификатора
        for i in range(2, 14):
            patch_idx = i - 0  # Смещение -0, так как marker_patches уже содержит только маркерные патчи
            patch = marker_patches[patch_idx]
            
            # Координаты патча
            x1, x2 = patch.x_range
            
            # Бит и его значение
            bit_info = next((b for b in diagnostics["pattern_bits"] if b["index"] == i), None)
            if bit_info:
                bit = bit_info["bit"]
                mean = bit_info["mean"]
                
                # Рисуем прямоугольник вокруг патча
                color = (0, 255, 0) if bit == '1' else (0, 0, 255)  # Зеленый для 1, Красный для 0
                cv2.rectangle(tech_region_large, 
                            (x1, 0), 
                            (x2, tech_region_large.shape[0]), 
                            color, 2)
                
                # Добавляем текст с битом и средним значением
                y_pos = tech_region_large.shape[0] // 2
                cv2.putText(tech_region_large, bit, 
                          (x1 + 5, y_pos - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(tech_region_large, f"{mean:.1f}", 
                          (x1 + 5, y_pos + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Патчи контрольной суммы
        for i in range(14, 18):
            patch_idx = i - 0  # Смещение -0, так как marker_patches уже содержит только маркерные патчи
            patch = marker_patches[patch_idx]
            
            # Координаты патча
            x1, x2 = patch.x_range
            
            # Бит и его значение
            bit_info = next((b for b in diagnostics["checksum_bits"] if b["index"] == i), None)
            if bit_info:
                bit = bit_info["bit"]
                mean = bit_info["mean"]
                
                # Рисуем прямоугольник вокруг патча
                color = (255, 255, 0) if bit == '1' else (255, 0, 255)  # Желтый для 1, Пурпурный для 0
                cv2.rectangle(tech_region_large, 
                            (x1, 0), 
                            (x2, tech_region_large.shape[0]), 
                            color, 2)
                
                # Добавляем текст с битом и средним значением
                y_pos = tech_region_large.shape[0] // 2
                cv2.putText(tech_region_large, bit, 
                          (x1 + 5, y_pos - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(tech_region_large, f"{mean:.1f}", 
                          (x1 + 5, y_pos + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Добавляем информацию о пороге и результатах
        # Создаем область для текста сверху
        info_height = 150
        info_img = np.ones((info_height, tech_region_large.shape[1], 3), dtype=np.uint8) * 50  # Темно-серый фон
        
        # Добавляем полную информацию о маркере
        y_pos = 30
        cv2.putText(info_img, f"ID: {binary_str} = {int(binary_str, 2) if len(binary_str) > 0 else 'Invalid'}", 
                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        cv2.putText(info_img, f"Checksum: {checksum_binary} (Expected: {expected_checksum})", 
                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                  (0, 255, 0) if checksum_binary == expected_checksum else (0, 0, 255), 1)
        y_pos += 30
        
        cv2.putText(info_img, f"Threshold: {threshold}", 
                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
        if "error" in diagnostics:
            error_type = diagnostics["error"]["type"]
            error_msg = diagnostics["error"].get("message", "Unknown error")
            cv2.putText(info_img, f"Error: {error_type} - {error_msg}", 
                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        # Объединяем области с информацией и изображением маркеров
        final_img = np.vstack([info_img, tech_region_large])
        
        # Сохраняем изображение
        cv2.imwrite(str(output_path), final_img)
    
    def extract_rgb_patch_values(
        self, 
        frame: np.ndarray, 
        patches_metadata: List[Dict[str, Any]]
    ) -> List[Tuple[float, float, float]]:
        """
        Извлекает средние значения RGB из патчей.
        
        Args:
            frame: RGB кадр
            patches_metadata: Метаданные о патчах
            
        Returns:
            List[Tuple[float, float, float]]: Список средних значений RGB для каждого патча
        """
        patch_values = []
        
        for patch in patches_metadata:
            y_range = tuple(patch["y_range"])
            x_range = tuple(patch["x_range"])
            
            # Игнорируем крайние пиксели для более точного измерения
            border = 2
            y_min, y_max = y_range
            x_min, x_max = x_range
            
            # Проверяем размер патча
            if y_max - y_min > 2*border and x_max - x_min > 2*border:
                y_min += border
                y_max -= border
                x_min += border
                x_max -= border
            
            # Проверяем, что координаты в допустимых пределах
            h, w, _ = frame.shape
            
            y_min = max(0, min(y_min, h - 1))
            y_max = max(y_min + 1, min(y_max, h))
            x_min = max(0, min(x_min, w - 1))
            x_max = max(x_min + 1, min(x_max, w))
            
            # Извлекаем патч
            rgb_patch = frame[y_min:y_max, x_min:x_max]
            
            # Создаем гауссово ядро для взвешенного усреднения
            patch_h, patch_w, _ = rgb_patch.shape
            y_grid, x_grid = np.mgrid[0:patch_h, 0:patch_w]
            center_y, center_x = patch_h//2, patch_w//2
            sigma = max(patch_h, patch_w) / 5.0
            
            weights = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2*sigma**2))
            weights = weights[:, :, np.newaxis]  # Расширяем для трех каналов
            weights = weights / weights.sum()
            
            # Взвешенное среднее для каждого канала RGB
            r_mean = float(np.sum(rgb_patch[:, :, 0] * weights[:, :, 0]))
            g_mean = float(np.sum(rgb_patch[:, :, 1] * weights[:, :, 0]))
            b_mean = float(np.sum(rgb_patch[:, :, 2] * weights[:, :, 0]))
            
            patch_values.append((r_mean, g_mean, b_mean))
        
        return patch_values
    
    def compare_rgb_patch_values(
        self, 
        extracted_values: List[Tuple[float, float, float]], 
        expected_colors: List[List[int]], 
        deviation: int = 10
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Сравнивает извлеченные значения RGB патчей с ожидаемыми.
        
        Args:
            extracted_values: Извлеченные значения RGB
            expected_colors: Ожидаемые RGB значения
            deviation: Допустимое отклонение
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: Результат сравнения и статистика по каждому патчу
        """
        if len(extracted_values) != len(expected_colors):
            print(f"Несоответствие количества патчей: извлечено {len(extracted_values)}, ожидалось {len(expected_colors)}")
            return False, []
        
        comparison_results = []
        all_valid = True
        errors_count = 0
        
        for i, (extracted, expected) in enumerate(zip(extracted_values, expected_colors)):
            r_extracted, g_extracted, b_extracted = extracted
            r_expected, g_expected, b_expected = expected
            
            # Вычисляем разницу по каждому каналу
            r_diff = abs(r_extracted - r_expected)
            g_diff = abs(g_extracted - g_expected)
            b_diff = abs(b_extracted - b_expected)
            
            # Проверяем, находится ли разница в пределах допустимого отклонения
            is_valid = (r_diff <= deviation and g_diff <= deviation and b_diff <= deviation)
            
            # Собираем результаты сравнения
            result = {
                "patch_idx": i,
                "r_expected": r_expected,
                "g_expected": g_expected,
                "b_expected": b_expected,
                "r_extracted": r_extracted,
                "g_extracted": g_extracted,
                "b_extracted": b_extracted,
                "r_diff": r_diff,
                "g_diff": g_diff,
                "b_diff": b_diff,
                "is_valid": is_valid
            }
            
            comparison_results.append(result)
            
            if not is_valid:
                errors_count += 1
                if errors_count <= 5:  # Ограничиваем вывод ошибок
                    print(f"Ошибка патча {i}: R={r_diff:.1f}, G={g_diff:.1f}, B={b_diff:.1f} > {deviation}")
                all_valid = False
        
        if errors_count > 5:
            print(f"... и еще {errors_count - 5} ошибок патчей")
            
        return all_valid, comparison_results
    
    def create_rgb_comparison_visualization(
        self, 
        frame: np.ndarray, 
        comparison_results: List[Dict[str, Any]], 
        patches_metadata: List[Dict[str, Any]], 
        output_path: Path
    ) -> None:
        """
        Создает визуализацию сравнения RGB патчей.
        
        Args:
            frame: RGB кадр
            comparison_results: Результаты сравнения
            patches_metadata: Метаданные о патчах
            output_path: Путь для сохранения визуализации
        """
        # Создаем копию для визуализации
        viz_img = frame.copy()
        
        # Считаем количество успешных и ошибочных патчей
        valid_count = sum(1 for r in comparison_results if r.get("is_valid", False))
        invalid_count = len(comparison_results) - valid_count
        
        # Добавляем информацию о результатах
        cv2.putText(viz_img, f"Valid: {valid_count}/{len(comparison_results)} ({valid_count/len(comparison_results)*100:.1f}%)", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отмечаем патчи с ошибками
        error_count = 0
        for i, result in enumerate(comparison_results):
            if i >= len(patches_metadata):
                continue
                
            patch = patches_metadata[i]
            y1, y2 = patch["y_range"]
            x1, x2 = patch["x_range"]
            
            # Рисуем рамку вокруг патча
            if result.get("is_valid", True):
                continue  # Пропускаем валидные патчи, чтобы не загромождать изображение
            
            error_count += 1
            if error_count > 100:  # Ограничиваем количество отображаемых ошибок
                continue
                
            # Красная рамка для невалидных патчей
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            # Добавляем текст с разницей
            if "r_diff" in result and "g_diff" in result and "b_diff" in result:
                if (error_count % 10 == 0):  # Отображаем текст только для каждого 10-го патча
                    diff_text = f"R:{result['r_diff']:.1f} G:{result['g_diff']:.1f} B:{result['b_diff']:.1f}"
                    cv2.putText(viz_img, diff_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Сохраняем визуализацию
        cv2.imwrite(str(output_path), viz_img)
    
    def rgb_visual_validate(
        self,
        video_path: Path, 
        pattern_generator, 
        frames_per_pattern: int, 
        intro_frames_count: int = 0, 
        deviation: int = 10,
        max_miss_percent: float = 0.002
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Выполняет RGB-ориентированную визуальную валидацию видео с паттернами.
        
        Args:
            video_path: Путь к видеофайлу
            pattern_generator: Генератор паттернов
            frames_per_pattern: Количество кадров на один паттерн
            intro_frames_count: Количество вводных кадров для пропуска
            deviation: Допустимое отклонение значений RGB
            max_miss_percent: Максимальный допустимый процент ошибок
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Результат валидации и статистика
        """
        # Статистика валидации
        validation_stats = {
            "total_frames": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "detected_patterns": [],
            "error_patterns": []
        }
        
        # Создаем директорию для визуализаций, если нужно
        if self.debug_mode and self.debug_dir:
            visual_debug_dir = self.debug_dir / "rgb_visual_validation"
            visual_debug_dir.mkdir(exist_ok=True)
        
        # Открываем видео напрямую через OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return False, validation_stats
        
        # Пропускаем кадры вводной последовательности
        for _ in range(intro_frames_count):
            cap.read()
            
        # ИЗМЕНЕНИЕ: Используем диапазон с 1 до patterns_count+1
        print(f"Validating patterns with indices from 1 to {pattern_generator.patterns_count}")
        
        with tqdm(total=pattern_generator.patterns_count, desc="RGB визуальная валидация") as pbar:
            for pattern_idx in range(1, pattern_generator.patterns_count + 1):  # Начинаем с 1, а не с 0
                # Получаем метаданные паттерна
                pattern_metadata = self.metadata_handler.load_pattern_metadata(pattern_idx)
                if not pattern_metadata:
                    print(f"Ошибка: метаданные для паттерна {pattern_idx} не найдены")
                    validation_stats["error_patterns"].append({
                        "pattern_idx": pattern_idx,
                        "error": "Метаданные не найдены"
                    })
                    continue
                
                # Получаем информацию о патчах и цветах
                patches_metadata = pattern_metadata.get("patches", [])
                expected_colors = pattern_metadata.get("colors", [])
                
                # Обрабатываем только первый кадр из каждого паттерна для валидации
                for frame_idx in range(frames_per_pattern):
                    ret, frame = cap.read()
                    validation_stats["total_frames"] += 1
                    
                    if not ret:
                        print(f"Ошибка чтения кадра (паттерн {pattern_idx}, кадр {frame_idx})")
                        validation_stats["invalid_frames"] += 1
                        continue
                    
                    # Преобразуем BGR (OpenCV) в RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Для первого кадра каждого паттерна выполняем проверку
                    if frame_idx == 0:
                        # Считываем маркер паттерна
                        detected_pattern_idx, marker_diagnostics = self.read_pattern_marker(
                            frame_rgb, pattern_generator)
                        
                        if detected_pattern_idx == -1:
                            print(f"Ошибка чтения маркера в кадре (ожидаемый паттерн {pattern_idx})")
                            validation_stats["invalid_frames"] += 1
                            validation_stats["error_patterns"].append({
                                "pattern_idx": pattern_idx,
                                "error": "Ошибка чтения маркера",
                                "marker_diagnostics": marker_diagnostics
                            })
                            continue
                        
                        # Проверяем, совпадает ли считанный паттерн с ожидаемым
                        if detected_pattern_idx != pattern_idx:
                            print(f"Несоответствие номера паттерна: обнаружен {detected_pattern_idx}, ожидался {pattern_idx}")
                            validation_stats["invalid_frames"] += 1
                            validation_stats["error_patterns"].append({
                                "pattern_idx": pattern_idx,
                                "detected_pattern_idx": detected_pattern_idx,
                                "error": "Несоответствие номера паттерна"
                            })
                            continue
                        
                        # Извлекаем RGB значения патчей
                        extracted_values = self.extract_rgb_patch_values(
                            frame_rgb, patches_metadata)
                        
                        # Сравниваем с ожидаемыми RGB значениями
                        is_valid, comparison_results = self.compare_rgb_patch_values(
                            extracted_values, expected_colors, deviation)
                        
                        if is_valid:
                            validation_stats["valid_frames"] += 1
                            validation_stats["detected_patterns"].append({
                                "pattern_idx": pattern_idx,
                                "is_valid": True
                            })
                        else:
                            validation_stats["invalid_frames"] += 1
                            validation_stats["error_patterns"].append({
                                "pattern_idx": pattern_idx,
                                "is_valid": False,
                                "comparison_results": comparison_results
                            })
                            
                            # Создаем визуализацию для отладки
                            if self.debug_mode and self.debug_dir:
                                viz_path = visual_debug_dir / f"rgb_pattern_{pattern_idx}_errors.png"
                                self.create_rgb_comparison_visualization(
                                    frame_rgb, comparison_results, patches_metadata, viz_path)
                    else:
                        # Остальные кадры в паттерне - считаем валидными, если первый валидный
                        if validation_stats["detected_patterns"] and validation_stats["detected_patterns"][-1]["pattern_idx"] == pattern_idx:
                            validation_stats["valid_frames"] += 1
                        else:
                            validation_stats["invalid_frames"] += 1
                
                pbar.update(1)
        
        # Закрываем видео
        cap.release()
            
        # Вычисляем общий результат
        total_expected_frames = pattern_generator.patterns_count * frames_per_pattern
        validation_success = (validation_stats["valid_frames"] / total_expected_frames) >= (1 - max_miss_percent)
        
        print(f"RGB визуальная валидация завершена: "
                f"проверено {validation_stats['total_frames']} кадров, "
                f"валидных {validation_stats['valid_frames']}, "
                f"недействительных {validation_stats['invalid_frames']}")
        
        # Сохраняем результаты валидации
        if self.debug_mode and self.debug_dir:
            validation_results_path = self.debug_dir / "rgb_visual_validation_results.json"
            import json
            with open(validation_results_path, 'w') as f:
                json.dump(validation_stats, f, indent=2)
        
        return validation_success, validation_stats