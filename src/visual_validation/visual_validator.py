"""Модуль для точной визуальной валидации паттернов."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from ..pattern_metadata import PatternMetadataHandler
from ..utils.color_transforms import rgb_to_yuv_bt709, yuv_to_rgb_bt709
from ..utils.constants import Y_BLACK, Y_WHITE, UV_NEUTRAL, PATTERN_NUMBER_BITS


class VisualValidationProcessor:
    """Класс для точной визуальной валидации цветовых паттернов."""
    
    def __init__(
        self, 
        output_dir: str = "output", 
        debug_mode: bool = True, 
        debug_dir: Optional[Path] = None
    ):
        """
        Инициализирует процессор визуальной валидации.
        
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
            self.marker_debug_dir = self.debug_dir / "markers"
            self.marker_debug_dir.mkdir(exist_ok=True)
        else:
            self.marker_debug_dir = None
        
        self.metadata_handler = PatternMetadataHandler(output_dir)
    
    def read_pattern_marker(self, frame: Dict[str, np.ndarray], pattern_generator) -> Tuple[int, Dict[str, Any]]:
        """
        Считывает маркер паттерна из технической строки с улучшенной проверкой якорных маркеров.
        
        Args:
            frame: Буфер кадра
            pattern_generator: Генератор паттернов с информацией о маркерах
            
        Returns:
            Tuple[int, Dict[str, Any]]: Считанный номер паттерна (-1 в случае ошибки) и диагностика
        """
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
        
        # Проверка якорных маркеров
        anchor_start_valid = self._check_anchor_pattern(frame, marker_patches[0:2], is_start=True)
        anchor_end_valid = self._check_anchor_pattern(frame, marker_patches[-2:], is_start=False)
        
        if not anchor_start_valid or not anchor_end_valid:
            diagnostics["error"] = {
                "type": "anchor",
                "message": f"Invalid anchor markers. Start anchor valid: {anchor_start_valid}, End anchor valid: {anchor_end_valid}"
            }
            diagnostics["anchor_check"] = False
            
            # Сохраняем отладочное изображение
            if self.debug_mode and self.marker_debug_dir:
                self._save_marker_debug_image(
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
            y_values = frame['Y'][patch.y_range[0]:patch.y_range[1], 
                            patch.x_range[0]:patch.x_range[1]]
            id_patches.append(y_values)
        
        checksum_patches = []
        for i in range(14, 18):  # Патчи контрольной суммы (14-17)
            patch = marker_patches[i - 0]  # -0 поскольку marker_patches уже содержит только маркерные патчи
            y_values = frame['Y'][patch.y_range[0]:patch.y_range[1], 
                            patch.x_range[0]:patch.x_range[1]]
            checksum_patches.append(y_values)
        
        # Используем фиксированный порог, основанный на известных значениях Y_BLACK и Y_WHITE
        threshold = (Y_BLACK + Y_WHITE) / 2
        diagnostics["threshold"] = float(threshold)
        
        # Считываем биты идентификатора
        binary_str = ""
        for i, patch in enumerate(id_patches):
            mean_value = float(np.mean(patch))
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
            mean_value = float(np.mean(patch))
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
                self._save_marker_debug_image(
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
                self._save_marker_debug_image(
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

    def _check_anchor_pattern(self, frame: Dict[str, np.ndarray], anchor_patches: List[Any], is_start: bool) -> bool:
        """
        Проверяет якорные маркеры на соответствие ожидаемому шаблону.
        
        Args:
            frame: Буфер кадра
            anchor_patches: Список якорных патчей (2 штуки)
            is_start: True для начальных якорей, False для конечных
        
        Returns:
            bool: True если якорные маркеры валидны
        """
        if len(anchor_patches) != 2:
            print(f"Неверное количество якорных патчей: {len(anchor_patches)}")
            return False
        
        # Извлекаем Y-значения из якорных патчей
        anchor_values = []
        
        for patch in anchor_patches:
            y_values = frame['Y'][patch.y_range[0]:patch.y_range[1], 
                            patch.x_range[0]:patch.x_range[1]]
            
            # Делим патч на 4 квадранта
            h, w = y_values.shape
            half_h, half_w = h // 2, w // 2
            
            quadrants = [
                y_values[:half_h, :half_w],       # верхний левый
                y_values[:half_h, half_w:],       # верхний правый
                y_values[half_h:, :half_w],       # нижний левый
                y_values[half_h:, half_w:]        # нижний правый
            ]
            
            quadrant_means = [np.mean(q) for q in quadrants]
            anchor_values.append(quadrant_means)
        
        # Проверка паттерна якорей
        # Начальная метка: 1-й патч (ЧБ/БЧ), 2-й патч (БЧ/ЧБ)
        # Конечная метка: 1-й патч (БЧ/ЧБ), 2-й патч (ЧБ/БЧ)
        
        # Пороговое значение для определения черного и белого
        threshold = (Y_BLACK + Y_WHITE) / 2
        
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
        min_contrast = 20  # Минимальная разница между черным и белым значениями
        
        for values in anchor_values:
            black_values = [values[0], values[3]] if values[0] < threshold else [values[1], values[2]]
            white_values = [values[1], values[2]] if values[0] < threshold else [values[0], values[3]]
            
            avg_black = sum(black_values) / len(black_values)
            avg_white = sum(white_values) / len(white_values)
            
            valid = valid and (avg_white - avg_black > min_contrast)
        
        if not valid and self.debug_mode:
            print(f"Якорные маркеры не прошли проверку: {anchor_values}")
        
        return valid
        
    def _save_marker_debug_image(
        self, 
        frame: Dict[str, np.ndarray], 
        marker_patches: List[Any],
        binary_str: str,
        checksum_binary: str,
        expected_checksum: str,
        diagnostics: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Сохраняет подробное отладочное изображение маркера.
        
        Args:
            frame: Буфер кадра
            marker_patches: Список индексов маркерных патчей
            binary_str: Двоичная строка идентификатора
            checksum_binary: Двоичная строка контрольной суммы
            expected_checksum: Ожидаемая контрольная сумма
            diagnostics: Диагностические данные
            output_path: Путь для сохранения изображения
        """
        # Преобразуем YUV в RGB для визуализации
        h, w = frame['Y'].shape
        u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
        v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
        
        rgb = yuv_to_rgb_bt709(frame['Y'], u_resized, v_resized)
        
        # Создаем увеличенное изображение для лучшей видимости
        # Определяем область, содержащую все маркерные патчи
        tech_row_idx = marker_patches[0].y_range[0]  # Верхняя координата Y технической строки
        tech_row_height = marker_patches[0].y_range[1] - marker_patches[0].y_range[0]
        
        # Вырезаем регион, содержащий техническую строку, с небольшим запасом сверху и снизу
        padding = tech_row_height // 2
        tech_region = rgb[max(0, tech_row_idx - padding):min(h, tech_row_idx + tech_row_height + padding), :].copy()
        
        # Увеличиваем в 2 раза для лучшей видимости
        scale_factor = 2
        tech_region_large = cv2.resize(
            tech_region, 
            (tech_region.shape[1], tech_region.shape[0] * scale_factor),
            interpolation=cv2.INTER_NEAREST
        )
        
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
        
        cv2.putText(info_img, f"Threshold: {diagnostics['threshold']:.1f}", 
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
    
    def extract_patch_values_422(self, frame: Dict[str, np.ndarray], patches_metadata: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
        """
        Извлекает средние значения YUV из патчей с учётом формата 422.
        
        Args:
            frame: Буфер кадра
            patches_metadata: Метаданные о патчах
            
        Returns:
            List[Tuple[float, float, float]]: Список средних значений YUV для каждого патча
        """
        patch_values = []
        
        for patch in patches_metadata:
            y_range = tuple(patch["y_range"])
            x_range = tuple(patch["x_range"])
            y_uv_range = tuple(patch["y_uv_range"])
            x_uv_range = tuple(patch["x_uv_range"])
            
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
            
            # Для YUV422 нужно корректировать только X-координаты для UV
            y_uv_min, y_uv_max = y_uv_range
            x_uv_min, x_uv_max = x_uv_range
            
            if x_uv_max - x_uv_min > 2:
                x_uv_min += 1
                x_uv_max -= 1
            
            # Извлекаем патчи
            y_patch = frame['Y'][y_min:y_max, x_min:x_max]
            u_patch = frame['U'][y_uv_min:y_uv_max, x_uv_min:x_uv_max]
            v_patch = frame['V'][y_uv_min:y_uv_max, x_uv_min:x_uv_max]
            
            # Создаем гауссово ядро для взвешенного усреднения
            h, w = y_patch.shape
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            center_y, center_x = h//2, w//2
            sigma = max(h, w) / 5.0
            
            weights = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2*sigma**2))
            weights /= weights.sum()
            
            # Взвешенное среднее для Y
            y_mean = float(np.sum(y_patch * weights))
            
            # Адаптируем веса для UV (в 422 нужно изменить только по ширине)
            h_uv, w_uv = u_patch.shape
            if h_uv != h:
                weights_uv = cv2.resize(weights, (w_uv, h_uv))
            else:
                # В случае 422 высота такая же, нужно изменить только ширину
                weights_uv = weights[:, ::2] if w_uv*2 == w else cv2.resize(weights, (w_uv, h_uv))
            
            weights_uv /= weights_uv.sum()
            
            # Взвешенное среднее для U и V
            u_mean = float(np.sum(u_patch * weights_uv))
            v_mean = float(np.sum(v_patch * weights_uv))
            
            patch_values.append((y_mean, u_mean, v_mean))
        
        return patch_values
    
    def compare_patch_values(
        self, 
        extracted_values: List[Tuple[float, float, float]], 
        expected_values: List[Dict[str, Any]], 
        deviation: int = 4
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Сравнивает извлеченные значения патчей с ожидаемыми.
        
        Args:
            extracted_values: Извлеченные значения YUV
            expected_values: Ожидаемые значения из метаданных
            deviation: Допустимое отклонение
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: Результат сравнения и статистика по каждому патчу
        """
        if len(extracted_values) != len(expected_values):
            print(f"Несоответствие количества патчей: извлечено {len(extracted_values)}, ожидалось {len(expected_values)}")
            return False, []
        
        comparison_results = []
        all_valid = True
        errors_count = 0
        
        for i, (extracted, expected) in enumerate(zip(extracted_values, expected_values)):
            y_extracted, u_extracted, v_extracted = extracted
            
            # Проверяем наличие ожидаемых значений YUV
            if "y_value" in expected and "u_value" in expected and "v_value" in expected:
                y_expected = expected["y_value"]
                u_expected = expected["u_value"]
                v_expected = expected["v_value"]
                
                # Вычисляем разницу
                y_diff = abs(y_extracted - y_expected)
                u_diff = abs(u_extracted - u_expected)
                v_diff = abs(v_extracted - v_expected)
                
                # Проверяем, находится ли разница в пределах допустимого отклонения
                is_valid = (y_diff <= deviation and u_diff <= deviation and v_diff <= deviation)
                
                # Собираем результаты сравнения
                result = {
                    "patch_idx": i,
                    "y_expected": y_expected,
                    "u_expected": u_expected,
                    "v_expected": v_expected,
                    "y_extracted": y_extracted,
                    "u_extracted": u_extracted,
                    "v_extracted": v_extracted,
                    "y_diff": y_diff,
                    "u_diff": u_diff,
                    "v_diff": v_diff,
                    "is_valid": is_valid
                }
                
                comparison_results.append(result)
                
                if not is_valid:
                    errors_count += 1
                    if errors_count <= 5:  # Ограничиваем вывод ошибок
                        print(f"Ошибка патча {i}: Y={y_diff:.1f}, U={u_diff:.1f}, V={v_diff:.1f} > {deviation}")
                    all_valid = False
            else:
                # Если ожидаемых значений нет, считаем патч недействительным
                comparison_results.append({
                    "patch_idx": i,
                    "is_valid": False,
                    "error": "Отсутствуют ожидаемые значения YUV"
                })
                all_valid = False
        
        if errors_count > 5:
            print(f"... и еще {errors_count - 5} ошибок патчей")
            
        return all_valid, comparison_results
    
    def create_comparison_visualization(
        self, 
        frame: Dict[str, np.ndarray], 
        comparison_results: List[Dict[str, Any]], 
        patches_metadata: List[Dict[str, Any]], 
        output_path: Path
    ) -> None:
        """
        Создает визуализацию сравнения патчей.
        
        Args:
            frame: Буфер кадра
            comparison_results: Результаты сравнения
            patches_metadata: Метаданные о патчах
            output_path: Путь для сохранения визуализации
        """
        # Преобразуем YUV в RGB для визуализации
        h, w = frame['Y'].shape
        u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
        v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Создаем RGB изображение
        rgb = yuv_to_rgb_bt709(frame['Y'], u_resized, v_resized)
        
        # Создаем копию для визуализации
        viz_img = rgb.copy()
        
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
            if "y_diff" in result and "u_diff" in result and "v_diff" in result:
                if (error_count % 10 == 0):  # Отображаем текст только для каждого 10-го патча
                    diff_text = f"Y:{result['y_diff']:.1f} U:{result['u_diff']:.1f} V:{result['v_diff']:.1f}"
                    cv2.putText(viz_img, diff_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Сохраняем визуализацию
        cv2.imwrite(str(output_path), viz_img)
    
    """
    Модификация метода visual_validate для добавления сдвига индексов на 1
    """

    def visual_validate(
        self, 
        video_processor,
        y4m_path: Path, 
        pattern_generator, 
        frames_per_pattern: int, 
        intro_frames_count: int = 0, 
        deviation: int = 4,
        max_miss_percent: float = 0.002
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Выполняет визуальную валидацию видео с паттернами.
        
        Args:
            video_processor: Объект VideoProcessor для чтения кадров
            y4m_path: Путь к Y4M файлу
            pattern_generator: Генератор паттернов
            frames_per_pattern: Количество кадров на один паттерн
            intro_frames_count: Количество вводных кадров для пропуска
            deviation: Допустимое отклонение значений
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
            visual_debug_dir = self.debug_dir / "visual_validation"
            visual_debug_dir.mkdir(exist_ok=True)
        
        with open(y4m_path, 'rb') as f:
            # Читаем заголовок Y4M
            header = f.readline().decode('ascii')
            
            # Извлекаем размеры
            import re
            width_match = re.search(r'W(\d+)', header)
            height_match = re.search(r'H(\d+)', header)
            
            if not width_match or not height_match:
                raise ValueError("Не удалось извлечь размеры из заголовка Y4M")
            
            width = int(width_match.group(1))
            height = int(height_match.group(1))
            
            # Проверяем настройки маркеров для отладки
            if self.debug_mode:
                technical_row_y = pattern_generator.tech_row * (pattern_generator.patch_size + pattern_generator.patch_gap)
                print(f"Technical row should start at Y: {technical_row_y}")
                print(f"Total frame height: {height}")
                print(f"Marker indices count: {len(pattern_generator.marker_indices)}")
                
                # Верификация маркерных координат
                if pattern_generator.marker_indices:
                    first_marker = pattern_generator.patch_coords[pattern_generator.marker_indices[0]]
                    print(f"First marker patch coordinates: {first_marker.y_range}")
                    if first_marker.y_range[0] != technical_row_y:
                        print(f"WARNING: Marker position mismatch! Expected Y={technical_row_y}, got Y={first_marker.y_range[0]}")
                        
            # Пропускаем кадры вводной последовательности
            for _ in range(intro_frames_count):
                _ = video_processor.read_y4m_frame(f, width, height)
            
            # ИЗМЕНЕНИЕ: Используем диапазон с 1 до patterns_count+1 вместо 0 до patterns_count
            print(f"Validating patterns with indices from 1 to {pattern_generator.patterns_count}")
            with tqdm(total=pattern_generator.patterns_count, desc="Визуальная валидация") as pbar:
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
                    
                    # Получаем информацию о патчах
                    patches_metadata = pattern_metadata.get("patches", [])
                    
                    # Обрабатываем только первый кадр из каждого паттерна для валидации
                    for frame_idx in range(frames_per_pattern):
                        frame = video_processor.read_y4m_frame(f, width, height)
                        validation_stats["total_frames"] += 1
                        
                        if frame is None:
                            print(f"Ошибка чтения кадра (паттерн {pattern_idx}, кадр {frame_idx})")
                            validation_stats["invalid_frames"] += 1
                            continue
                        
                        # Для первого кадра каждого паттерна выполняем проверку
                        if frame_idx == 0:
                            # Считываем маркер паттерна
                            detected_pattern_idx, marker_diagnostics = self.read_pattern_marker(frame, pattern_generator)
                            
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
                            
                            # Извлекаем значения патчей
                            extracted_values = self.extract_patch_values(frame, patches_metadata)
                            
                            # Сравниваем с ожидаемыми значениями
                            is_valid, comparison_results = self.compare_patch_values(
                                extracted_values, patches_metadata, deviation)
                            
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
                                    viz_path = visual_debug_dir / f"pattern_{pattern_idx}_errors.png"
                                    self.create_comparison_visualization(
                                        frame, comparison_results, patches_metadata, viz_path)
                        else:
                            # Остальные кадры в паттерне - считаем валидными, если первый валидный
                            if validation_stats["detected_patterns"] and validation_stats["detected_patterns"][-1]["pattern_idx"] == pattern_idx:
                                validation_stats["valid_frames"] += 1
                            else:
                                validation_stats["invalid_frames"] += 1
                    
                    pbar.update(1)
            
            # Вычисляем общий результат
            total_expected_frames = pattern_generator.patterns_count * frames_per_pattern
            validation_success = (validation_stats["valid_frames"] / total_expected_frames) >= (1 - max_miss_percent)
            
            print(f"Визуальная валидация завершена: "
                    f"проверено {validation_stats['total_frames']} кадров, "
                    f"валидных {validation_stats['valid_frames']}, "
                    f"недействительных {validation_stats['invalid_frames']}")
            
            # Сохраняем результаты валидации
            if self.debug_mode and self.debug_dir:
                validation_results_path = self.debug_dir / "visual_validation_results.json"
                import json
                with open(validation_results_path, 'w') as f:
                    json.dump(validation_stats, f, indent=2)
            
            return validation_success, validation_stats