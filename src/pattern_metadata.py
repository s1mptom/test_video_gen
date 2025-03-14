"""Модуль для работы с метаданными цветовых паттернов."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

class PatternMetadataHandler:
    """Класс для сохранения и загрузки метаданных паттернов."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Инициализирует обработчик метаданных.
        
        Args:
            output_dir: Директория для сохранения файлов метаданных
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "pattern_metadata.json"
    
    def save_pattern_metadata(
        self, 
        pattern_idx: int, 
        colors: List[Tuple[int, int, int]], 
        patch_coords: List[Any],
        y_values: Optional[np.ndarray] = None,
        u_values: Optional[np.ndarray] = None,
        v_values: Optional[np.ndarray] = None
    ) -> None:
        """
        Сохраняет метаданные паттерна.
        
        Args:
            pattern_idx: Индекс паттерна
            colors: Список RGB цветов используемых в паттерне
            patch_coords: Координаты патчей
            y_values: Значения Y компонент (опционально)
            u_values: Значения U компонент (опционально)
            v_values: Значения V компонент (опционально)
        """
        # Преобразуем numpy-значения в Python-нативные типы
        colors_python = []
        for color in colors:
            # Преобразуем каждый компонент RGB в обычный Python int
            colors_python.append([int(c) if isinstance(c, (np.integer, np.floating)) else c for c in color])
        
        # Создаем структуру данных для сохранения
        metadata = {
            "pattern_idx": int(pattern_idx),
            "colors": colors_python,
            "patches": []
        }
        
        # Конвертируем numpy массивы в обычные Python списки, если они предоставлены
        y_values_list = None
        u_values_list = None
        v_values_list = None
        
        if y_values is not None:
            y_values_list = [int(val) for val in y_values]
        if u_values is not None:
            u_values_list = [int(val) for val in u_values]
        if v_values is not None:
            v_values_list = [int(val) for val in v_values]
        
        # Добавляем данные о патчах (исключая технические)
        color_idx = 0
        for i, coords in enumerate(patch_coords):
            if not coords.is_tech:
                patch_data = {
                    "index": i,
                    "y_range": [int(val) for val in coords.y_range],
                    "x_range": [int(val) for val in coords.x_range],
                    "y_uv_range": [int(val) for val in coords.y_uv_range],
                    "x_uv_range": [int(val) for val in coords.x_uv_range]
                }
                
                # Если предоставлены значения YUV, сохраняем их тоже
                if y_values_list and u_values_list and v_values_list:
                    if color_idx < len(y_values_list):
                        patch_data["y_value"] = y_values_list[color_idx]
                        patch_data["u_value"] = u_values_list[color_idx]
                        patch_data["v_value"] = v_values_list[color_idx]
                
                metadata["patches"].append(patch_data)
                color_idx += 1
        
        # Проверяем, существует ли уже файл метаданных
        all_metadata = []
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            except json.JSONDecodeError:
                # Если файл поврежден, начинаем с пустого списка
                all_metadata = []
        
        # Добавляем метаданные текущего паттерна
        # Сначала проверяем, есть ли уже паттерн с таким индексом
        for i, pattern in enumerate(all_metadata):
            if pattern["pattern_idx"] == pattern_idx:
                all_metadata[i] = metadata
                break
        else:
            # Если паттерна с таким индексом нет, добавляем новый
            all_metadata.append(metadata)
        
        # Создаем класс для сериализации numpy типов
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return int(obj) if isinstance(obj, np.integer) else float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Сохраняем все метаданные с использованием специального энкодера
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2, cls=NumpyEncoder)
    
    def load_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Загружает все метаданные паттернов.
        
        Returns:
            List[Dict[str, Any]]: Список метаданных всех паттернов
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_pattern_metadata(self, pattern_idx: int) -> Optional[Dict[str, Any]]:
        """
        Загружает метаданные конкретного паттерна.
        
        Args:
            pattern_idx: Индекс паттерна
            
        Returns:
            Optional[Dict[str, Any]]: Метаданные паттерна или None, если не найдены
        """
        all_metadata = self.load_all_metadata()
        for pattern in all_metadata:
            if pattern["pattern_idx"] == pattern_idx:
                return pattern
        return None