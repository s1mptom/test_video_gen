"""Модуль для работы с метаданными цветовых паттернов."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


# Определяем энкодер numpy типов один раз
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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
        
        # Буфер для метаданных, чтобы сохранить их все за один раз
        self.metadata_buffer = defaultdict(dict)
        
        # Загружаем существующие метаданные при инициализации
        self._load_existing_metadata()
    
    def _load_existing_metadata(self):
        """Загружает существующие метаданные в буфер."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
                    for pattern in all_metadata:
                        pattern_idx = pattern.get("pattern_idx")
                        if pattern_idx is not None:
                            self.metadata_buffer[pattern_idx] = pattern
            except (json.JSONDecodeError, IOError):
                # Если файл поврежден или не может быть прочитан, начинаем с пустого буфера
                pass
    
    def save_pattern_metadata(self, pattern_idx: int, metadata: Dict) -> None:
        """
        Сохраняет метаданные паттерна в буфер (без записи на диск).
        Совместимо с исходной сигнатурой.
        
        Args:
            pattern_idx: Индекс паттерна
            metadata: Словарь с метаданными паттерна
        """
        # Сохраняем в буфер
        self.metadata_buffer[pattern_idx] = metadata
    
    def flush_metadata(self) -> None:
        """Сохраняет все метаданные из буфера в файл."""
        # Преобразуем словарь в список
        all_metadata = list(self.metadata_buffer.values())
        
        # Сохраняем все метаданные за один раз
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, cls=NumpyEncoder)
    
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
        # Сначала проверяем буфер
        if pattern_idx in self.metadata_buffer:
            return self.metadata_buffer[pattern_idx]
        
        # Если нет в буфере, загружаем из файла
        all_metadata = self.load_all_metadata()
        for pattern in all_metadata:
            if pattern["pattern_idx"] == pattern_idx:
                return pattern
        return None