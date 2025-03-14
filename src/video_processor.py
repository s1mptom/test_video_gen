"""Модуль для обработки видео (кодирование/декодирование)."""

import os
import subprocess
import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, BinaryIO

from .utils.constants import Y_BLACK, Y_WHITE, UV_NEUTRAL, ENCODER_CMD, MUXER_CMD, DECODER_CMD
from .utils.yuv_utils import create_yuv_buffer


class VideoProcessor:
    """Класс для обработки видео (кодирование/декодирование)."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Инициализирует процессор видео.
        
        Args:
            output_dir: Директория для выходных файлов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_y4m_header(self, file: BinaryIO, width: int, height: int, fps: int, colorspace: str = "420") -> None:
        """
        Записывает заголовок Y4M файла.
        
        Args:
            file: Файловый объект для записи
            width: Ширина кадра
            height: Высота кадра
            fps: Частота кадров
            colorspace: Цветовое пространство
        """
        header = f"YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C{colorspace}\n"
        file.write(header.encode('ascii'))
    
    def write_y4m_frame(self, file: BinaryIO, frame: Dict[str, np.ndarray]) -> None:
        """
        Записывает кадр в Y4M файл.
        
        Args:
            file: Файловый объект для записи
            frame: Буфер кадра с Y, U и V плоскостями
        """
        file.write(b"FRAME\n")
        file.write(frame['Y'].tobytes())
        file.write(frame['U'].tobytes())
        file.write(frame['V'].tobytes())
    
    def generate_intro_sequence(
        self, 
        width: int, 
        height: int, 
        fps: int, 
        intro_duration_seconds: int = 10
    ) -> List[Dict[str, np.ndarray]]:
        """
        Генерирует вводную последовательность (черный экран и обратный отсчет).
        
        Args:
            width: Ширина кадра
            height: Высота кадра
            fps: Частота кадров
            intro_duration_seconds: Длительность вводной последовательности в секундах
            
        Returns:
            List[Dict[str, np.ndarray]]: Список кадров вводной последовательности
        """
        frames_count = int(intro_duration_seconds * fps)
        intro_frames = []
        
        # Создаем черный кадр
        black_frame = create_yuv_buffer(height, width)
        
        # 80% времени - черный экран
        black_frames = int(frames_count * 0.8)
        for _ in range(black_frames):
            intro_frames.append(black_frame)
        
        # Остальное время - обратный отсчет
        countdown_frames = frames_count - black_frames
        seconds_per_digit = max(1, countdown_frames // 5)  # 5 секунд на отсчет
        
        for digit in range(5, 0, -1):
            # Создаем кадр с цифрой
            digit_frame = create_yuv_buffer(height, width)
            
            # Рисуем крупную цифру в центре
            self._draw_large_digit(digit_frame, digit, width, height)
            
            # Повторяем кадр с этой цифрой на протяжении нужного времени
            for _ in range(seconds_per_digit):
                intro_frames.append(digit_frame)
        
        return intro_frames
    
    def _draw_large_digit(
        self, 
        frame: Dict[str, np.ndarray], 
        digit: int, 
        width: int, 
        height: int
    ) -> None:
        """
        Рисует крупную цифру в центре кадра.
        
        Args:
            frame: Буфер кадра
            digit: Цифра для отображения (0-9)
            width: Ширина кадра
            height: Высота кадра
        """
        # Размер цифры (примерно 1/4 высоты кадра)
        digit_height = height // 4
        stroke_width = max(4, digit_height // 10)
        
        # Центр кадра
        center_x = width // 2
        center_y = height // 2
        
        # Простой рендеринг цифр на основе сегментов
        segments = {
            0: [0, 1, 2, 4, 5, 6],
            1: [2, 5],
            2: [0, 2, 3, 4, 6],
            3: [0, 2, 3, 5, 6],
            4: [1, 2, 3, 5],
            5: [0, 1, 3, 5, 6],
            6: [0, 1, 3, 4, 5, 6],
            7: [0, 2, 5],
            8: [0, 1, 2, 3, 4, 5, 6],
            9: [0, 1, 2, 3, 5, 6]
        }
        
        # Координаты сегментов относительно верхнего левого угла цифры
        digit_width = digit_height // 2
        segment_coords = {
            0: [(0, 0), (digit_width, 0)],                            # верхняя горизонталь
            1: [(0, 0), (0, digit_height // 2)],                      # верхняя левая вертикаль
            2: [(digit_width, 0), (digit_width, digit_height // 2)],  # верхняя правая вертикаль
            3: [(0, digit_height // 2), (digit_width, digit_height // 2)],  # средняя горизонталь
            4: [(0, digit_height // 2), (0, digit_height)],           # нижняя левая вертикаль
            5: [(digit_width, digit_height // 2), (digit_width, digit_height)],  # нижняя правая вертикаль
            6: [(0, digit_height), (digit_width, digit_height)]       # нижняя горизонталь
        }
        
        # Начальная позиция для отрисовки (центрировано)
        start_x = center_x - digit_width // 2
        start_y = center_y - digit_height // 2
        
        # Рисуем включенные сегменты
        color = Y_WHITE  # белый
        for segment in segments.get(digit, []):
            x1, y1 = segment_coords[segment][0]
            x2, y2 = segment_coords[segment][1]
            
            # Переносим координаты в абсолютные
            x1 += start_x
            y1 += start_y
            x2 += start_x
            y2 += start_y
            
            # Рисуем сегмент (линию)
            if x1 == x2:  # вертикальная линия
                for y in range(y1, y2 + 1):
                    for x in range(x1 - stroke_width // 2, x1 + stroke_width // 2 + 1):
                        if 0 <= y < height and 0 <= x < width:
                            frame['Y'][y, x] = color
                            # UV координаты (половинное разрешение)
                            frame['U'][y // 2, x // 2] = UV_NEUTRAL
                            frame['V'][y // 2, x // 2] = UV_NEUTRAL
            else:  # горизонтальная линия
                for x in range(x1, x2 + 1):
                    for y in range(y1 - stroke_width // 2, y1 + stroke_width // 2 + 1):
                        if 0 <= y < height and 0 <= x < width:
                            frame['Y'][y, x] = color
                            # UV координаты (половинное разрешение)
                            frame['U'][y // 2, x // 2] = UV_NEUTRAL
                            frame['V'][y // 2, x // 2] = UV_NEUTRAL
    
    def generate_y4m(
        self,
        pattern_generator,
        frames_per_pattern: int,
        fps: int,
        filename: str = "output.y4m",
        debug_mode: bool = False,
        debug_dir: Optional[Path] = None,
        add_intro: bool = True
    ) -> Tuple[Path, Dict[int, Dict[str, np.ndarray]], int]:
        """
        Генерирует Y4M файл с последовательностью цветовых паттернов.
        
        Args:
            pattern_generator: Генератор паттернов
            frames_per_pattern: Количество кадров на один паттерн
            fps: Частота кадров
            filename: Имя выходного файла
            debug_mode: Режим отладки
            debug_dir: Директория для отладочных файлов
            add_intro: Добавлять ли вводную последовательность
            
        Returns:
            Tuple[Path, Dict[int, Dict[str, np.ndarray]], int]: 
                Путь к Y4M файлу, словарь ожидаемых кадров, количество вводных кадров
        """
        from tqdm import tqdm
        
        y4m_path = self.output_dir / filename
        
        # Сохраняем все шаблоны для последующей проверки
        expected_frames = {}
        intro_frames_count = 0
        
        with open(y4m_path, 'wb') as f:
            # Записываем заголовок
            self.write_y4m_header(f, pattern_generator.width, pattern_generator.height, fps)
            
            # Добавляем вводную последовательность, если требуется
            if add_intro:
                intro_frames = self.generate_intro_sequence(
                    pattern_generator.width, pattern_generator.height, fps)
                intro_frames_count = len(intro_frames)
                for intro_frame in intro_frames:
                    self.write_y4m_frame(f, intro_frame)
            
            # Для каждого паттерна - с прогресс-баром
            with tqdm(total=pattern_generator.patterns_count, desc="Генерация паттернов") as pbar_patterns:
                for pattern_idx in range(pattern_generator.patterns_count):
                    # Создаем кадр с этим паттерном
                    frame = pattern_generator.generate_pattern_frame(pattern_idx)
                    
                    # Сохраняем ожидаемые значения для проверки
                    expected_frames[pattern_idx] = frame
                    
                    # Сохраняем отладочное изображение, если нужно
                    if debug_mode and debug_dir:
                        self.save_debug_frame(frame, f"pattern_{pattern_idx}", debug_dir)
                    
                    # Повторяем кадр нужное количество раз
                    for _ in range(frames_per_pattern):
                        self.write_y4m_frame(f, frame)
                    
                    pbar_patterns.update(1)
        
        print(f"Y4M файл создан: {y4m_path}")
        return y4m_path, expected_frames, intro_frames_count
    
    def save_debug_frame(self, frame: Dict[str, np.ndarray], name: str, debug_dir: Path) -> None:
        """
        Сохраняет кадр в PNG для отладки.
        
        Args:
            frame: Буфер кадра
            name: Имя файла
            debug_dir: Директория для отладочных файлов
        """
        try:
            h, w = frame['Y'].shape
            u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
            v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
            
            yuv = np.stack([frame['Y'], u_resized, v_resized], axis=-1).astype(np.float32)
            
            # Нормализация
            yuv[:,:,0] = (yuv[:,:,0] - 16) / 219
            yuv[:,:,1] = (yuv[:,:,1] - 128) / 112
            yuv[:,:,2] = (yuv[:,:,2] - 128) / 112
            
            # Матрица преобразования BT.709
            m = np.array([
                [1.0, 0.0, 1.5748],
                [1.0, -0.1873, -0.4681],
                [1.0, 1.8556, 0.0]
            ])
            
            # Векторизованное преобразование
            rgb = np.zeros(yuv.shape, dtype=np.float32)
            rgb[:,:,0] = np.clip(yuv[:,:,0] + m[0,2] * yuv[:,:,2], 0, 1) * 255
            rgb[:,:,1] = np.clip(yuv[:,:,0] + m[1,1] * yuv[:,:,1] + m[1,2] * yuv[:,:,2], 0, 1) * 255
            rgb[:,:,2] = np.clip(yuv[:,:,0] + m[2,1] * yuv[:,:,1], 0, 1) * 255
            
            cv2.imwrite(str(debug_dir / f"{name}.png"), rgb.astype(np.uint8))
        except Exception as e:
            print(f"Ошибка при сохранении отладочного кадра: {e}")

    def encode_video(self, y4m_path: Path, output_name: str = "output.mp4") -> Path:
        """
        Кодирует Y4M в видео файл с максимальным качеством.
        
        Args:
            y4m_path: Путь к Y4M файлу
            output_name: Имя выходного файла
            
        Returns:
            Path: Путь к закодированному видео
        """
        # Промежуточный HEVC файл
        hevc_path = self.output_dir / f"{output_name}.hevc"
        mp4_path = self.output_dir / output_name
        
        # Параметры для максимального качества
        cmd_hevc = [
            ENCODER_CMD,
            "--input", str(y4m_path), "--y4m",
            "--output", str(hevc_path),
            "--profile", "main",
            "--preset", "veryslow",
            "--lossless",  # Используем lossless режим
            "--colorprim", "1",  # BT.709
            "--transfer", "1",   # BT.709
            "--colormatrix", "1", # BT.709
            "--range", "limited"  # Ограниченный диапазон видео
        ]
        
        print("Кодирование в HEVC...")
        subprocess.run(cmd_hevc, check=True)
        
        # Мультиплексирование в MP4
        cmd_mp4 = [
            MUXER_CMD,
            "-add", str(hevc_path), 
            "-brand", "mp42",
            str(mp4_path)
        ]
        
        print("Мультиплексирование в MP4...")
        subprocess.run(cmd_mp4, check=True)
        
        # Удаляем промежуточные файлы
        if os.path.exists(hevc_path):
            os.remove(hevc_path)
        
        print(f"Видео файл создан: {mp4_path}")
        return mp4_path
    
    def decode_for_validation(self, mp4_path: Path) -> Path:
        """
        Декодирует видео обратно в Y4M для валидации.
        
        Args:
            mp4_path: Путь к MP4 файлу
            
        Returns:
            Path: Путь к декодированному Y4M файлу
        """
        validation_y4m = self.output_dir / "validation.y4m"
        
        cmd = [
            DECODER_CMD,
            "-i", str(mp4_path),
            "-pix_fmt", "yuv420p",
            "-f", "yuv4mpegpipe",
            str(validation_y4m)
        ]
        
        print("Декодирование для валидации...")
        subprocess.run(cmd, check=True)
        
        return validation_y4m
    
    def read_y4m_frame(self, file: BinaryIO, width: int, height: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Читает один кадр из Y4M файла.
        
        Args:
            file: Файловый объект для чтения
            width: Ширина кадра
            height: Высота кадра
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Буфер кадра с Y, U и V плоскостями или None
        """
        # Пропускаем заголовок кадра
        frame_header = file.readline()
        if not frame_header.startswith(b"FRAME"):
            return None
        
        # Размеры для YUV 4:2:0
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        
        # Читаем Y-плоскость
        y_data = file.read(y_size)
        if len(y_data) != y_size:
            return None
        
        # Читаем U-плоскость
        u_data = file.read(uv_size)
        if len(u_data) != uv_size:
            return None
        
        # Читаем V-плоскость
        v_data = file.read(uv_size)
        if len(v_data) != uv_size:
            return None
        
        # Преобразуем в numpy массивы
        y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape(height // 2, width // 2)
        v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape(height // 2, width // 2)
        
        return {'Y': y_plane, 'U': u_plane, 'V': v_plane}