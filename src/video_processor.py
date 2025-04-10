"""Модуль для обработки видео (кодирование/декодирование)."""

import os
import subprocess
import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, BinaryIO, Any

from .utils.constants import (
    Y_BLACK, Y_WHITE, UV_NEUTRAL, 
    ENCODER_CMD, MUXER_CMD, DECODER_CMD,
    ColorRange, ChromaFormat
)
from .utils.yuv_utils import create_yuv_buffer, get_chroma_dimensions


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
    
    def write_y4m_header(
        self, 
        file: BinaryIO, 
        width: int, 
        height: int, 
        fps: int, 
        chroma_format: str = ChromaFormat.YUV_422, 
        color_range: str = ColorRange.LIMITED
    ):
        """
        Записывает заголовок Y4M файла.
        
        Args:
            file: Файловый объект для записи
            width: Ширина кадра
            height: Высота кадра
            fps: Частота кадров
            chroma_format: Формат цветовой субдискретизации ("420", "422", "444")
            color_range: Цветовой диапазон ('limited' или 'full')
        """
        # Преобразуем формат для Y4M
        y4m_format = chroma_format
        # По спецификации Y4M для формата 4:2:2 используется "422" или "UYVY"
        if chroma_format == ChromaFormat.YUV_422:
            y4m_format = "422"
        # Для 4:2:0 используется "420" или "420jpeg", "420mpeg2", "420paldv"
        elif chroma_format == ChromaFormat.YUV_420:
            y4m_format = "420"
        # Для 4:4:4 используется "444"
        elif chroma_format == ChromaFormat.YUV_444:
            y4m_format = "444"
            
        # Добавляем информацию о цветовом диапазоне
        range_flag = "XCOLORRANGE=FULL" if color_range == ColorRange.FULL else "XCOLORRANGE=LIMITED"
        
        header = f"YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C{y4m_format} {range_flag}\n"
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
        intro_duration_seconds: int = 10,
        chroma_format: str = ChromaFormat.YUV_422,
        color_range: str = ColorRange.LIMITED
    ) -> List[Dict[str, np.ndarray]]:
        """
        Генерирует вводную последовательность (черный экран и обратный отсчет).
        
        Args:
            width: Ширина кадра
            height: Высота кадра
            fps: Частота кадров
            intro_duration_seconds: Длительность вводной последовательности в секундах
            chroma_format: Формат цветовой субдискретизации
            color_range: Цветовой диапазон
            
        Returns:
            List[Dict[str, np.ndarray]]: Список кадров вводной последовательности
        """
        frames_count = int(intro_duration_seconds * fps)
        intro_frames = []
        
        # Создаем черный кадр с учетом цветового диапазона и формата
        black_frame = create_yuv_buffer(
            height, 
            width, 
            chroma_format, 
            color_range
        )
        
        # 80% времени - черный экран
        black_frames = int(frames_count * 0.8)
        for _ in range(black_frames):
            intro_frames.append(black_frame)
        
        # Остальное время - обратный отсчет
        countdown_frames = frames_count - black_frames
        seconds_per_digit = max(1, countdown_frames // 5)  # 5 секунд на отсчет
        
        for digit in range(5, 0, -1):
            # Создаем кадр с цифрой (с учетом цветового диапазона и формата)
            digit_frame = create_yuv_buffer(
                height, 
                width, 
                chroma_format, 
                color_range
            )
            
            # Рисуем крупную цифру в центре
            self._draw_large_digit(digit_frame, digit, width, height, color_range, chroma_format)
            
            # Повторяем кадр с этой цифрой на протяжении нужного времени
            for _ in range(seconds_per_digit):
                intro_frames.append(digit_frame)
        
        return intro_frames
    
    def _draw_large_digit(
        self, 
        frame: Dict[str, np.ndarray], 
        digit: int, 
        width: int, 
        height: int,
        color_range: str = ColorRange.LIMITED,
        chroma_format: str = ChromaFormat.YUV_422  # Добавляем параметр формата для консистентности
    ) -> None:
        """
        Рисует крупную цифру в центре кадра.
        
        Args:
            frame: Буфер кадра
            digit: Цифра для отображения (0-9)
            width: Ширина кадра
            height: Высота кадра
            color_range: Цветовой диапазон
            chroma_format: Формат цветовой субдискретизации
        """
        # Получаем значения в зависимости от цветового диапазона
        from .utils.constants import get_yuv_constants
        
        yuv_const = get_yuv_constants(color_range)
        y_white = yuv_const["Y_WHITE"]
        uv_neutral = yuv_const["UV_NEUTRAL"]
        
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
        color = y_white  # белый
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
                            
                            # Вычисляем соответствующие UV координаты в зависимости от формата
                            if chroma_format == ChromaFormat.YUV_420:
                                uv_y, uv_x = y // 2, x // 2
                                if 0 <= uv_y < frame['U'].shape[0] and 0 <= uv_x < frame['U'].shape[1]:
                                    frame['U'][uv_y, uv_x] = uv_neutral
                                    frame['V'][uv_y, uv_x] = uv_neutral
                            elif chroma_format == ChromaFormat.YUV_422:
                                uv_y, uv_x = y, x // 2
                                if 0 <= uv_y < frame['U'].shape[0] and 0 <= uv_x < frame['U'].shape[1]:
                                    frame['U'][uv_y, uv_x] = uv_neutral
                                    frame['V'][uv_y, uv_x] = uv_neutral
                            elif chroma_format == ChromaFormat.YUV_444:
                                if 0 <= y < frame['U'].shape[0] and 0 <= x < frame['U'].shape[1]:
                                    frame['U'][y, x] = uv_neutral
                                    frame['V'][y, x] = uv_neutral
            else:  # горизонтальная линия
                for x in range(x1, x2 + 1):
                    for y in range(y1 - stroke_width // 2, y1 + stroke_width // 2 + 1):
                        if 0 <= y < height and 0 <= x < width:
                            frame['Y'][y, x] = color
                            
                            # Вычисляем соответствующие UV координаты в зависимости от формата
                            if chroma_format == ChromaFormat.YUV_420:
                                uv_y, uv_x = y // 2, x // 2
                                if 0 <= uv_y < frame['U'].shape[0] and 0 <= uv_x < frame['U'].shape[1]:
                                    frame['U'][uv_y, uv_x] = uv_neutral
                                    frame['V'][uv_y, uv_x] = uv_neutral
                            elif chroma_format == ChromaFormat.YUV_422:
                                uv_y, uv_x = y, x // 2
                                if 0 <= uv_y < frame['U'].shape[0] and 0 <= uv_x < frame['U'].shape[1]:
                                    frame['U'][uv_y, uv_x] = uv_neutral
                                    frame['V'][uv_y, uv_x] = uv_neutral
                            elif chroma_format == ChromaFormat.YUV_444:
                                if 0 <= y < frame['U'].shape[0] and 0 <= x < frame['U'].shape[1]:
                                    frame['U'][y, x] = uv_neutral
                                    frame['V'][y, x] = uv_neutral
                                    
    def generate_y4m(
        self,
        pattern_generator,
        frames_per_pattern: int,
        fps: int,
        filename: str = "output.y4m",
        debug_mode: bool = False,
        debug_dir: Optional[Path] = None,
        add_intro: bool = True,
        chroma_format: str = None,
        color_range: str = None
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
            chroma_format: Формат цветовой субдискретизации (если None, берется из pattern_generator)
            color_range: Цветовой диапазон (если None, берется из pattern_generator)
            
        Returns:
            Tuple[Path, Dict[int, Dict[str, np.ndarray]], int]: 
                Путь к Y4M файлу, словарь ожидаемых кадров, количество вводных кадров
        """
        from tqdm import tqdm
        
        # Если формат и диапазон не указаны, берем из генератора паттернов
        if chroma_format is None:
            chroma_format = pattern_generator.chroma_subsampling
        
        if color_range is None:
            color_range = pattern_generator.color_range
            
        y4m_path = self.output_dir / filename
        
        # Сохраняем все шаблоны для последующей проверки
        expected_frames = {}
        intro_frames_count = 0
        
        with open(y4m_path, 'wb') as f:
            # Записываем заголовок с учетом формата и диапазона
            self.write_y4m_header(
                f, 
                pattern_generator.width, 
                pattern_generator.height, 
                fps,
                chroma_format,
                color_range
            )
            
            # Добавляем вводную последовательность, если требуется
            if add_intro:
                intro_frames = self.generate_intro_sequence(
                    width=pattern_generator.width,
                    height=pattern_generator.height,
                    fps=fps,
                    intro_duration_seconds=10,
                    chroma_format=chroma_format,
                    color_range=color_range
                )
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
            
            # Определяем формат по размерам UV плоскостей
            if frame['U'].shape[0] == h // 2 and frame['U'].shape[1] == w // 2:
                format_name = "420"
            elif frame['U'].shape[0] == h and frame['U'].shape[1] == w // 2:
                format_name = "422"
            elif frame['U'].shape[0] == h and frame['U'].shape[1] == w:
                format_name = "444"
            else:
                format_name = "unknown"
            
            # Масштабируем UV до размеров Y для визуализации
            u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
            v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Определяем цветовой диапазон по Y
            if np.min(frame['Y']) == 0 and np.max(frame['Y']) > 235:
                range_name = "full"
            else:
                range_name = "limited"
            
            yuv = np.stack([frame['Y'], u_resized, v_resized], axis=-1).astype(np.float32)
            
            # Нормализация в зависимости от диапазона
            if range_name == "limited":
                # Limited range (16-235, 16-240)
                yuv[:,:,0] = (yuv[:,:,0] - 16) / 219
                yuv[:,:,1] = (yuv[:,:,1] - 128) / 112
                yuv[:,:,2] = (yuv[:,:,2] - 128) / 112
            else:
                # Full range (0-255)
                yuv[:,:,0] = yuv[:,:,0] / 255
                yuv[:,:,1] = (yuv[:,:,1] - 128) / 128
                yuv[:,:,2] = (yuv[:,:,2] - 128) / 128
            
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
            
            # Добавляем информацию о формате и диапазоне в имя файла
            output_name = f"{name}_{format_name}_{range_name}.png"
            cv2.imwrite(str(debug_dir / output_name), rgb.astype(np.uint8))
        except Exception as e:
            print(f"Ошибка при сохранении отладочного кадра: {e}")

    def encode_video(
        self, 
        y4m_path: Path, 
        output_name: str = "output.mp4",
        color_range: str = ColorRange.LIMITED,
        bit_depth: int = 8,
        chroma_format: str = ChromaFormat.YUV_422
    ) -> Path:
        """
        Кодирует Y4M в видео файл с максимальным качеством.
        
        Args:
            y4m_path: Путь к Y4M файлу
            output_name: Имя выходного файла
            color_range: Цветовой диапазон ('limited' или 'full')
            
        Returns:
            Path: Путь к закодированному видео
        """
        # Промежуточный HEVC файл
        hevc_path = self.output_dir / f"{output_name}.hevc"
        mp4_path = self.output_dir / output_name
        
        # Значение range для x265
        range_value = "full" if color_range == ColorRange.FULL else "limited"
        
        if bit_depth == 10:
            if chroma_format == ChromaFormat.YUV_420:
                profile = "main10"
            elif chroma_format == ChromaFormat.YUV_422:
                profile = "main422-10"
            elif chroma_format == ChromaFormat.YUV_444:
                profile = "main444-10"
            else:
                profile = "main10"
        else:
            if chroma_format == ChromaFormat.YUV_420:
                profile = "main"
            elif chroma_format == ChromaFormat.YUV_422:
                profile = "main422"
            elif chroma_format == ChromaFormat.YUV_444:
                profile = "main444"
            else:
                profile = "main"

        # Параметры для максимального качества
        cmd_hevc = [
            ENCODER_CMD,
            "--input", str(y4m_path), "--y4m",
            "--output", str(hevc_path),
            "--profile", profile,
            "--preset", "veryslow",
            "--crf", "0",
            "--lossless",  # Используем lossless режим
            "--no-sao",
            "--colorprim", "1",  # BT.709
            "--transfer", "1",   # BT.709
            "--colormatrix", "1", # BT.709
            "--range", range_value  # Передаем выбранный диапазон
        ]
        
        print(f"Кодирование в HEVC (диапазон: {range_value})...")
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
    
    def decode_for_validation(
        self, 
        mp4_path: Path, 
        chroma_format: str = ChromaFormat.YUV_422,
        color_range: str = ColorRange.LIMITED
    ) -> Path:
        """
        Декодирует видео обратно в Y4M для валидации.
        
        Args:
            mp4_path: Путь к MP4 файлу
            chroma_format: Формат цветовой субдискретизации
            color_range: Цветовой диапазон
            
        Returns:
            Path: Путь к декодированному Y4M файлу
        """
        validation_y4m = self.output_dir / "validation.y4m"
        
        # Определяем параметры pix_fmt для ffmpeg
        if chroma_format == ChromaFormat.YUV_420:
            pix_fmt = "yuv420p"
        elif chroma_format == ChromaFormat.YUV_422:
            pix_fmt = "yuv422p"
        elif chroma_format == ChromaFormat.YUV_444:
            pix_fmt = "yuv444p"
        else:
            raise ValueError(f"Неподдерживаемый формат субдискретизации: {chroma_format}")
        
        # Добавляем информацию о диапазоне
        range_flag = "-color_range 1" if color_range == ColorRange.LIMITED else "-color_range 2"
        
        cmd = [
            DECODER_CMD,
            "-i", str(mp4_path),
            "-pix_fmt", pix_fmt,
            "-color_range", "1" if color_range == ColorRange.LIMITED else "2",  # 1=limited, 2=full
            "-f", "yuv4mpegpipe",
            str(validation_y4m)
        ]
        
        print(f"Декодирование для валидации (формат: {chroma_format}, диапазон: {color_range})...")
        subprocess.run(cmd, check=True)
        
        return validation_y4m
    
    def read_y4m_frame(
        self, 
        file: BinaryIO, 
        width: int, 
        height: int, 
        chroma_format: str = ChromaFormat.YUV_422
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Читает один кадр из Y4M файла.
        
        Args:
            file: Файловый объект для чтения
            width: Ширина кадра
            height: Высота кадра
            chroma_format: Формат YUV (может определяться автоматически из заголовка Y4M)
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Буфер кадра с Y, U и V плоскостями или None
        """
        # Пропускаем заголовок кадра
        frame_header = file.readline()
        if not frame_header.startswith(b"FRAME"):
            return None
        
        # Размеры Y-плоскости всегда одинаковы
        y_size = width * height
        
        # Определяем размеры UV в зависимости от формата
        if chroma_format == ChromaFormat.YUV_420:
            uv_height, uv_width = height // 2, width // 2
        elif chroma_format == ChromaFormat.YUV_422:
            uv_height, uv_width = height, width // 2
        elif chroma_format == ChromaFormat.YUV_444:
            uv_height, uv_width = height, width
        else:
            # Пытаемся автоматически определить из заголовка Y4M
            # (не реализовано в этом примере, но можно добавить)
            raise ValueError(f"Неподдерживаемый формат YUV: {chroma_format}")
        
        uv_size = uv_width * uv_height
        
        # Читаем данные
        y_data = file.read(y_size)
        if len(y_data) != y_size:
            return None
        
        u_data = file.read(uv_size)
        if len(u_data) != uv_size:
            return None
        
        v_data = file.read(uv_size)
        if len(v_data) != uv_size:
            return None
        
        # Преобразуем в numpy массивы
        y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape(uv_height, uv_width)
        v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape(uv_height, uv_width)
        
        return {'Y': y_plane, 'U': u_plane, 'V': v_plane}
    
    def parse_y4m_header(self, file_path: Path) -> Dict[str, Any]:
        """
        Парсит заголовок Y4M файла для определения его параметров.
        
        Args:
            file_path: Путь к Y4M файлу
            
        Returns:
            Dict[str, Any]: Словарь с параметрами файла (width, height, fps, chroma_format, color_range)
        """
        with open(file_path, 'rb') as f:
            header = f.readline().decode('ascii')
        
        # Извлекаем параметры
        params = {}
        
        # Ширина и высота
        width_match = re.search(r'W(\d+)', header)
        height_match = re.search(r'H(\d+)', header)
        
        if width_match and height_match:
            params['width'] = int(width_match.group(1))
            params['height'] = int(height_match.group(1))
        
        # Частота кадров
        fps_match = re.search(r'F(\d+):(\d+)', header)
        if fps_match:
            numerator = int(fps_match.group(1))
            denominator = int(fps_match.group(2))
            params['fps'] = numerator / denominator
        
        # Формат цветности
        chroma_match = re.search(r'C(\w+)', header)
        if chroma_match:
            chroma = chroma_match.group(1)
            if chroma in ['420', '420jpeg', '420paldv', '420mpeg2']:
                params['chroma_format'] = ChromaFormat.YUV_420
            elif chroma in ['422', 'UYVY']:
                params['chroma_format'] = ChromaFormat.YUV_422
            elif chroma in ['444']:
                params['chroma_format'] = ChromaFormat.YUV_444
            else:
                # По умолчанию предполагаем 422
                params['chroma_format'] = ChromaFormat.YUV_422
        
        # Цветовой диапазон
        range_match = re.search(r'XCOLORRANGE=(\w+)', header)
        if range_match:
            range_value = range_match.group(1)
            params['color_range'] = ColorRange.FULL if range_value.upper() == 'FULL' else ColorRange.LIMITED
        else:
            # По умолчанию считаем limited
            params['color_range'] = ColorRange.LIMITED
        
        return params