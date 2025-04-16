#!/usr/bin/env python3
"""
Валидатор цветов в MP4 с использованием FFmpeg
Независимая проверка от OpenCV и Y4M-валидатора
"""

import json
import numpy as np
import ffmpeg
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2

class FFmpegColorValidator:
    def __init__(self, metadata_path, video_path, output_dir="validation_ffmpeg", 
                 deviation=10, max_miss_percent=0.002):
        """
        Инициализирует валидатор.
        
        Args:
            metadata_path: Путь к файлу pattern_metadata.json
            video_path: Путь к MP4 файлу для валидации
            output_dir: Директория для сохранения результатов
            deviation: Допустимое отклонение в значениях RGB
            max_miss_percent: Максимальный процент ошибок
        """
        self.metadata_path = Path(metadata_path)
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deviation = deviation
        self.max_miss_percent = max_miss_percent
        
        # Загружаем метаданные
        self.metadata = self._load_metadata()
        
        # Извлекаем конфигурацию
        self.config = self._extract_config()
        
        # Статистика валидации
        self.stats = {
            "total_patterns": 0,
            "valid_patterns": 0,
            "error_patterns": [],
            "total_patches": 0,
            "valid_patches": 0,
            "error_patches": 0
        }

        # --- Расширенная информация о видео --- 
        self.video_width = 0
        self.video_height = 0
        self.video_pix_fmt = None
        self.video_color_range = None
        self.video_bit_depth = None
        self.video_chroma_format = None
        # ----------------------------------------

        # Получаем информацию о видео один раз
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            self.video_width = int(video_info['width'])
            self.video_height = int(video_info['height'])
        except ffmpeg.Error as e:
            raise ValueError(f"Error probing video file {self.video_path}: {e.stderr.decode()}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error probing video file {self.video_path}: {e}") from e

        # Проверяем формат видео при инициализации
        if not self.validate_video_format():
             # Прерываем инициализацию, если формат не поддерживается
            raise ValueError("Unsupported video format detected during initialization.")
    
    def _load_metadata(self):
        """Загружает метаданные паттернов."""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def _extract_config(self):
        """Извлекает конфигурацию из метаданных."""
        config = None
        # 1. Ищем явную конфигурацию (pattern_idx=-1 или type=config)
        for entry in self.metadata:
            if entry.get("pattern_idx") == -1 or entry.get("type") == "config":
                if "config" in entry:
                    config = entry["config"]
                else:
                    # Берем всю запись, если нет ключа 'config'
                    config = entry 
                break
        
        # 2. Если явной конфигурации нет, ищем config или color_range в любой записи
        if config is None:
            for entry in self.metadata:
                if "config" in entry:
                    config = entry["config"]
                    break
                elif "color_range" in entry: # Ищем color_range напрямую
                    # Создаем базовый конфиг и добавляем найденный color_range
                    config = { 
                        "color_range": entry["color_range"]
                        # Можно добавить другие ключи, если они есть на верхнем уровне
                    }
                    break 
            
        # 3. Если конфигурацию так и не нашли, используем дефолтные значения
        if config is None:
            print("Warning: Configuration not found in metadata, using defaults.")
            config = {
                "frames_per_pattern": 5,
                "add_intro": True,
                # "intro_frames_count": 300, # Не будем задавать дефолт, пусть рассчитывается ниже
                "chroma_format": "422",
                "color_range": "limited" # Дефолтный color_range
            }
            
        # --- Дальнейшая обработка конфига (остается как было) --- 

        # Убедимся, что color_range установлен (если не был найден или в дефолтах)
        if "color_range" not in config:
            print("Warning: 'color_range' not found in config, defaulting to 'limited'.")
            config["color_range"] = "limited"
        
        # Если intro_frames_count не задан, но add_intro=True, используем дефолтное значение
        if config.get("add_intro", True) and "intro_frames_count" not in config:
            fps = config.get("fps", 30)
            config["intro_frames_count"] = 10 * fps  # 10 секунд
        elif not config.get("add_intro", True):
            config["intro_frames_count"] = 0
            
        print(f"Using configuration: {config}") # Добавим вывод используемой конфигурации
        return config
    
    def extract_frame(self, frame_number):
        """
        Извлекает кадр из видео с помощью ffmpeg в нативном формате видео.
        Разбирает YUV данные в зависимости от формата видео.
        
        Args:
            frame_number: Номер кадра для извлечения
            
        Returns:
            Dict[str, np.ndarray]: YUV кадр с Y, U и V компонентами
        """
        try:
            # Используем сохраненные width и height
            width = self.video_width
            height = self.video_height
            
            # Используем фактический формат пикселей и диапазон из видео
            target_pix_fmt = self.video_pix_fmt
            ffmpeg_color_range = self.video_color_range
            
            # Извлекаем кадр в его родном формате
            out, _ = (
                ffmpeg
                .input(str(self.video_path))
                .filter('select', f'eq(n,{frame_number})')
                .output('pipe:', 
                        format='rawvideo', 
                        pix_fmt=target_pix_fmt, # Используем формат видео
                        vframes=1,
                        colorspace="bt709",
                        color_range=ffmpeg_color_range)
                .run(capture_stdout=True, quiet=True)
            )
            
            # --- Динамическая обработка планарного YUV --- 
            dtype = np.uint16 if self.video_bit_depth == 10 else np.uint8
            bytes_per_sample = 2 if self.video_bit_depth == 10 else 1
            
            y_plane_size = width * height
            
            # Определяем размеры U/V плоскостей в зависимости от chroma format
            if self.video_chroma_format == '444':
                uv_width, uv_height = width, height
            elif self.video_chroma_format == '422':
                uv_width, uv_height = width // 2, height
            elif self.video_chroma_format == '420':
                uv_width, uv_height = width // 2, height // 2
            else:
                # Эта проверка уже есть в validate_video_format, но добавим на всякий случай
                raise ValueError(f"Unsupported chroma format for frame parsing: {self.video_chroma_format}")
            
            uv_plane_size = uv_width * uv_height

            # Проверяем размер буфера
            expected_bytes = (y_plane_size + 2 * uv_plane_size) * bytes_per_sample
            if len(out) != expected_bytes:
                raise ValueError(
                    f"Unexpected buffer size for {self.video_pix_fmt}. "
                    f"Expected {expected_bytes} (Y:{y_plane_size}, UV:{uv_plane_size}, BPS:{bytes_per_sample}), got {len(out)}"
                )

            # Читаем Y, U, V плоскости
            y_plane = np.frombuffer(out, dtype=dtype, count=y_plane_size, offset=0).reshape((height, width))
            u_plane = np.frombuffer(out, dtype=dtype, count=uv_plane_size, offset=y_plane_size * bytes_per_sample).reshape((uv_height, uv_width))
            v_plane = np.frombuffer(out, dtype=dtype, count=uv_plane_size, offset=(y_plane_size + uv_plane_size) * bytes_per_sample).reshape((uv_height, uv_width))

            # Растягиваем U и V до полного размера кадра (upsampling)
            # Используем ближайшего соседа (простое повторение)
            scale_x = width // uv_width
            scale_y = height // uv_height
            
            # np.repeat для растягивания
            u_full = np.repeat(np.repeat(u_plane, scale_y, axis=0), scale_x, axis=1)
            v_full = np.repeat(np.repeat(v_plane, scale_y, axis=0), scale_x, axis=1)

            # Обработка нечетной ширины (если возможно)
            if u_full.shape[1] > width:
                u_full = u_full[:, :width]
            if v_full.shape[1] > width:
                v_full = v_full[:, :width]
            
            # Проверяем финальные размеры
            if u_full.shape != (height, width) or v_full.shape != (height, width):
                 raise ValueError(f"Failed to upscale U/V planes. Final shapes: U={u_full.shape}, V={v_full.shape}")

            return {'Y': y_plane, 'U': u_full, 'V': v_full}
            
        except ffmpeg.Error as e:
            print(f"Error extracting frame {frame_number}: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"Unexpected error extracting frame: {e}")
            return None

    def yuv_to_rgb(self, yuv_frame, input_color_range, bit_depth):
        """
        Конвертирует YUV в RGB используя формулы BT.709.
        Обрабатывает Limited/Full range и 8/10 бит входные данные.
        
        Args:
            yuv_frame: YUV кадр с Y, U(Cb), V(Cr) компонентами (8 или 10 бит)
            input_color_range: Цветовой диапазон входных YUV данных ('limited'/'full')
            bit_depth: Глубина цвета входных YUV данных (8 или 10)
            
        Returns:
            np.ndarray: RGB кадр (8-бит, 0-255)
        """
        y_in = yuv_frame['Y'].astype(np.float64)
        u_in = yuv_frame['U'].astype(np.float64)
        v_in = yuv_frame['V'].astype(np.float64)
        
        height, width = y_in.shape
        rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # Масштабируем к диапазону 0.0-1.0 в зависимости от bit_depth
        max_val = (1 << bit_depth) - 1 # 255 для 8 бит, 1023 для 10 бит
        
        # Корректная нормализация YUV в зависимости от color_range и bit_depth
        if input_color_range == 'limited': 
            # Limited range (Y: 16..235 или 64..940, Cb/Cr: 16..240 или 64..960)
            y_min_scaled = 16 << (bit_depth - 8) # 16 или 64
            y_max_scaled = 235 << (bit_depth - 8) # 235 или 940
            uv_min_scaled = 16 << (bit_depth - 8)  # 16 или 64
            uv_max_scaled = 240 << (bit_depth - 8) # 240 или 960
            uv_neutral_scaled = 128 << (bit_depth - 8) # 128 или 512
            
            y_range_scaled = y_max_scaled - y_min_scaled
            uv_range_scaled = uv_max_scaled - uv_min_scaled
            
            y_norm = (y_in - y_min_scaled) / y_range_scaled
            # Нормализуем U/V к [-0.5, 0.5]
            u_norm = (u_in - uv_neutral_scaled) / uv_range_scaled 
            v_norm = (v_in - uv_neutral_scaled) / uv_range_scaled
            
        elif input_color_range == 'full': 
            # Full range (Y: 0..max_val, Cb/Cr: 0..max_val, centered at mid)
            uv_neutral_scaled = 1 << (bit_depth - 1) # 128 или 512
            
            y_norm = y_in / max_val
            # Нормализуем U/V к [-0.5, 0.5]
            u_norm = (u_in - uv_neutral_scaled) / max_val 
            v_norm = (v_in - uv_neutral_scaled) / max_val
        else:
            raise ValueError(f"Unsupported input_color_range: {input_color_range}")

        # Применяем формулы конвертации BT.709 YCbCr -> RGB (для нормализованных [0,1] Y и [-0.5, 0.5] Cb/Cr)
        # R = Y + 1.5748 * Cr 
        # G = Y - 0.1873 * Cb - 0.4681 * Cr
        # B = Y + 1.8556 * Cb
        r_float = y_norm + 1.5748 * v_norm
        g_float = y_norm - 0.1873 * u_norm - 0.4681 * v_norm
        b_float = y_norm + 1.8556 * u_norm
        
        # Масштабируем результат [0.0, 1.0] к [0, 255] и клиппим
        rgb[:, :, 0] = np.round(np.clip(r_float * 255.0, 0, 255)).astype(np.uint8)
        rgb[:, :, 1] = np.round(np.clip(g_float * 255.0, 0, 255)).astype(np.uint8)
        rgb[:, :, 2] = np.round(np.clip(b_float * 255.0, 0, 255)).astype(np.uint8)
            
        return rgb

    def validate_video_format(self):
        """
        Проверяет формат входного видео и сохраняет его параметры.
        
        Returns:
            bool: True если формат поддерживается (YUV 420/422/444, 8/10 bit)
        """
        try:
            print(f"Probing video file: {self.video_path}")
            probe = ffmpeg.probe(str(self.video_path))
            video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            if video_info is None:
                print("Error: No video stream found.")
                return False
                
            # Сохраняем базовую информацию
            self.video_width = int(video_info.get('width', 0))
            self.video_height = int(video_info.get('height', 0))
            self.video_pix_fmt = video_info.get('pix_fmt', 'unknown')
            self.video_color_range = video_info.get('color_range', 'unknown') # 'tv' (limited) or 'pc' (full)
            profile = video_info.get('profile', '')

            print(f"Detected format: pix_fmt={self.video_pix_fmt}, range={self.video_color_range}, profile={profile}")

            # Определяем Chroma Format (420, 422, 444)
            if '444' in self.video_pix_fmt:
                self.video_chroma_format = '444'
            elif '422' in self.video_pix_fmt:
                self.video_chroma_format = '422'
            elif '420' in self.video_pix_fmt:
                self.video_chroma_format = '420'
            else:
                print(f"Error: Unsupported chroma subsampling in pix_fmt: {self.video_pix_fmt}")
                return False

            # Определяем Bit Depth (8 или 10)
            # Ищем '10le', '10be', или предполагаем 8 бит, если нет указания на 10
            if '10le' in self.video_pix_fmt or '10be' in self.video_pix_fmt:
                self.video_bit_depth = 10
            elif 'p' in self.video_pix_fmt: # Форматы типа yuv420p, yuv422p, yuv444p обычно 8-битные
                self.video_bit_depth = 8
            else:
                 # Пробуем угадать по профилю, если доступен (например, Main 10 для HEVC)
                 if '10' in profile: 
                      self.video_bit_depth = 10
                 else:
                    # Если ничего не помогло, проверяем наличие 'pix_fmt' в известных 8-битных
                    known_8bit = ['yuv420p', 'yuv422p', 'yuv444p', 'nv12', 'nv16', 'nv21', 'nv42']
                    if self.video_pix_fmt in known_8bit:
                        self.video_bit_depth = 8
                    else:
                         print(f"Warning: Could not reliably determine bit depth for {self.video_pix_fmt}. Assuming 8-bit.")
                         self.video_bit_depth = 8 # По умолчанию 8 бит
            
            # Проверяем Color Range
            if self.video_color_range not in ['tv', 'pc']:
                 print(f"Warning: Unknown color range '{self.video_color_range}'. Assuming 'tv' (limited).")
                 self.video_color_range = 'tv' # По умолчанию limited

            print(f"Parsed video format: {self.video_chroma_format}, {self.video_bit_depth}-bit, {self.video_color_range} range")
            return True
            
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"Error probing video format: {stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error checking video format: {e}")
            return False

    def validate_pattern(self, pattern_data, frame_yuv):
        """
        Валидирует один паттерн, используя YUV кадр и 8-битные эталоны.
        
        Args:
            pattern_data: Данные паттерна из метаданных
            frame_yuv: YUV кадр для валидации
            
        Returns:
            Tuple[bool, List, List]: Результат валидации, список успешных и ошибочных патчей
        """
        patches = pattern_data.get("patches", [])
        colors = pattern_data.get("colors", [])  # Ожидаем 8-битные цвета [R, G, B]
        
        valid_patches = []
        error_patches = []
        
        # Для каждого патча проверяем его цвет
        for patch in patches:
            y_range = patch.get("y_range", [0, 0])
            x_range = patch.get("x_range", [0, 0])
            color_idx = patch.get("color_idx", 0)
            
            # Проверяем границы
            if color_idx >= len(colors):
                print(f"Warning: color_idx {color_idx} out of range, skipping patch")
                continue
                
            # Извлекаем ожидаемый 8-битный цвет
            expected_color_8bit = np.array(colors[color_idx], dtype=np.uint8)
            
            # Извлекаем фактический цвет из кадра
            y1, y2 = y_range
            x1, x2 = x_range
            
            # Добавляем границы для более точного измерения
            border = 2
            if y2 - y1 > 2*border and x2 - x1 > 2*border:
                y1 += border
                y2 -= border
                x1 += border
                x2 -= border
            
            try:
                # Убеждаемся, что координаты в пределах кадра
                h, w = frame_yuv['Y'].shape
                
                y1 = max(0, min(y1, h - 1))
                y2 = max(y1 + 1, min(y2, h))
                x1 = max(0, min(x1, w - 1))
                x2 = max(x1 + 1, min(x2, w))
                
                # Извлекаем YUV патч
                patch_yuv = {
                    'Y': frame_yuv['Y'][y1:y2, x1:x2],
                    'U': frame_yuv['U'][y1:y2, x1:x2],
                    'V': frame_yuv['V'][y1:y2, x1:x2]
                }
                
                # Конвертируем YUV в RGB
                # Определяем входной диапазон из конфига
                input_color_range = self.config.get("color_range", "limited") # 'limited' или 'full'
                patch_rgb = self.yuv_to_rgb(patch_yuv, input_color_range=input_color_range, bit_depth=self.video_bit_depth)
                
                # Вычисляем средний цвет
                actual_color_8bit = np.mean(patch_rgb, axis=(0, 1))
                
                # Сравниваем 8-битные значения
                diff_8bit = np.abs(actual_color_8bit.astype(np.int16) - expected_color_8bit.astype(np.int16))
                r_diff, g_diff, b_diff = diff_8bit
                
                is_valid = (r_diff <= self.deviation and 
                           g_diff <= self.deviation and 
                           b_diff <= self.deviation)
                
                # Добавляем отладочную информацию
                if not is_valid and len(error_patches) < 5:  # Показываем только первые 5 ошибок
                    # --- Добавим отладку YUV --- 
                    avg_y = np.mean(patch_yuv['Y'])
                    avg_u = np.mean(patch_yuv['U'])
                    avg_v = np.mean(patch_yuv['V'])
                    print(f"\nDebug YUV for patch {patches.index(patch)}:")
                    print(f"Avg Y: {avg_y:.2f}, Avg U: {avg_u:.2f}, Avg V: {avg_v:.2f}")
                    # ---------------------------
                    print(f"\nDebug info for patch {patches.index(patch)}:")
                    print(f"Expected color: {expected_color_8bit}")
                    print(f"Actual color: {actual_color_8bit}")
                    print(f"Diff: {diff_8bit}")
                    print(f"Deviation threshold: {self.deviation}")
                
                # Создаем отчет о патче
                patch_report = {
                    "patch_idx": patches.index(patch),
                    "color_idx": color_idx,
                    "expected_color_8bit": expected_color_8bit.tolist(),
                    "actual_color_8bit": actual_color_8bit.tolist(),
                    "diff_8bit": [int(d) for d in diff_8bit],
                    "is_valid": is_valid,
                    "coordinates": {
                        "y_range": y_range,
                        "x_range": x_range
                    }
                }
                
                if is_valid:
                    valid_patches.append(patch_report)
                else:
                    error_patches.append(patch_report)
                    
            except Exception as e:
                print(f"Error processing patch: {e}")
                continue
                
        # Общий результат валидации
        pattern_valid = len(error_patches) / max(1, len(patches)) <= self.max_miss_percent
        
        return pattern_valid, valid_patches, error_patches
    
    def create_validation_visualization(self, pattern_idx, frame_yuv, valid_patches, error_patches):
        """
        Создает визуализацию результатов валидации.
        
        Args:
            pattern_idx: Индекс паттерна
            frame_yuv: YUV кадр
            valid_patches: Список корректных патчей
            error_patches: Список ошибочных патчей
        """
        # Конвертируем YUV в RGB для визуализации
        # Определяем входной диапазон из конфига
        input_color_range = self.config.get("color_range", "limited") # 'limited' или 'full'
        frame_rgb = self.yuv_to_rgb(frame_yuv, input_color_range=input_color_range, bit_depth=self.video_bit_depth)
        
        # Создаем копию для рисования
        viz_frame = frame_rgb.copy()
        
        # Рисуем рамки и информацию для ошибочных патчей
        error_count = 0
        for patch in error_patches:
            coords = patch.get("coordinates", {})
            y_range = coords.get("y_range", [0, 0])
            x_range = coords.get("x_range", [0, 0])
            
            y1, y2 = y_range
            x1, x2 = x_range
            
            # Рисуем красную рамку
            viz_frame[y1:y2, x1:x1+2, :] = [255, 0, 0]  # Левая граница
            viz_frame[y1:y2, x2-2:x2, :] = [255, 0, 0]  # Правая граница
            viz_frame[y1:y1+2, x1:x2, :] = [255, 0, 0]  # Верхняя граница
            viz_frame[y2-2:y2, x1:x2, :] = [255, 0, 0]  # Нижняя граница
            
            # Добавляем текст для некоторых патчей
            error_count += 1
            if error_count % 10 == 0:  # Каждый 10-й патч
                expected = patch.get("expected_color_8bit", [0, 0, 0])
                actual = patch.get("actual_color_8bit", [0, 0, 0])
                diff = patch.get("diff_8bit", [0, 0, 0])
                
                # Рисуем маленький черный квадрат для текста
                text_y = max(0, y1 - 15)
                text_x = x1
                text_w = 100
                text_h = 12
                
                viz_frame[text_y:text_y+text_h, text_x:text_x+text_w, :] = [0, 0, 0]
        
        # Сохраняем визуализацию
        plt.figure(figsize=(16, 9))
        plt.imshow(viz_frame)
        
        # Добавляем информацию о валидации
        valid_count = len(valid_patches)
        error_count = len(error_patches)
        total_count = valid_count + error_count
        
        plt.title(f"Паттерн {pattern_idx}: Валидно {valid_count}/{total_count} патчей ({valid_count/total_count*100:.1f}%)")
        
        # Сохраняем изображение
        output_path = self.output_dir / f"ffmpeg_validation_pattern_{pattern_idx}.png"
        plt.savefig(output_path)
        plt.close()
        
        # Сохраняем исходный кадр
        np.save(self.output_dir / f"ffmpeg_pattern_{pattern_idx}_frame.npy", frame_rgb)
        
        return output_path
    
    def validate(self):
        """
        Выполняет валидацию всего видео.
        
        Returns:
            bool: Общий результат валидации
        """
        # Проверяем формат видео
        if not self.validate_video_format():
            print("Video format validation failed. Expected 10-bit YUV422 format.")
            return False
            
        intro_frames = self.config.get("intro_frames_count", 0)
        frames_per_pattern = self.config.get("frames_per_pattern", 5)
        
        # Фильтруем метаданные, чтобы оставить только паттерны
        pattern_entries = [entry for entry in self.metadata 
                         if "pattern_idx" in entry and entry.get("pattern_idx") >= 0 and "patches" in entry]
        
        self.stats["total_patterns"] = len(pattern_entries)
        print(f"Found {len(pattern_entries)} patterns to validate")
        
        # Сортируем по pattern_idx
        pattern_entries.sort(key=lambda x: x.get("pattern_idx", 0))
        
        results = []
        
        # Для каждого паттерна
        for pattern_data in tqdm(pattern_entries, desc="Validating patterns"):
            pattern_idx = pattern_data.get("pattern_idx", 0)
            
            # Вычисляем номер кадра
            frame_number = intro_frames + (pattern_idx - 1) * frames_per_pattern
            
            # Извлекаем кадр
            frame_yuv = self.extract_frame(frame_number)
            if frame_yuv is None:
                print(f"Error: Could not extract frame for pattern {pattern_idx}")
                continue
            
            # Валидируем паттерн
            is_valid, valid_patches, error_patches = self.validate_pattern(pattern_data, frame_yuv)
            
            # Обновляем статистику
            if is_valid:
                self.stats["valid_patterns"] += 1
            else:
                self.stats["error_patterns"].append({
                    "pattern_idx": pattern_idx,
                    "valid_patches": len(valid_patches),
                    "error_patches": len(error_patches)
                })
            
            self.stats["total_patches"] += len(valid_patches) + len(error_patches)
            self.stats["valid_patches"] += len(valid_patches)
            self.stats["error_patches"] += len(error_patches)
            
            # Создаем визуализацию
            viz_path = self.create_validation_visualization(
                pattern_idx, frame_yuv, valid_patches, error_patches)
            
            # Добавляем результат
            result = {
                "pattern_idx": pattern_idx,
                "is_valid": is_valid,
                "valid_patches": len(valid_patches),
                "error_patches": len(error_patches),
                "viz_path": str(viz_path)
            }
            
            # Добавляем некоторые ошибочные патчи для отчета
            if len(error_patches) > 0:
                # Берем только первые 5 для компактности
                result["sample_errors"] = error_patches[:5]
            
            results.append(result)
            
            # Опционально, сохраняем подробные результаты каждого паттерна
            pattern_result = {
                "pattern_idx": pattern_idx,
                "is_valid": is_valid,
                "valid_patches": valid_patches,
                "error_patches": error_patches
            }
            
            # Определяем функцию для обработки несериализуемых типов (для NumPy bool)
            def default_serializer(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                # Добавим обработку и других NumPy типов на всякий случай
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

            with open(self.output_dir / f"pattern_{pattern_idx}_result.json", 'w') as f:
                json.dump(pattern_result, f, indent=2, default=default_serializer)
        
        # Сохраняем общие результаты
        validation_success = self.stats["valid_patterns"] == self.stats["total_patterns"]
        
        summary = {
            "validation_success": validation_success,
            "stats": self.stats,
            "details": results
        }
        
        with open(self.output_dir / "ffmpeg_validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=default_serializer)
        
        # Выводим краткий отчет
        print("\nValidation Summary:")
        print(f"  Patterns: {self.stats['valid_patterns']}/{self.stats['total_patterns']} valid")
        print(f"  Patches: {self.stats['valid_patches']}/{self.stats['total_patches']} valid "
              f"({self.stats['valid_patches']/self.stats['total_patches']*100:.1f}%)")
        print(f"  Overall result: {'SUCCESS' if validation_success else 'FAILURE'}")
        print(f"  Full report saved to {self.output_dir}/ffmpeg_validation_summary.json")
        
        return validation_success

def main():
    parser = argparse.ArgumentParser(description="Validate colors in MP4 file using FFmpeg")
    parser.add_argument("--metadata", type=str, default="output/pattern_metadata.json",
                      help="Path to pattern metadata JSON file")
    parser.add_argument("--video", type=str, default="output/output.mp4",
                      help="Path to MP4 file to validate")
    parser.add_argument("--output-dir", type=str, default="output/ffmpeg_validation",
                      help="Directory for output files")
    parser.add_argument("--deviation", type=int, default=1,
                      help="Allowed deviation in RGB values")
    parser.add_argument("--max-miss-percent", type=float, default=0.002,
                      help="Maximum percentage of allowed errors")
    args = parser.parse_args()
    
    validator = FFmpegColorValidator(
        metadata_path=args.metadata,
        video_path=args.video,
        output_dir=args.output_dir,
        deviation=args.deviation,
        max_miss_percent=args.max_miss_percent
    )
    
    success = validator.validate()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())