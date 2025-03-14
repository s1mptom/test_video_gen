#!/usr/bin/env python3
"""
Генератор цветовых паттернов для калибровки и построения 3D LUT.

Пример использования:
    python main.py --width 1920 --height 1080 --patch-size 16 --patch-gap 4 --color-range 100
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

from src.pattern_generator import PatternGenerator
from src.video_processor import VideoProcessor
from src.validation_processor import ValidationProcessor
from src.visual_validation.visual_validator import VisualValidationProcessor
from src.pattern_metadata import PatternMetadataHandler


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    
    Returns:
        argparse.Namespace: Распарсенные аргументы
    """
    parser = argparse.ArgumentParser(
        description="Генератор цветовых паттернов для калибровки и построения 3D LUT."
    )
    
    # Основные параметры
    parser.add_argument("--width", type=int, default=1920, help="Ширина видео (пикс)")
    parser.add_argument("--height", type=int, default=1080, help="Высота видео (пикс)")
    parser.add_argument("--patch-size", type=int, default=16, help="Размер цветового патча (пикс)")
    parser.add_argument("--patch-gap", type=int, default=4, help="Промежуток между патчами (пикс)")
    
    # Параметры цветов и видео
    parser.add_argument("--color-range", type=float, default=100.0,
                      help="Процент цветового диапазона [0-100]")
    parser.add_argument("--bit-depth", type=int, default=8, help="Глубина цвета (бит)")
    parser.add_argument("--fps", type=int, default=30, help="Частота кадров")
    parser.add_argument("--frames-per-pattern", type=int, default=5,
                      help="Количество кадров на паттерн")
    
    # Другие параметры
    parser.add_argument("--output-dir", type=str, default="output",
                      help="Директория для выходных файлов")
    parser.add_argument("--output-name", type=str, default="output.mp4",
                      help="Имя выходного файла")
    parser.add_argument("--add-intro", action="store_true", default=True,
                      help="Добавить вводную последовательность")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    parser.add_argument("--deviation", type=int, default=1,
                      help="Допустимое отклонение при валидации")
    parser.add_argument("--max-miss-percent", type=float, default=0.002,
                      help="Максимальный процент ошибок при валидации")
    parser.add_argument("--skip-visual-validation", action="store_true", 
                      help="Пропустить визуальную валидацию")
    
    return parser.parse_args()


"""
Оптимизированная функция generate_and_validate из main.py 
с более эффективной обработкой метаданных
"""

def generate_and_validate(args: argparse.Namespace) -> Tuple[bool, Optional[Path]]:
    """
    Выполняет полный цикл генерации и валидации видео.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Tuple[bool, Optional[Path]]: Результат валидации и путь к выходному файлу
    """
    # Создаем директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    debug_dir = output_dir / "debug" if args.debug else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
    
    # Создаем компоненты системы
    pattern_generator = PatternGenerator(
        width=args.width,
        height=args.height,
        patch_size=args.patch_size,
        patch_gap=args.patch_gap,
        color_range_percent=args.color_range,
        bit_depth=args.bit_depth
    )
    
    video_processor = VideoProcessor(output_dir=args.output_dir)
    validation_processor = ValidationProcessor(debug_mode=args.debug, debug_dir=debug_dir)
    metadata_handler = PatternMetadataHandler(output_dir=args.output_dir)
    visual_validator = VisualValidationProcessor(
        output_dir=args.output_dir, debug_mode=args.debug, debug_dir=debug_dir)
    
    # Оптимизация сохранения метаданных паттернов
    print("Сохранение метаданных паттернов...")
    from tqdm import tqdm
    
    # Для больших паттернов используем пакетную обработку
    batch_size = 50  # Настройте в зависимости от вашей системы
    total_patterns = pattern_generator.patterns_count
    
    with tqdm(total=total_patterns, desc="Пакетное сохранение метаданных") as pbar:
        for pattern_idx in range(total_patterns):
            # Используем оригинальную функцию pattern_generator.save_pattern_metadata
            # Она подготовит метаданные и вызовет metadata_handler.save_pattern_metadata
            pattern_generator.save_pattern_metadata(pattern_idx, metadata_handler)
            pbar.update(1)
    
    # Записываем все метаданные на диск за один раз
    metadata_handler.flush_metadata()
    
    # Генерируем Y4M
    y4m_path, expected_frames, intro_frames_count = video_processor.generate_y4m(
        pattern_generator=pattern_generator,
        frames_per_pattern=args.frames_per_pattern,
        fps=args.fps,
        filename="temp.y4m",
        debug_mode=args.debug,
        debug_dir=debug_dir,
        add_intro=args.add_intro
    )
    
    # Кодируем видео
    mp4_path = video_processor.encode_video(y4m_path, output_name=args.output_name)
    
    # Декодируем для валидации
    validation_y4m = video_processor.decode_for_validation(mp4_path)
    
    # Валидируем
    is_valid = validation_processor.validate(
        validation_y4m=validation_y4m,
        expected_frames=expected_frames,
        pattern_count=pattern_generator.patterns_count,
        frames_per_pattern=args.frames_per_pattern,
        width=args.width,
        height=args.height,
        patches_mask=pattern_generator.patches_mask,
        deviation=args.deviation,
        max_miss_percent=args.max_miss_percent,
        intro_frames_count=intro_frames_count
    )
    
    # Запускаем визуальную валидацию (если не отключена)
    visual_validation_result = True
    if not args.skip_visual_validation:
        print("\nЗапуск визуальной валидации...")
        visual_validation_result, validation_stats = visual_validator.visual_validate(
            video_processor=video_processor,
            y4m_path=validation_y4m,
            pattern_generator=pattern_generator,
            frames_per_pattern=args.frames_per_pattern,
            intro_frames_count=intro_frames_count,
            deviation=args.deviation,
            max_miss_percent=args.max_miss_percent
        )
        
        # Сохраняем статистику визуальной валидации
        if args.debug:
            stats_path = output_dir / "visual_validation_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(validation_stats, f, indent=2)
    
    # Общий результат валидации (технической и визуальной)
    overall_valid = is_valid and visual_validation_result
    
    # Очистка
    if os.path.exists(validation_y4m) and not args.debug:
        os.remove(validation_y4m)
    
    if os.path.exists(y4m_path) and not args.debug:
        os.remove(y4m_path)
    
    return overall_valid, mp4_path

def main() -> None:
    """
    Основная функция программы.
    """
    args = parse_args()
    
    try:
        is_valid, mp4_path = generate_and_validate(args)
        
        if is_valid:
            print(f"✅ Валидация успешна! Видео: {mp4_path}")
        else:
            print(f"❌ Ошибка валидации! Видео: {mp4_path}")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()