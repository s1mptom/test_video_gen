#!/usr/bin/env python3
"""
Интерактивный визард для настройки и запуска генератора цветовых паттернов.
"""

import os
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple

# Импортируем необходимые модули проекта
from main import generate_and_validate
# Импортируем константы для правильного преобразования форматов
from src.utils.constants import ChromaFormat, ColorRange


class PatternWizard:
    """Интерактивный визард для настройки генератора паттернов."""
    
    def __init__(self, save_config: bool = True):
        """
        Инициализирует визард.
        
        Args:
            save_config: Сохранять ли конфигурацию в файл
        """
        self.config = {
            "width": 1920,
            "height": 1080,
            "patch_size": 16,
            "patch_gap": 4,
            "color_range_percent": 10.0,
            "bit_depth": 8,
            "fps": 30,
            "frames_per_pattern": 5,
            "chroma_format": "422",
            "color_range": "limited",
            "output_dir": "output",
            "output_name": "output.mp4",
            "add_intro": True,
            "debug": False,
            "deviation": 1,
            "max_miss_percent": 0.002,
            "skip_visual_validation": False
        }
        self.save_config = save_config
        self.config_file = Path("pattern_config.json")
        
        # Проверяем, есть ли сохраненная конфигурация
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    print(f"Загружена сохраненная конфигурация из {self.config_file}")
            except (json.JSONDecodeError, IOError):
                print("Ошибка при загрузке сохраненной конфигурации, используются значения по умолчанию")
    
    def run(self) -> Dict[str, Any]:
        """
        Запускает интерактивный визард.
        
        Returns:
            Dict[str, Any]: Итоговая конфигурация
        """
        print("\n===== Визард генератора цветовых паттернов =====\n")
        
        choice = self._get_choice(
            "Выберите режим настройки:",
            [
                "Экспресс (использовать текущие настройки)",
                "Полная настройка",
                "Настроить только основные параметры",
                "Загрузить конфигурацию из файла"
            ]
        )
        
        if choice == 0:  # Экспресс
            print("\nИспользуем текущие настройки:")
            self._print_current_config()
        elif choice == 1:  # Полная настройка
            self._configure_all()
        elif choice == 2:  # Основные параметры
            self._configure_basic()
        elif choice == 3:  # Загрузка из файла
            config_path = self._get_input("Введите путь к файлу конфигурации", self.config_file)
            self._load_config(config_path)
        
        # Запрашиваем подтверждение перед запуском
        print("\nИтоговая конфигурация:")
        self._print_current_config()
        
        if self._get_yes_no("Запустить генератор с этими параметрами?"):
            # Сохраняем конфигурацию если нужно
            if self.save_config:
                self._save_config()
            
            # Возвращаем конфигурацию для запуска
            return self.config
        else:
            print("Генерация отменена.")
            sys.exit(0)
    
    def _configure_all(self) -> None:
        """Настройка всех параметров."""
        print("\n--- Настройка разрешения ---")
        self.config["width"] = self._get_int("Ширина видео (пикс)", self.config["width"], 640, 7680)
        self.config["height"] = self._get_int("Высота видео (пикс)", self.config["height"], 480, 4320)
        
        print("\n--- Настройка патчей ---")
        self.config["patch_size"] = self._get_int("Размер патча (пикс)", self.config["patch_size"], 4, 64)
        self.config["patch_gap"] = self._get_int("Промежуток между патчами (пикс)", self.config["patch_gap"], 0, 32)
        
        print("\n--- Настройка цветов ---")
        self.config["color_range_percent"] = self._get_float(
            "Процент цветового диапазона [0-100]", self.config["color_range_percent"], 1.0, 100.0
        )
        self.config["bit_depth"] = self._get_choice(
            "Глубина цвета:",
            ["8 бит", "10 бит"],
            default_idx=0 if self.config["bit_depth"] == 8 else 1
        )
        self.config["bit_depth"] = 8 if self.config["bit_depth"] == 0 else 10
        
        print("\n--- Настройка формата ---")
        format_choice = self._get_choice(
            "Формат цветовой субдискретизации:",
            ["4:2:0", "4:2:2", "4:4:4"],
            default_idx={"420": 0, "422": 1, "444": 2}.get(self.config["chroma_format"], 1)
        )
        self.config["chroma_format"] = ["420", "422", "444"][format_choice]
        
        range_choice = self._get_choice(
            "Цветовой диапазон:",
            ["Ограниченный (limited)", "Полный (full)"],
            default_idx=0 if self.config["color_range"] == "limited" else 1
        )
        self.config["color_range"] = "limited" if range_choice == 0 else "full"
        
        print("\n--- Настройка видео ---")
        self.config["fps"] = self._get_int("Частота кадров", self.config["fps"], 1, 120)
        self.config["frames_per_pattern"] = self._get_int(
            "Количество кадров на паттерн", self.config["frames_per_pattern"], 1, 60
        )
        self.config["add_intro"] = self._get_yes_no(
            "Добавить вводную последовательность?",
            self.config["add_intro"]
        )
        
        print("\n--- Настройка вывода ---")
        self.config["output_dir"] = self._get_input(
            "Директория для выходных файлов", self.config["output_dir"]
        )
        self.config["output_name"] = self._get_input(
            "Имя выходного файла", self.config["output_name"]
        )
        
        print("\n--- Настройка валидации ---")
        validation_choice = self._get_choice(
            "Запускать валидацию после генерации?",
            ["Нет", "Только числовую", "Только визуальную", "Обе"],
            default_idx=3 if not self.config["skip_visual_validation"] else 1
        )
        self.config["skip_visual_validation"] = validation_choice in [0, 1]
        self.config["skip_validation"] = validation_choice == 0
        
        if validation_choice > 0:  # Если запускаем какую-то валидацию
            self.config["deviation"] = self._get_int(
                "Допустимое отклонение при валидации", self.config["deviation"], 0, 50
            )
            self.config["max_miss_percent"] = self._get_float(
                "Максимальный процент ошибок [0-100]", 
                self.config["max_miss_percent"] * 100, 0.0, 100.0
            ) / 100.0
        
        print("\n--- Дополнительные настройки ---")
        self.config["debug"] = self._get_yes_no("Режим отладки?", self.config["debug"])
    
    def _configure_basic(self) -> None:
        """Настройка только основных параметров."""
        print("\n--- Основные параметры ---")
        resolution_choice = self._get_choice(
            "Разрешение видео:",
            ["720p (1280x720)", "1080p (1920x1080)", "4K (3840x2160)", "Другое"],
            default_idx=1 if self.config["width"] == 1920 else (
                0 if self.config["width"] == 1280 else (
                    2 if self.config["width"] == 3840 else 3
                )
            )
        )
        
        if resolution_choice == 0:
            self.config["width"] = 1280
            self.config["height"] = 720
        elif resolution_choice == 1:
            self.config["width"] = 1920
            self.config["height"] = 1080
        elif resolution_choice == 2:
            self.config["width"] = 3840
            self.config["height"] = 2160
        elif resolution_choice == 3:
            self.config["width"] = self._get_int("Ширина видео (пикс)", self.config["width"], 640, 7680)
            self.config["height"] = self._get_int("Высота видео (пикс)", self.config["height"], 480, 4320)
        
        self.config["color_range_percent"] = self._get_float(
            "Процент цветового диапазона [0-100]", self.config["color_range_percent"], 1.0, 100.0
        )
        
        format_choice = self._get_choice(
            "Формат цветовой субдискретизации:",
            ["4:2:0", "4:2:2", "4:4:4"],
            default_idx={"420": 0, "422": 1, "444": 2}.get(self.config["chroma_format"], 1)
        )
        self.config["chroma_format"] = ["420", "422", "444"][format_choice]
        
        range_choice = self._get_choice(
            "Цветовой диапазон:",
            ["Ограниченный (limited)", "Полный (full)"],
            default_idx=0 if self.config["color_range"] == "limited" else 1
        )
        self.config["color_range"] = "limited" if range_choice == 0 else "full"
        
        self.config["output_name"] = self._get_input(
            "Имя выходного файла", self.config["output_name"]
        )
        
        validation_choice = self._get_choice(
            "Запускать валидацию после генерации?",
            ["Нет", "Только числовую", "Только визуальную", "Обе"],
            default_idx=3 if not self.config["skip_visual_validation"] else 1
        )
        self.config["skip_visual_validation"] = validation_choice in [0, 1]
        self.config["skip_validation"] = validation_choice == 0
    
    def _print_current_config(self) -> None:
        """Выводит текущую конфигурацию."""
        print("\nТекущая конфигурация:")
        print(f"- Разрешение: {self.config['width']}x{self.config['height']} пикселей")
        print(f"- Патчи: размер {self.config['patch_size']}x{self.config['patch_size']} пикс, промежуток {self.config['patch_gap']} пикс")
        print(f"- Цвета: {self.config['color_range_percent']}% диапазона, {self.config['bit_depth']} бит")
        print(f"- Формат: YUV {self.config['chroma_format']}, {self.config['color_range']} range")
        print(f"- Видео: {self.config['fps']} FPS, {self.config['frames_per_pattern']} кадров/паттерн")
        print(f"- Вывод: {self.config['output_dir']}/{self.config['output_name']}")
        
        validation_status = "Нет"
        if not self.config.get("skip_validation", False):
            validation_types = []
            if not self.config.get("skip_visual_validation", False):
                validation_types.append("визуальная")
            validation_types.append("числовая")
            validation_status = ", ".join(validation_types)
        
        print(f"- Валидация: {validation_status}")
        print(f"- Режим отладки: {'Да' if self.config['debug'] else 'Нет'}")
    
    def _save_config(self) -> None:
        """Сохраняет конфигурацию в файл."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Конфигурация сохранена в {self.config_file}")
        except IOError as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
    
    def _load_config(self, config_path: str) -> None:
        """
        Загружает конфигурацию из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            print(f"Конфигурация загружена из {config_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
    
    # Вспомогательные методы для интерактивного ввода
    
    def _get_input(self, prompt: str, default: Any) -> str:
        """Запрашивает строковый ввод с возможностью использования значения по умолчанию."""
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else str(default)
    
    def _get_int(self, prompt: str, default: int, min_val: int = 0, max_val: int = 1000000) -> int:
        """Запрашивает целое число с валидацией."""
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Значение должно быть в диапазоне от {min_val} до {max_val}")
            except ValueError:
                print("Пожалуйста, введите целое число")
    
    def _get_float(self, prompt: str, default: float, min_val: float = 0.0, max_val: float = 1000000.0) -> float:
        """Запрашивает число с плавающей точкой с валидацией."""
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
                value = float(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Значение должно быть в диапазоне от {min_val} до {max_val}")
            except ValueError:
                print("Пожалуйста, введите число")
    
    def _get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Запрашивает ответ Да/Нет."""
        default_str = "Д" if default else "Н"
        while True:
            user_input = input(f"{prompt} [Д/Н] [{default_str}]: ").strip().upper()
            if not user_input:
                return default
            elif user_input in ["Д", "Y", "ДА", "YES", "1"]:
                return True
            elif user_input in ["Н", "N", "НЕТ", "NO", "0"]:
                return False
            else:
                print("Пожалуйста, введите Д (Да) или Н (Нет)")
    
    def _get_choice(self, prompt: str, options: List[str], default_idx: int = 0) -> int:
        """Запрашивает выбор из списка опций."""
        print(f"\n{prompt}")
        for i, option in enumerate(options):
            print(f"{i+1}. {option}{' (по умолчанию)' if i == default_idx else ''}")
        
        while True:
            try:
                user_input = input(f"Выберите опцию [1-{len(options)}] [{default_idx+1}]: ").strip()
                if not user_input:
                    return default_idx
                
                choice = int(user_input) - 1
                if 0 <= choice < len(options):
                    return choice
                else:
                    print(f"Пожалуйста, введите число от 1 до {len(options)}")
            except ValueError:
                print("Пожалуйста, введите целое число")


def adapt_config_for_generator(config):
    """
    Адаптирует конфигурацию визарда для совместимости с генератором паттернов.
    
    Args:
        config: Исходная конфигурация
        
    Returns:
        Dict: Адаптированная конфигурация с правильными типами параметров
    """
    adapted_config = config.copy()
    
    # Преобразуем строковые значения форматов в константы
    if "chroma_format" in adapted_config:
        try:
            adapted_config["chroma_format"] = getattr(ChromaFormat, f"YUV_{adapted_config['chroma_format']}")
            print(f"Формат преобразован: {adapted_config['chroma_format']}")
        except AttributeError:
            print(f"ВНИМАНИЕ: Не удалось преобразовать формат: {adapted_config['chroma_format']}")
    
    if "color_range" in adapted_config:
        try:
            adapted_config["color_range"] = getattr(ColorRange, adapted_config['color_range'].upper())
            print(f"Диапазон преобразован: {adapted_config['color_range']}")
        except AttributeError:
            print(f"ВНИМАНИЕ: Не удалось преобразовать диапазон: {adapted_config['color_range']}")
    
    return adapted_config


def main():
    """Основная функция запуска визарда и генератора."""
    parser = argparse.ArgumentParser(description="Визард для генератора цветовых паттернов")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять конфигурацию в файл")
    args = parser.parse_args()
    
    # Запускаем визард
    wizard = PatternWizard(save_config=not args.no_save)
    config = wizard.run()
    
    # Адаптируем конфигурацию для совместимости с генератором
    adapted_config = adapt_config_for_generator(config)
    
    # Преобразуем конфигурацию в аргументы командной строки
    config_args = argparse.Namespace(**adapted_config)
    
    # Запускаем генератор с полученными параметрами
    print("\n===== Запуск генератора паттернов =====\n")
    
    try:
        is_valid, mp4_path = generate_and_validate(config_args)
        
        if is_valid:
            print(f"\n✅ Генерация и валидация успешны! Видео: {mp4_path}")
        else:
            print(f"\n❌ Ошибка валидации! Видео: {mp4_path}")
    except Exception as e:
        print(f"\n❌ Произошла ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()