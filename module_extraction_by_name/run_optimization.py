"""
Скрипт для запуска оптимизации модуля извлечения с GEPA
"""
import os
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module_extraction_by_name.optimize import optimize

def main():
    print("🚀 Запуск GEPA оптимизации модуля извлечения...")
    print("=" * 60)

    # Запуск оптимизации
    # Используем dataset_path=None, так как дефолтный путь уже настроен в optimize.py
    optimized_module = optimize(
        hf_username=None,
        max_metric_calls=30, # Ограничим для скорости, можно увеличить
    )

    print("\n" + "=" * 60)
    print("✅ Оптимизация завершена!")

    # Сохранение
    save_path = Path(__file__).parent / "optimized_module.json"
    try:
        optimized_module.save(str(save_path))
        print(f"\n💾 Модуль сохранён: {save_path}")
    except Exception as e:
        print(f"\n⚠️  Не удалось сохранить: {e}")

if __name__ == "__main__":
    main()

