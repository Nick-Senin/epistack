"""
Быстрый пример использования модуля TransformationMarker
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_transformation_marker import (
    TransformationMarker,
    configure_module_llm,
    create_example_dataset,
    optimize
)


def quick_test():
    """Быстрый тест базовой функциональности"""
    print("🔧 Настройка LLM...")
    configure_module_llm()
    
    print("✅ Создание модуля TransformationMarker...")
    module = TransformationMarker()
    
    print("\n📝 Тестовый текст:")
    text = "Данные проходят нормализацию и фильтрацию перед агрегацией."
    transformations = ["нормализация", "фильтрация", "агрегация"]
    
    print(f"   Текст: {text}")
    print(f"   Преобразования: {', '.join(transformations)}")
    
    print("\n🚀 Запуск модуля...")
    result = module(text=text, transformations=transformations)
    
    print("\n📤 Результат:")
    print(f"   {result.marked_text}")
    
    print("\n✅ Модуль работает корректно!")


def show_example_dataset():
    """Показать пример датасета для оптимизации"""
    print("\n📊 Пример датасета для оптимизации:")
    dataset = create_example_dataset()
    
    for i, example in enumerate(dataset, 1):
        print(f"\n   Пример {i}:")
        print(f"   Текст: {example.text[:80]}...")
        print(f"   Преобразования: {example.transformations}")


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Быстрый тест TransformationMarker")
    print("=" * 80)
    
    try:
        quick_test()
        show_example_dataset()
        
        print("\n" + "=" * 80)
        print("✅ Всё работает! Модуль готов к использованию.")
        print("=" * 80)
        
        print("\n💡 Следующие шаги:")
        print("   1. Запустите demo_test.py для полного демо:")
        print("      python module_transformation_marker/demo_test.py")
        print("\n   2. Для оптимизации создайте датасет и вызовите:")
        print("      from module_transformation_marker import optimize")
        print("      optimized = optimize(dataset, max_metric_calls=50)")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

