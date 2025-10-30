"""
Демонстрационный тест модуля извлечения библиографической информации
"""
import os
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_bibliography_extraction.config import configure_module_llm
from module_bibliography_extraction.module import BibliographyExtraction


def print_result(title, text, result):
    """Красивый вывод результата"""
    print(f"\n{'=' * 70}")
    print(f"📚 {title}")
    print(f"{'=' * 70}")
    print(f"\n📥 Входной текст:")
    print(f"   {text}")
    print(f"\n📤 Извлеченная информация:\n")
    print(f"   {result.title}")
    print(f"   {result.author}")
    print(f"   {result.publisher}")
    print(f"   {result.year}")
    print(f"   {result.place}")
    print(f"\n💡 Дополнительная информация (предположения LLM):")
    print(f"   {result.inferred_info}")


def main():
    """
    Демо-тест модуля
    """
    print("🔧 Настройка LLM...")
    try:
        configure_module_llm()
        print("✅ LLM настроен успешно")
    except Exception as e:
        print(f"⚠️  Ошибка настройки LLM: {e}")
        print("Продолжаем с дефолтными настройками...")
    
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ МОДУЛЯ ИЗВЛЕЧЕНИЯ БИБЛИОГРАФИЧЕСКОЙ ИНФОРМАЦИИ")
    print("=" * 70)
    
    # Создаем экземпляр модуля
    module = BibliographyExtraction()
    
    # Тест 1: Полная библиографическая информация
    test1_text = """
    Война и мир - роман-эпопея Льва Николаевича Толстого.
    Издательство: Художественная литература
    Год издания: 1869
    Место издания: Москва
    """
    
    try:
        result1 = module(text=test1_text)
        print_result("Тест 1: Полная информация", test1_text.strip(), result1)
    except Exception as e:
        print(f"\n❌ Ошибка в тесте 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 2: Неполная информация
    test2_text = """
    Мастер и Маргарита. Автор - Михаил Булгаков.
    Издано в 1967 году.
    """
    
    try:
        result2 = module(text=test2_text)
        print_result("Тест 2: Неполная информация", test2_text.strip(), result2)
    except Exception as e:
        print(f"\n❌ Ошибка в тесте 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 3: Краткая библиографическая запись
    test3_text = """
    Пушкин А.С. Евгений Онегин. СПб.: Просвещение, 1833.
    """
    
    try:
        result3 = module(text=test3_text)
        print_result("Тест 3: Краткая запись", test3_text.strip(), result3)
    except Exception as e:
        print(f"\n❌ Ошибка в тесте 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 4: Минимальная информация
    test4_text = """
    Какой-то текст без явной библиографической информации.
    Возможно, здесь упоминается книга, но без деталей.
    """
    
    try:
        result4 = module(text=test4_text)
        print_result("Тест 4: Минимальная информация", test4_text.strip(), result4)
    except Exception as e:
        print(f"\n❌ Ошибка в тесте 4: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 5: Современная книга с полными данными
    test5_text = """
    "Преступление и наказание" - роман Федора Михайловича Достоевского,
    впервые опубликованный в 1866 году в журнале "Русский вестник".
    Издательство: АСТ
    Место издания: Москва
    Год: 2020 (переиздание)
    """
    
    try:
        result5 = module(text=test5_text)
        print_result("Тест 5: Современное переиздание", test5_text.strip(), result5)
    except Exception as e:
        print(f"\n❌ Ошибка в тесте 5: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Все тесты завершены")
    print("=" * 70)
    print("\n💡 Подсказка: Для оптимизации модуля используйте функцию optimize():")
    print("   from module_bibliography_extraction import optimize")
    print("   optimized = optimize(dataset, max_metric_calls=50)")


if __name__ == "__main__":
    main()

