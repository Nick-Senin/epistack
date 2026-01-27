"""
TODO: Добавьте демо-тест для проверки работы модуля
"""
import os
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_template.config import configure_module_llm
from module_template.module import ModuleName


def main():
    """
    Демо-тест модуля
    """
    print("🔧 Настройка LLM...")
    configure_module_llm()
    
    print("\n" + "=" * 60)
    print("🧪 Тестирование модуля")
    print("=" * 60)
    
    # TODO: Создайте экземпляр модуля
    module = ModuleName()
    
    # TODO: Подготовьте тестовые данные
    # Пример:
    test_input = "TODO: Добавьте тестовые входные данные"
    
    # TODO: Вызовите модуль
    print(f"\n📥 Входные данные:\n   {test_input}")
    
    try:
        result = module(input_field=test_input)
        
        # TODO: Выведите результаты
        print(f"\n📤 Результат:")
        print(f"   {result.output_data}")
        
        # TODO: Добавьте дополнительные проверки если нужно
        # if hasattr(result, 'confidence'):
        #     print(f"   Уверенность: {result.confidence}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Тест завершён")


if __name__ == "__main__":
    main()





