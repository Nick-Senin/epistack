"""
Демо-скрипт для использования модуля NaiveStateTripleAbstraction
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.llm import configure_llm
from module_abstraction.module import NaiveStateTripleAbstraction
from data_models.state_triple import StateTriple

OPTIMIZED_MODULE_PATH = Path(__file__).with_name("optimized_module.json")


def main():
    # Настройка LLM
    print("🔧 Настройка LLM...")
    configure_llm()
    
    # Создание экземпляра модуля
    print("\n📦 Создание модуля NaiveStateTripleAbstraction...")
    abstractor = NaiveStateTripleAbstraction()

    if OPTIMIZED_MODULE_PATH.exists():
        print(f"\n🧠 Найден оптимизированный модуль: {OPTIMIZED_MODULE_PATH}")
        try:
            abstractor.load(str(OPTIMIZED_MODULE_PATH))
            print("✅ Загружены оптимизированные настройки StateTripleAbstraction")
        except Exception as exc:
            print(f"⚠️ Не удалось загрузить оптимизированный модуль: {exc}")
            print("   ➤ Используется базовая версия NaiveStateTripleAbstraction")
    else:
        print(f"\n⚠️ Оптимизированный модуль не найден ({OPTIMIZED_MODULE_PATH})")
        print("   ➤ Используется базовая версия NaiveStateTripleAbstraction")
    
    # Пример тройки
    example_triple: StateTriple = {
        "initial_state": "Операторы Enigma нарушают правила, используя упрощённые настройки (например, повторяют начальные положения колёсок или делают предсказуемые установки из-за усталости — Herivel tip) или допускают двойные ошибки в последовательных сообщениях (cillies)",
        "transformation": "Аналитики из Хат-6 используют эти ошибки операторов для сужения пространства возможных настроек Enigma, особенно по первым сообщениям дня, и выявляют вероятные Ringstellungen и порядок колёс без использования бомб",
        "final_state": "Ключи Enigma взламываются вручную, без применения бомб, что позволяет обеспечить непрерывность расшифровки даже до появления достаточного количества бомб"
    }

    print("\n" + "=" * 80)
    print("📝 Исходная тройка:")
    print("=" * 80)
    print(json.dumps(example_triple, indent=2, ensure_ascii=False))
    print("\n" + "=" * 80)
    
    # Обработка текста модулем
    print("\n🚀 Обработка тройки модулем NaiveStateTripleAbstraction...")
    print("⏳ Ожидайте, это может занять некоторое время...\n")
    
    result = abstractor(state_triple=example_triple)
    
    # Вывод результата
    print("=" * 80)
    print("✨ Результат (Абстракция):")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 80)
    
    print("\n✅ Демо завершено!")


if __name__ == "__main__":
    main()

