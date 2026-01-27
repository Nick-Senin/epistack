"""
Запуск оптимизации SemanticHalver с GEPA оптимизатором
"""
from module_semantic_parallel_splitter import (
    optimize,
    load_dataset,
    configure_module_llm,
    create_reflection_lm
)

# Настройка LLM
configure_module_llm()

# Загрузка датасета
dataset = load_dataset()
print(f"\n📊 Размер датасета: {len(dataset)} примеров")

# Создаём reflection LM
reflection_lm = create_reflection_lm()
print(f"🤖 Reflection LM: {reflection_lm.model}")

# Запуск оптимизации
print("\n" + "="*60)
print("🚀 ЗАПУСК ОПТИМИЗАЦИИ GEPA")
print("="*60)

optimized_module = optimize(
    dataset=dataset,
    optimizer_type='gepa',
    reflection_lm=reflection_lm,
    auto='light',  # light, medium, heavy
    num_threads=4
)

print("\n" + "="*60)
print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
print("="*60)

# Сохранение оптимизированного модуля
from module_semantic_parallel_splitter import save_optimized_module
save_path = "module_semantic_parallel_splitter/optimized_halver.json"
save_optimized_module(optimized_module, save_path)

print(f"\n💾 Оптимизированный модуль сохранён в: {save_path}")
