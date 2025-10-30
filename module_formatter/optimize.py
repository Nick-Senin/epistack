"""
Оптимизация модуля форматирования текста
"""
import dspy
from .module import TextFormatter
from .metrics import FormatterMetric


def optimize(dataset=None, max_metric_calls=50, optimizer_type='gepa'):
    """
    Оптимизация модуля форматирования с использованием DSPy оптимизаторов
    
    Args:
        dataset: Датасет для оптимизации (list of dspy.Example)
                 Каждый пример должен содержать:
                 - text: исходный неотформатированный текст
                 - expected_formatted_text (опционально): ожидаемый результат
        max_metric_calls: Максимальное количество вызовов метрики
        optimizer_type: Тип оптимизатора ('gepa', 'mipro', 'bootstrap')
        
    Returns:
        Оптимизированный модуль TextFormatter
    """
    # Загрузка датасета
    if dataset is None:
        # Для оптимизации необходим датасет с примерами текстов для форматирования
        # Пример создания датасета:
        # from epistack_data import load_module_dataset
        # dataset = load_module_dataset()
        # 
        # Или создайте вручную:
        # dataset = [
        #     dspy.Example(text="неотформатированный текст 1").with_inputs('text'),
        #     dspy.Example(text="неотформатированный текст 2").with_inputs('text'),
        # ]
        raise ValueError(
            "Необходим датасет для оптимизации. "
            "Предоставьте список dspy.Example с полем 'text'"
        )
    
    # Разделение на train/val
    split_idx = int(len(dataset) * 0.8)
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # Создание метрики
    metric = FormatterMetric()
    
    # Выбор и настройка оптимизатора
    if optimizer_type == 'gepa':
        # GEPA - эволюционная оптимизация промптов с рефлексией
        # Рекомендуется для форматирования, т.к. учится на feedback
        reflection_lm = dspy.settings.lm
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max_metric_calls,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
            candidate_selection_strategy='pareto',
            skip_perfect_score=True,
            track_stats=True,
            seed=42
        )
    elif optimizer_type == 'mipro':
        # MIPRO - оптимизация промптов и примеров
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=10,
            init_temperature=1.0
        )
    elif optimizer_type == 'bootstrap':
        # Bootstrap - генерация примеров для few-shot
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
    
    # Создание и компиляция модуля
    module = TextFormatter()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print(f"✅ Модуль форматирования оптимизирован с {optimizer_type.upper()}")
    return optimized
