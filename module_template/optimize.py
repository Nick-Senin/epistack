"""
TODO: Настройте оптимизацию модуля
"""
import dspy
from .module import ModuleName
from .metrics import ModuleMetric


def optimize(dataset=None, max_metric_calls=50, optimizer_type='gepa'):
    """
    Оптимизация модуля с использованием DSPy оптимизаторов
    
    Args:
        dataset: TODO: Датасет для оптимизации (list of dspy.Example)
        max_metric_calls: Максимальное количество вызовов метрики
        optimizer_type: Тип оптимизатора ('gepa', 'mipro', 'bootstrap')
        
    Returns:
        Оптимизированный модуль ModuleName
    """
    # TODO: Загрузите или создайте датасет
    if dataset is None:
        # Пример загрузки датасета:
        # from epistack_data import load_module_dataset
        # dataset = load_module_dataset()
        raise ValueError("TODO: Предоставьте датасет для оптимизации")
    
    # TODO: Разделите на train/val если нужно
    split_idx = int(len(dataset) * 0.8)
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # TODO: Создайте метрику
    metric = ModuleMetric()
    
    # TODO: Выберите и настройте оптимизатор
    if optimizer_type == 'gepa':
        # GEPA - эволюционная оптимизация промптов с рефлексией
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
    
    # TODO: Создайте и скомпилируйте модуль
    module = ModuleName()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print(f"✅ Модуль оптимизирован с {optimizer_type.upper()}")
    return optimized



