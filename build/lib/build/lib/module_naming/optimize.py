"""
Оптимизация модуля именования
"""
import dspy
from epistack_data import for_naming_module
from .module import RelationNamer
from .metrics import create_metric


def optimize(hf_username: str = None, max_metric_calls: int = 50):
    """
    Оптимизация модуля именования с использованием GEPA
    
    Args:
        hf_username: HuggingFace username для загрузки датасета (None = локальный)
        max_metric_calls: Максимальное количество вызовов метрики
        
    Returns:
        Оптимизированный модуль RelationNamer
    """
    # Загрузка датасета (локальный если username не указан)
    dataset = for_naming_module(hf_username)
    
    # Разделяем на train/val (80/20)
    split_idx = int(len(dataset) * 0.8)
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # Создаем метрику LLM as Judge
    metric = create_metric()
    
    # Создаем LM для рефлексии (используем текущую настроенную LM)
    reflection_lm = dspy.settings.lm
    
    # GEPA оптимизация с рефлективной эволюцией промптов
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
    
    module = RelationNamer()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print("✅ RelationNamer оптимизирован с GEPA")
    return optimized

