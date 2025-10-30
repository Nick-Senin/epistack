"""
Оптимизация модуля абстракции
"""
import dspy
from epistack_data import for_abstraction_module
from .module import NaiveATBAbstraction


def abstraction_metric(example, pred, trace=None):
    """Метрика: абстракции отличаются от конкретных названий"""
    if not hasattr(pred, 'abstracted_chains'):
        return False
    
    # Простая проверка: абстрактные термины должны быть более общими
    abstracted = str(pred.abstracted_chains).lower()
    original = str(example.extracted_chains).lower()
    
    # Не должны быть идентичны
    return abstracted != original


def optimize(hf_username: str):
    """
    Оптимизация модуля абстракции
    
    Args:
        hf_username: HuggingFace username для загрузки датасета
        
    Returns:
        Оптимизированный модуль NaiveATBAbstraction
    """
    # Загрузка датасета
    trainset = for_abstraction_module(hf_username)
    
    # Оптимизация
    optimizer = dspy.BootstrapFewShot(
        metric=abstraction_metric,
        max_bootstrapped_demos=3
    )
    
    module = NaiveATBAbstraction()
    optimized = optimizer.compile(module, trainset=trainset)
    
    print("✅ NaiveATBAbstraction оптимизирован")
    return optimized

