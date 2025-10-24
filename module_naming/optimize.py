"""
Оптимизация модуля именования
"""
import dspy
from epistack_data import for_naming_module
from .module import RelationNamer


def naming_metric(example, pred, trace=None):
    """Метрика: точность названия"""
    return example.title.lower() in pred.name.lower() or pred.name.lower() in example.title.lower()


def optimize(hf_username: str):
    """
    Оптимизация модуля именования
    
    Args:
        hf_username: HuggingFace username для загрузки датасета
        
    Returns:
        Оптимизированный модуль RelationNamer
    """
    # Загрузка датасета
    trainset = for_naming_module(hf_username)
    
    # Оптимизация
    optimizer = dspy.BootstrapFewShot(
        metric=naming_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    module = RelationNamer()
    optimized = optimizer.compile(module, trainset=trainset)
    
    print("✅ RelationNamer оптимизирован")
    return optimized

