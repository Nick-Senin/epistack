"""
Оптимизация модуля извлечения связок
"""
import dspy
from epistack_data import for_extraction_module
from .module import RelationExtractor


def extraction_metric(example, pred, trace=None):
    """Метрика: количество найденных связок"""
    expected_count = len(example.extracted_chains)
    predicted_count = len(pred.chains) if hasattr(pred, 'chains') else 0
    
    # Допуск ±1
    return abs(expected_count - predicted_count) <= 1


def optimize(hf_username: str):
    """
    Оптимизация модуля извлечения связок
    
    Args:
        hf_username: HuggingFace username для загрузки датасета
        
    Returns:
        Оптимизированный модуль RelationExtractor
    """
    # Загрузка датасета
    trainset = for_extraction_module(hf_username)
    
    # Оптимизация
    optimizer = dspy.BootstrapFewShot(
        metric=extraction_metric,
        max_bootstrapped_demos=3
    )
    
    module = RelationExtractor()
    optimized = optimizer.compile(module, trainset=trainset)
    
    print("✅ RelationExtractor оптимизирован")
    return optimized

