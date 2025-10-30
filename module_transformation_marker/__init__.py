"""
Модуль TransformationMarker - выделение текста, связанного с преобразованиями

Использует LLM для семантического поиска частей текста, относящихся к указанным
преобразованиям, и выделяет их жирным шрифтом в markdown формате.
"""
from .module import TransformationMarker
from .signatures import TransformationMarkerSignature
from .optimize import optimize, create_example_dataset
from .metrics import TransformationMarkerMetric
from .config import configure_module_llm

__all__ = [
    "TransformationMarker",
    "TransformationMarkerSignature", 
    "optimize",
    "create_example_dataset",
    "TransformationMarkerMetric",
    "configure_module_llm"
]

