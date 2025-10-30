"""
Модуль форматирования текста с сохранением содержимого
"""
from .module import TextFormatter
from .signatures import FormatterSignature
from .optimize import optimize
from .metrics import FormatterMetric
from .config import configure_module_llm

__all__ = [
    "TextFormatter",
    "FormatterSignature", 
    "optimize",
    "FormatterMetric",
    "configure_module_llm"
]
