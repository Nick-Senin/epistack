"""
Модуль для извлечения библиографической информации из текста

Этот модуль использует DSPy для извлечения структурированной библиографической
информации из неструктурированного текста.

Основные компоненты:
- BibliographyExtraction: основной модуль для извлечения
- BibliographyExtractionSignature: сигнатура DSPy
- BibliographyMetric: метрика для оценки качества
- optimize: функция для оптимизации модуля
- configure_module_llm: настройка LLM

Пример использования:
    >>> from module_bibliography_extraction import BibliographyExtraction, configure_module_llm
    >>> 
    >>> # Настройка LLM
    >>> configure_module_llm()
    >>> 
    >>> # Создание и использование модуля
    >>> module = BibliographyExtraction()
    >>> result = module(text="Война и мир. Лев Толстой. М.: Наука, 1869.")
    >>> 
    >>> print(result.title)
    >>> print(result.author)
    >>> print(result.publisher)
    >>> print(result.year)
    >>> print(result.place)
"""

from .module import BibliographyExtraction
from .signatures import BibliographyExtractionSignature
from .optimize import optimize
from .metrics import BibliographyMetric
from .config import configure_module_llm

__all__ = [
    "BibliographyExtraction",
    "BibliographyExtractionSignature", 
    "optimize",
    "BibliographyMetric",
    "configure_module_llm"
]

__version__ = "1.0.0"
__author__ = "epistack"
__description__ = "Модуль для извлечения библиографической информации из текста"

