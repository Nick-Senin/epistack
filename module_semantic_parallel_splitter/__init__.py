"""
Модуль для семантической сегментации текста на параллельные блоки
"""
from .module import SemanticParallelSplitter, SemanticHalver
from .signatures import SemanticSplitSignature, SemanticHalverSignature
from .optimize import optimize, SemanticHalverMetric, load_dataset, save_optimized_module, load_optimized_module, create_reflection_lm
from .metrics import SemanticSplitMetric
from .config import configure_module_llm

__all__ = [
    "SemanticParallelSplitter",
    "SemanticHalver",
    "SemanticSplitSignature",
    "SemanticHalverSignature",
    "optimize",
    "SemanticHalverMetric",
    "load_dataset",
    "save_optimized_module",
    "load_optimized_module",
    "create_reflection_lm",
    "SemanticSplitMetric",
    "configure_module_llm"
]
