from .module import StateTransformationExtractor
from .signatures import StateTransformationAnalyzerSignature
from .optimize import optimize

# Для обратной совместимости
RelationExtractor = StateTransformationExtractor

__all__ = ["StateTransformationExtractor", "RelationExtractor", "StateTransformationAnalyzerSignature", "optimize"]
