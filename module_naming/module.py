import dspy
from .signatures import CausalRelationExtractorSignature

class RelationNamer(dspy.Module):
    """
    Модуль для извлечения причинно-следственных связей из текста.
    """
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(CausalRelationExtractorSignature)

    def forward(self, text):
        """
        Извлекает причинно-следственные связи из текста.
        
        Args:
            text (str): Текст для анализа
            
        Returns:
            dspy.Prediction: Результат с полем causal_relation
        """
        result = self.extractor(text_fragment=text)
        return result
