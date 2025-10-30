import dspy
from .signatures import CausalRelationExtractorSignature

class RelationNamer(dspy.Module):
    """
    Модуль для создания названия для текста на основе причинно-следственных связей.
    """
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(CausalRelationExtractorSignature)

    def forward(self, source_text=None, text=None):
        # Поддержка обоих форматов: source_text (из датасета) и text (legacy)
        text_input = source_text if source_text is not None else text
        result = self.extractor(text_fragment=text_input)
        return result
