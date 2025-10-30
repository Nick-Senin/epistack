import dspy
from .signatures import StateTransformationAnalyzerSignature
from module_naming import RelationNamer


class StateTransformationExtractor(dspy.Module):
    """
    Модуль для извлечения состояний и преобразований из текста.
    Сначала получает название связки через RelationNamer, 
    затем анализирует состояния через StateTransformationAnalyzerSignature.
    """
    def __init__(self, cot=False):
        super().__init__()
        self.namer = RelationNamer()
        self.analyzer = dspy.ChainOfThought(StateTransformationAnalyzerSignature) if cot else dspy.Predict(StateTransformationAnalyzerSignature)

    def forward(self, source_text: str = None, text: str = None):
        # Поддержка обоих форматов параметров
        text_input = source_text if source_text is not None else text
        
        # Шаг 1: получаем название связки
        naming_result = self.namer(source_text=text_input)
        relation_title = naming_result.causal_relation
        
        # Шаг 2: анализируем состояния и преобразования
        analysis_result = self.analyzer(
            relation_title=relation_title,
            source_text=text_input
        )
        
        return dspy.Prediction(
            relation_title=relation_title,
            state_analysis=analysis_result.state_analysis
        )
