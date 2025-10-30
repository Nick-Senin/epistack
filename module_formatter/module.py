"""
Модуль для форматирования текста с сохранением содержимого
"""
import dspy
from .signatures import FormatterSignature


class TextFormatter(dspy.Module):
    """
    Модуль для форматирования текста с улучшением читаемости.
    
    Преобразует неструктурированный текст в красиво отформатированный markdown,
    сохраняя при этом всё содержимое без изменений. Улучшает:
    - Структуру через заголовки и абзацы
    - Читаемость через списки и выделение
    - Оформление через правильные пробелы и пунктуацию
    
    Использует ChainOfThought для обдуманного форматирования.
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(FormatterSignature)
    
    def forward(self, text):
        """
        Форматирует переданный текст
        
        Args:
            text (str): Исходный неотформатированный текст
            
        Returns:
            dspy.Prediction: Результат с полем formatted_text
        """
        result = self.predictor(text=text)
        return result
