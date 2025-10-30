"""
Модуль TransformationMarker для выделения текста, связанного с преобразованиями
"""
import dspy
from .signatures import TransformationMarkerSignature


class TransformationMarker(dspy.Module):
    """
    Модуль для семантического поиска и выделения жирным шрифтом (markdown)
    частей текста, которые относятся к указанным преобразованиям.
    
    Использует LLM для понимания семантической связи между текстом и преобразованиями,
    а не просто поиск по ключевым словам.
    
    Пример:
        module = TransformationMarker()
        result = module(
            text="Данные проходят нормализацию и фильтрацию...",
            transformations=["нормализация", "фильтрация"]
        )
        print(result.marked_text)  # Текст с выделениями **...**
    """
    
    def __init__(self):
        super().__init__()
        
        self.predictor = dspy.Predict(TransformationMarkerSignature)
    
    def forward(self, text, transformations):
        """
        Выделяет в тексте части, связанные с преобразованиями
        
        Args:
            text (str): Исходный текст для обработки
            transformations (list[str]): Массив преобразований для поиска
                
        Returns:
            dspy.Prediction: Результат с полем marked_text - текст с выделениями
        """
        # Вызываем предиктор для семантического анализа
        # Передаём transformations напрямую как массив
        result = self.predictor(
            text=text,
            transformations=transformations
        )
        
        # Возвращаем результат с выделенным текстом
        return result

