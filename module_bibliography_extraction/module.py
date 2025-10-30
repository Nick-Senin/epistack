"""
Модуль для извлечения библиографической информации из текста
"""
import dspy
from .signatures import BibliographyExtractionSignature


class BibliographyExtraction(dspy.Module):
    """
    Модуль для извлечения библиографической информации из текста.
    
    Извлекает следующие поля:
    - Название книги
    - Автор
    - Издательство
    - Год издания
    - Место издания
    - Дополнительная информация (предположения о недостающих полях)
    
    Модуль корректно обрабатывает случаи, когда некоторые поля отсутствуют в тексте.
    """
    
    def __init__(self):
        super().__init__()
        
        # Используем ChainOfThought для рассуждений при извлечении информации
        self.predictor = dspy.ChainOfThought(BibliographyExtractionSignature)
    
    def forward(self, text):
        """
        Извлечение библиографической информации из текста
        
        Args:
            text (str): Текст для анализа, содержащий библиографическую информацию
            
        Returns:
            dspy.Prediction: Объект с полями title, author, publisher, year, place, inferred_info
        """
        # Предобработка: удаляем лишние пробелы
        if text:
            text = text.strip()
        
        # Вызываем предиктор для извлечения информации
        result = self.predictor(text=text)
        
        # Постобработка: форматируем выходные данные
        # Убеждаемся, что каждое поле имеет правильный формат
        result.title = self._format_field(result.title, "Название")
        result.author = self._format_field(result.author, "Автор")
        result.publisher = self._format_field(result.publisher, "Издательство")
        result.year = self._format_field(result.year, "Год издания")
        result.place = self._format_field(result.place, "Место издания")
        
        # inferred_info оставляем как есть (без жесткого форматирования)
        if not hasattr(result, 'inferred_info') or not result.inferred_info:
            result.inferred_info = "Нет дополнительных предположений"
        
        return result
    
    def _format_field(self, value, field_name):
        """
        Форматирует поле в требуемый формат
        
        Args:
            value (str): Значение поля
            field_name (str): Название поля
            
        Returns:
            str: Отформатированное значение
        """
        if not value:
            return f"**{field_name}:** Не указано"
        
        # Если значение уже содержит правильный формат, возвращаем как есть
        if value.startswith(f"**{field_name}:**"):
            return value
        
        # Иначе добавляем формат
        return f"**{field_name}:** {value.strip()}"

