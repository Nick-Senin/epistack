"""
Метрики для оценки качества извлечения библиографической информации
"""
import dspy
from difflib import SequenceMatcher


class BibliographyMetric:
    """
    Метрика для оценки качества извлечения библиографической информации.
    
    Оценивает качество по каждому из пяти полей:
    - Название
    - Автор
    - Издательство
    - Год издания
    - Место издания
    
    Использует частичное совпадение для более гибкой оценки.
    Учитывает случаи отсутствия информации.
    """
    
    def __init__(self, threshold=0.8):
        """
        Инициализация метрики
        
        Args:
            threshold (float): Порог для частичного совпадения (0.0 - 1.0)
        """
        self.threshold = threshold
        self.field_names = ['title', 'author', 'publisher', 'year', 'place']
    
    def __call__(self, example, pred, trace=None):
        """
        Оценка качества предсказания
        
        Args:
            example: Пример из датасета с ground truth (ожидаемые значения)
            pred: Предсказание модуля
            trace: Опциональный трейс выполнения (для отладки)
            
        Returns:
            dspy.Prediction: Оценка с полями score и feedback
        """
        try:
            scores = []
            feedback_parts = []
            
            # Оцениваем каждое поле
            for field_name in self.field_names:
                expected = self._extract_value(getattr(example, field_name, ""))
                predicted = self._extract_value(getattr(pred, field_name, ""))
                
                # Вычисляем оценку для поля
                field_score = self._calculate_field_score(expected, predicted)
                scores.append(field_score)
                
                # Формируем feedback
                if field_score == 1.0:
                    feedback_parts.append(f"✓ {field_name}: точное совпадение")
                elif field_score > 0.5:
                    feedback_parts.append(f"~ {field_name}: частичное совпадение ({field_score:.2f})")
                else:
                    feedback_parts.append(f"✗ {field_name}: не совпадает (ожидалось: '{expected}', получено: '{predicted}')")
            
            # Общая оценка - среднее по всем полям
            total_score = sum(scores) / len(scores) if scores else 0.0
            
            # Детальный feedback
            feedback = f"Общая оценка: {total_score:.2f}\n" + "\n".join(feedback_parts)
            
            return dspy.Prediction(
                score=total_score,
                feedback=feedback
            )
            
        except Exception as e:
            # В случае ошибки возвращаем низкую оценку
            return dspy.Prediction(
                score=0.0,
                feedback=f"Ошибка при оценке: {str(e)}"
            )
    
    def _extract_value(self, field_text):
        """
        Извлекает значение из форматированного поля
        
        Args:
            field_text (str): Текст поля в формате "**Название:** значение"
            
        Returns:
            str: Извлеченное значение
        """
        if not field_text:
            return ""
        
        # Ищем двоеточие и берем все после него
        if ":**" in field_text or ":" in field_text:
            parts = field_text.split(":", 1)
            if len(parts) > 1:
                value = parts[1].strip()
                # Убираем "Не указано" как пустое значение
                if value.lower() in ["не указано", "не найдено", "отсутствует", ""]:
                    return ""
                return value
        
        return field_text.strip()
    
    def _calculate_field_score(self, expected, predicted):
        """
        Вычисляет оценку для одного поля
        
        Args:
            expected (str): Ожидаемое значение
            predicted (str): Предсказанное значение
            
        Returns:
            float: Оценка от 0.0 до 1.0
        """
        # Нормализуем строки
        expected = expected.lower().strip()
        predicted = predicted.lower().strip()
        
        # Оба пустые - идеальное совпадение (правильно определили отсутствие)
        if not expected and not predicted:
            return 1.0
        
        # Одно пустое, другое нет
        if not expected or not predicted:
            return 0.0
        
        # Exact match
        if expected == predicted:
            return 1.0
        
        # Частичное совпадение с использованием SequenceMatcher
        similarity = SequenceMatcher(None, expected, predicted).ratio()
        
        # Также проверяем, содержится ли одно в другом
        if expected in predicted or predicted in expected:
            similarity = max(similarity, 0.8)
        
        return similarity

