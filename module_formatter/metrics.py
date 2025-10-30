"""
Метрики для оценки качества форматирования текста
"""
import dspy
import re
from difflib import SequenceMatcher


class JudgeFormattingSignature(dspy.Signature):
    """
    Оцени качество форматирования текста.
    
    Критерии оценки:
    1. Сохранение содержимого (0-40 баллов):
       - Вся информация из исходного текста присутствует
       - Ничего не добавлено, ничего не удалено
       - Смысл полностью сохранен
    
    2. Markdown форматирование (0-30 баллов):
       - Правильные заголовки для структуры
       - Списки где уместно
       - Выделение для акцентов
    
    3. Читаемость (0-30 баллов):
       - Правильные абзацы и переносы
       - Исправленные пробелы и пунктуация
       - Улучшенная визуальная структура
    
    Верни оценку от 0 до 100 и детальное обоснование.
    """
    
    original_text = dspy.InputField(desc="Исходный текст")
    formatted_text = dspy.InputField(desc="Отформатированный текст")
    
    score = dspy.OutputField(desc="Оценка качества от 0 до 100")
    reasoning = dspy.OutputField(desc="Детальное обоснование оценки")


class FormatterMetric:
    """
    Метрика для оценки качества форматирования текста.
    
    Комбинирует автоматические проверки с LLM as Judge для
    комплексной оценки сохранения содержимого и качества форматирования.
    """
    
    def __init__(self):
        """Инициализация метрики с LLM судьей"""
        self.judge = dspy.ChainOfThought(JudgeFormattingSignature)
    
    def _calculate_content_similarity(self, text1, text2):
        """
        Рассчитывает similarity между двумя текстами.
        
        Используется для проверки сохранения содержимого.
        """
        # Нормализуем тексты (убираем markdown, пробелы)
        def normalize(text):
            # Убираем markdown символы
            text = re.sub(r'[#*`_\[\]()]', '', text)
            # Убираем лишние пробелы
            text = re.sub(r'\s+', ' ', text)
            return text.strip().lower()
        
        norm1 = normalize(text1)
        norm2 = normalize(text2)
        
        # Используем SequenceMatcher для similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity
    
    def _check_markdown_elements(self, text):
        """
        Проверяет наличие markdown элементов в тексте.
        
        Returns:
            float: Оценка от 0.0 до 1.0
        """
        score = 0.0
        max_score = 4.0
        
        # Проверка заголовков
        if re.search(r'^#{1,6}\s+.+', text, re.MULTILINE):
            score += 1.0
        
        # Проверка списков
        if re.search(r'^[-*+]\s+.+', text, re.MULTILINE) or \
           re.search(r'^\d+\.\s+.+', text, re.MULTILINE):
            score += 1.0
        
        # Проверка выделения
        if re.search(r'\*\*.+\*\*', text) or re.search(r'\*.+\*', text):
            score += 1.0
        
        # Проверка кода
        if re.search(r'`.+`', text):
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    def __call__(self, example, pred, trace=None):
        """
        Оценка качества форматирования
        
        Args:
            example: Пример из датасета с полем text и опционально expected_formatted_text
            pred: Предсказание модуля с полем formatted_text
            trace: Опциональный трейс выполнения
            
        Returns:
            dspy.Prediction: Оценка с score и feedback
        """
        try:
            original_text = example.text
            formatted_text = pred.formatted_text
            
            # 1. Автоматическая проверка сохранения содержимого
            content_similarity = self._calculate_content_similarity(
                original_text, 
                formatted_text
            )
            
            # 2. Проверка наличия markdown элементов
            markdown_score = self._check_markdown_elements(formatted_text)
            
            # 3. LLM as Judge для комплексной оценки
            judge_result = self.judge(
                original_text=original_text,
                formatted_text=formatted_text
            )
            
            # Извлекаем оценку от LLM
            try:
                llm_score = float(judge_result.score) / 100.0
            except (ValueError, AttributeError):
                llm_score = 0.5
            
            # Комбинируем оценки с весами
            # Сохранение содержимого - самое важное (50%)
            # LLM оценка - 30%
            # Markdown элементы - 20%
            final_score = (
                content_similarity * 0.5 +
                llm_score * 0.3 +
                markdown_score * 0.2
            )
            
            # Формируем детальный feedback
            feedback = f"""
Оценка форматирования:
- Сохранение содержимого: {content_similarity:.2%}
- Markdown элементы: {markdown_score:.2%}
- LLM оценка: {llm_score:.2%}
- Итоговая оценка: {final_score:.2%}

LLM обоснование: {judge_result.reasoning}

Рекомендации:
"""
            if content_similarity < 0.9:
                feedback += "\n- ⚠️  КРИТИЧНО: Содержимое изменено! Сохраняй всю информацию."
            if markdown_score < 0.3:
                feedback += "\n- Добавь больше markdown форматирования (заголовки, списки)."
            if content_similarity >= 0.95 and markdown_score >= 0.5:
                feedback += "\n- ✅ Отличное форматирование с сохранением содержимого!"
            
            return dspy.Prediction(
                score=final_score,
                feedback=feedback
            )
            
        except Exception as e:
            # В случае ошибки возвращаем низкую оценку
            return dspy.Prediction(
                score=0.0,
                feedback=f"Ошибка при оценке: {str(e)}"
            )
