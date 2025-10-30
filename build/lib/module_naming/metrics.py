"""
Метрики для оптимизации модуля именования
"""
import dspy
from typing import Optional


class JudgeSignature(dspy.Signature):
    """
    Оцени качество сгенерированного названия для текста.
    
    Критерии оценки (каждый от 0 до 1):
    1. Точность: насколько название отражает основную идею текста и оригинальное название
    2. Краткость: соответствует ли формату (до 20 слов, без точки)
    3. Информативность: содержит ли причинно-следственные связи
    4. Качество формулировки
    
    Верни общую оценку от 0.0 до 1.0 и объяснение.
    """
    original_title = dspy.InputField(desc="Оригинальное название текста")
    text_fragment = dspy.InputField(desc="Фрагмент текста")
    generated_name = dspy.InputField(desc="Сгенерированное название")
    
    score = dspy.OutputField(desc="Оценка от 0.0 до 1.0")
    feedback = dspy.OutputField(desc="Детальное объяснение оценки с указанием недостатков и рекомендаций")


class NamingMetric:
    """Метрика с LLM as Judge для оценки качества названий"""
    
    def __init__(self):
        self.judge = dspy.ChainOfThought(JudgeSignature)
    
    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Оценка качества сгенерированного названия
        
        Args:
            example: Пример с полями title и source_text
            pred: Предикт с полем causal_relation
            trace: Опциональный трейс выполнения программы
            pred_name: Имя предиктора (для GEPA)
            pred_trace: Трейс предиктора (для GEPA)
            
        Returns:
            dspy.Prediction с score и feedback
        """
        try:
            # Получаем название из предикта
            generated_name = pred.causal_relation if hasattr(pred, 'causal_relation') else pred.name if hasattr(pred, 'name') else str(pred)
            
            # Получаем текст из примера (поддержка обоих форматов)
            text_fragment = example.source_text if hasattr(example, 'source_text') else example.text if hasattr(example, 'text') else ""
            
            # LLM as Judge оценивает качество
            judgment = self.judge(
                original_title=example.title,
                text_fragment=text_fragment,
                generated_name=generated_name
            )
            
            # Парсим оценку
            score = float(judgment.score)
            score = max(0.0, min(1.0, score))  # Ограничиваем [0, 1]
            
            # Возвращаем результат с фидбеком для GEPA
            return dspy.Prediction(
                score=score,
                feedback=judgment.feedback
            )
            
        except Exception as e:
            # В случае ошибки возвращаем низкую оценку с описанием проблемы
            return dspy.Prediction(
                score=0.0,
                feedback=f"Ошибка при оценке: {str(e)}"
            )


def create_metric():
    """Фабрика для создания метрики"""
    return NamingMetric()

