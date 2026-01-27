"""
Метрики для оптимизации модуля именования
"""
import dspy
from typing import Optional
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback


class TitleSimilaritySignature(dspy.Signature):
    """
    Оцени, насколько хорошо сгенерированное название совпадает с эталонным.
    
    Критерии:
    1. Смысловое соответствие (главная идея и ключевые элементы).
    2. Упоминание ключевых сущностей / действий.
    3. Структурная и стилистическая близость (по необходимости).
    
    Верни итоговый скор от 0.0 до 1.0 и краткое объяснение различий.
    """
    original_title = dspy.InputField(desc="Эталонное название текста")
    generated_title = dspy.InputField(desc="Название, предложенное моделью")
    
    score = dspy.OutputField(desc="Сходство от 0.0 до 1.0")
    feedback = dspy.OutputField(desc="Объяснение: что совпало, чего не хватает")


class ContextCoverageSignature(dspy.Signature):
    """
    Проверь, содержит ли предложенное название явное указание на контекст исходного текста:
    проект, организацию, коллектив, ключевое действующее лицо или сущность.
    
    Критерии:
    1. Название должно упоминать специфический объект или акторов, а не быть абстрактным.
    2. Допустимы отсылки к проекту/инициативе, названию команды, героям, месту действия.
    3. Если контекст отсутствует, опиши, чего не хватает.
    
    Верни скор от 0.0 до 1.0 и краткий фидбек.
    """
    source_text = dspy.InputField(desc="Исходный текст или фрагмент, описывающий кейс")
    generated_title = dspy.InputField(desc="Название, предложенное моделью")
    
    score = dspy.OutputField(desc="0.0 — контекст отсутствует, 1.0 — контекст явно отражён")
    feedback = dspy.OutputField(desc="Комментарий о том, что найдено или чего не хватает")


class NamingMetric:
    """LLM-as-Judge метрика для оценки качества и контекстуальности названий"""
    
    def __init__(self, context_weight: float = 0.4):
        context_weight = max(0.0, min(1.0, float(context_weight)))
        self.similarity_weight = 1.0 - context_weight
        self.context_weight = context_weight
        
        self.similarity_judge = dspy.ChainOfThought(TitleSimilaritySignature)
        self.context_judge = dspy.ChainOfThought(ContextCoverageSignature)
    
    @staticmethod
    def _clip_score(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    
    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Оценка качества и наличия контекстуальных указаний в названии
        
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
            
            # Получаем названия
            original_title = getattr(example, 'title', None) or getattr(example, 'name', "")
            source_text = (
                getattr(example, 'source_text', None)
                or getattr(example, 'text', None)
                or getattr(example, 'content', None)
            )
            
            # LLM as Judge оценивает совпадение между оригиналом и предсказанием
            similarity_judgment = self.similarity_judge(
                original_title=original_title,
                generated_title=generated_name
            )
            
            # Парсим оценку
            similarity_score = self._clip_score(similarity_judgment.score)
            similarity_feedback = similarity_judgment.feedback
            
            # Проверка, что в названии отражён контекст
            context_score = 1.0
            context_feedback = ""
            if source_text:
                try:
                    context_judgment = self.context_judge(
                        source_text=source_text,
                        generated_title=generated_name,
                    )
                    context_score = self._clip_score(context_judgment.score)
                    context_feedback = context_judgment.feedback
                except Exception as context_error:
                    context_score = 0.0
                    context_feedback = f"Ошибка контекстной проверки: {context_error}"
            else:
                context_feedback = "Контекстная проверка пропущена: нет исходного текста."
            
            # Агрегируем общий скор
            combined_score = (
                self.similarity_weight * similarity_score +
                self.context_weight * context_score
            )
            combined_score = self._clip_score(combined_score)
            
            feedback_parts = []
            if similarity_feedback:
                feedback_parts.append(f"Сходство: {similarity_feedback}")
            if context_feedback:
                feedback_parts.append(f"Контекст: {context_feedback}")
            combined_feedback = "\n".join(feedback_parts).strip()
            
            # Возвращаем результат с фидбеком для GEPA
            return dspy.Prediction(
                score=combined_score,
                feedback=combined_feedback
            )
            
        except Exception as e:
            # В случае ошибки возвращаем низкую оценку с описанием проблемы
            return dspy.Prediction(
                score=0.0,
                feedback=f"Ошибка при оценке: {str(e)}"
            )


_NAMING_METRIC_SINGLETON: Optional[NamingMetric] = None


def _get_naming_metric() -> NamingMetric:
    global _NAMING_METRIC_SINGLETON
    if _NAMING_METRIC_SINGLETON is None:
        _NAMING_METRIC_SINGLETON = NamingMetric()
    return _NAMING_METRIC_SINGLETON


def create_metric():
    """Фабрика для создания метрики"""
    return NamingMetric()


def metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    GEPA-совместимая метрика, возвращающая числовой скор или ScoreWithFeedback.
    
    Args:
        gold: Эталонный пример (dspy.Example).
        pred: Предсказание модели (dspy.Prediction).
        trace: Полный трейс выполнения программы.
        pred_name: Имя конкретного предиктора, для которого запрашивается фидбек.
        pred_trace: Трейс, соответствующий предиктору `pred_name`.
    """
    evaluation = _get_naming_metric()(
        example=gold,
        pred=pred,
        trace=trace,
        pred_name=pred_name,
        pred_trace=pred_trace,
    )
    score = float(getattr(evaluation, "score", 0.0))
    feedback = getattr(evaluation, "feedback", "")
    if feedback and pred_name:
        feedback = f"[{pred_name}] {feedback}"
    if feedback:
        return ScoreWithFeedback(score=score, feedback=str(feedback))
    return score
