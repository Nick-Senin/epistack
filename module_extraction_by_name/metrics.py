"""
Метрики для оптимизации модуля извлечения связок (Extraction Module)
"""
import json
import dspy
from typing import Optional, List, Any
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

# -------------------------------------------------------------------------------------------
# Сигнатуры для LLM-судьи
# -------------------------------------------------------------------------------------------

class ExtractionQualitySignature(dspy.Signature):
    """
    Проведи детальную оценку качества извлечения причинно-следственных связок.
    
    Твоя задача - сравнить предсказанные цепочки (predicted) с эталонными (expected) и оценить точность выделения КАЖДОГО компонента.
    
    Алгоритм оценки для каждой ожидаемой связки:
    1. Сопоставление: Найди соответствующую предсказанную связку по смыслу. Если связка не найдена - это грубая ошибка.
    2. Проверка компонентов (Semantic Match):
       - Initial State (Начальное состояние): Точно ли выделен субъект/состояние до изменения? Совпадает ли смысл с эталоном?
       - Transformation (Преобразование): Точно ли выделено действие/событие?
       - Final State (Конечное состояние): Точно ли выделен результат/последствия?
    
    Правила начисления баллов:
    - Идеальное совпадение всех компонентов всех связок = 1.0
    - Если связка найдена, но один из компонентов (например, transformation) неверен или искажает смысл -> Снижай оценку (частичный балл за связку).
    - Если связка пропущена целиком -> 0 баллов за эту связку.
    - Если предсказаны лишние связки (галлюцинации), которых нет в тексте -> Штраф.
    
    ВАЖНО: Оценивай смысловое совпадение (Semantic Match), а не дословное. Разные формулировки одного смысла допустимы.
    Но если смысл компонента искажен или взят другой фрагмент текста - это ошибка выделения.
    
    Ответ должен содержать:
    - score: Результирующая оценка от 0.0 до 1.0.
    - feedback: Детальный разбор ошибок. Укажи конкретно, какая часть (initial/transformation/final) неверна в какой связке.
    """
    source_text = dspy.InputField(desc="Исходный текст")
    expected_chains = dspy.InputField(desc="Эталонные цепочки (JSON) [{'initial_state':..., 'transformation':..., 'final_state':...}]")
    predicted_chains = dspy.InputField(desc="Предсказанные цепочки (JSON)")
    
    score = dspy.OutputField(desc="Оценка качества (0.0 - 1.0)")
    feedback = dspy.OutputField(desc="Анализ ошибок: какие конкретно компоненты (initial/transformation/final) неверны")


# -------------------------------------------------------------------------------------------
# Класс метрики
# -------------------------------------------------------------------------------------------

class ExtractionMetric:
    """LLM-as-Judge метрика для оценки качества извлечения связок"""
    
    def __init__(self):
        self.judge = dspy.ChainOfThought(ExtractionQualitySignature)
    
    @staticmethod
    def _clip_score(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _format_chains(self, chains: Any) -> str:
        """Безопасное форматирование цепочек в JSON-строку"""
        try:
            if hasattr(chains, 'to_dict'):
                return json.dumps(chains.to_dict(), ensure_ascii=False, indent=2)
            if isinstance(chains, (list, dict)):
                return json.dumps(chains, ensure_ascii=False, indent=2)
            return str(chains)
        except Exception:
            return str(chains)

    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Оценка совпадения предсказанных связок с эталоном.
        
        Args:
            example: Пример с полем extracted_chains
            pred: Предикт с полем state_analysis
        """
        try:
            # 1. Получаем данные
            source_text = getattr(example, 'source_text', "")
            
            expected = getattr(example, 'extracted_chains', [])
            predicted = getattr(pred, 'state_analysis', [])
            
            # 2. Базовая проверка типов
            if not isinstance(predicted, list):
                return dspy.Prediction(
                    score=0.0,
                    feedback=f"Ошибка формата: ожидался список, получен {type(predicted)}"
                )

            # 3. Если оба пустые - идеально (но странно для задачи извлечения)
            if not expected and not predicted:
                return dspy.Prediction(score=1.0, feedback="Оба списка пусты (совпадение).")

            # 4. Форматируем для судьи
            expected_str = self._format_chains(expected)
            predicted_str = self._format_chains(predicted)
            
            # 5. LLM Judge оценивает
            judgment = self.judge(
                source_text=source_text,
                expected_chains=expected_str,
                predicted_chains=predicted_str
            )
            
            score = self._clip_score(judgment.score)
            feedback = judgment.feedback
            
            return dspy.Prediction(
                score=score,
                feedback=feedback
            )
            
        except Exception as e:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Ошибка при выполнении метрики: {str(e)}"
            )


# -------------------------------------------------------------------------------------------
# Глобальный доступ к метрике (Singleton pattern, как в module_naming)
# -------------------------------------------------------------------------------------------

_EXTRACTION_METRIC_SINGLETON: Optional[ExtractionMetric] = None

def _get_extraction_metric() -> ExtractionMetric:
    global _EXTRACTION_METRIC_SINGLETON
    if _EXTRACTION_METRIC_SINGLETON is None:
        _EXTRACTION_METRIC_SINGLETON = ExtractionMetric()
    return _EXTRACTION_METRIC_SINGLETON


def create_metric():
    """Фабрика для создания метрики"""
    return ExtractionMetric()


def metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    GEPA-совместимая функция метрики.
    """
    evaluation = _get_extraction_metric()(
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
