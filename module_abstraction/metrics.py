import json
from typing import Dict
import dspy
from .signatures import AbstractATBSig


class JudgeAbstractionQualitySig(dspy.Signature):
    """Оценка достаточности абстракции: A-T-B должно обобщать связку без потери ключевого смысла.
    Верни JSON: {"score": 0|1, "why": "..."}"""
    relation = dspy.InputField()
    atb_json = dspy.InputField()
    verdict_json = dspy.OutputField()


class JudgeOverAbstractionSig(dspy.Signature):
    """Оценка пере-абстракции: A-T-B не должен быть слишком общим.
    Верни JSON: {"score": 0|1, "why": "..."} где 1 = НЕ пере-абстрагировано."""
    relation = dspy.InputField()
    atb_json = dspy.InputField()
    verdict_json = dspy.OutputField()


class AbstractionMetrics(dspy.Module):
    """Метрики для оценки качества абстрагирования (id:24)"""
    def __init__(self):
        super().__init__()
        self.absq = dspy.Predict(JudgeAbstractionQualitySig)
        self.no_over = dspy.Predict(JudgeOverAbstractionSig)

    def _score_json(self, raw: str) -> int:
        try:
            return int(json.loads(raw)["score"])
        except Exception:
            return 1 if "1" in raw.strip().split()[:1] else 0

    def sufficient_abstraction(self, relation: str, atb: Dict[str,str]) -> int:
        """Метрика id:24 - достаточность абстракции"""
        raw = self.absq(relation=relation, atb_json=json.dumps(atb, ensure_ascii=False)).verdict_json
        return self._score_json(raw)

    def not_over_abstracted(self, relation: str, atb: Dict[str,str]) -> int:
        """Метрика id:24 - отсутствие пере-абстрагирования"""
        raw = self.no_over(relation=relation, atb_json=json.dumps(atb, ensure_ascii=False)).verdict_json
        return self._score_json(raw)

