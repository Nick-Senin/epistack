import json
from difflib import SequenceMatcher
from typing import Any, Dict, Mapping, Optional

import dspy

from data_models.state_triple import StateTriple


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


class StateTripleSimilarityMetric:
    """
    Числовая метрика (0–1) сходства эталонной и предсказанной троек A-T-B.
    """

    def __init__(self, component_weights: Optional[Dict[str, float]] = None):
        base_weights = {
            "initial_state": 1.0,
            "transformation": 1.0,
            "final_state": 1.0,
        }
        if component_weights:
            normalized = {}
            for key, value in component_weights.items():
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    continue
                normalized[key] = max(0.0, weight)
            self.component_weights = normalized or base_weights
        else:
            self.component_weights = base_weights

        self._total_weight = sum(self.component_weights.values()) or len(self.component_weights)

    def _normalize_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(str(value).lower().split())
        return normalized if normalized else None

    def _normalize_triple(self, payload: Mapping[str, Any]) -> Optional[StateTriple]:
        initial = self._normalize_value(payload.get("initial_state") or payload.get("initial"))
        transformation = self._normalize_value(
            payload.get("transformation") or payload.get("action")
        )
        final_state = self._normalize_value(
            payload.get("final_state") or payload.get("result") or payload.get("final")
        )
        if not (initial and transformation and final_state):
            return None
        triple: StateTriple = {
            "initial_state": initial,
            "transformation": transformation,
            "final_state": final_state,
        }
        return triple

    def _extract_triple(self, obj: Any, attr: str) -> Optional[StateTriple]:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return self._normalize_triple(obj)

        candidate = getattr(obj, attr, None)
        if candidate is None and hasattr(obj, "abstract_state_triple"):
            candidate = getattr(obj, "abstract_state_triple")
        if candidate is None and hasattr(obj, "state_triple"):
            candidate = getattr(obj, "state_triple")
        if candidate is None or not isinstance(candidate, Mapping):
            return None

        return self._normalize_triple(candidate)

    @staticmethod
    def _component_score(reference: str, predicted: str) -> float:
        if not reference and not predicted:
            return 1.0
        if not reference or not predicted:
            return 0.0
        return SequenceMatcher(None, reference, predicted).ratio()

    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        """
        Возвращает среднее по компонентам сходство (0–1).
        """
        target = self._extract_triple(example, "abstract_state_triple")
        predicted = self._extract_triple(pred, "abstract_state_triple")

        if target is None or predicted is None:
            return 0.0

        score_sum = 0.0
        for field, weight in self.component_weights.items():
            score_sum += weight * self._component_score(
                target.get(field, ""),
                predicted.get(field, ""),
            )

        return score_sum / self._total_weight if self._total_weight else 0.0

