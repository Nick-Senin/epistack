import json
import dspy


class JudgeEquivalenceSig(dspy.Signature):
    """LLM-as-judge: 0/1 эквивалентность по смыслу между исходной связкой и проверяемым текстом.
    Верни строго 0 или 1, и короткое пояснение JSON: {"score": 0|1, "why": "..."}"""
    source_relation = dspy.InputField()
    candidate_relation = dspy.InputField()
    verdict_json = dspy.OutputField()


class ConcretizationMetrics(dspy.Module):
    """Метрики для оценки качества конкретизации (id:23)"""
    def __init__(self):
        super().__init__()
        self.eq = dspy.Predict(JudgeEquivalenceSig)

    def _score_json(self, raw: str) -> int:
        try:
            return int(json.loads(raw)["score"])
        except Exception:
            return 1 if "1" in raw.strip().split()[:1] else 0

    def equivalence_after_concretization(self, source_rel: str, candidate_rel: str) -> int:
        """Метрика id:23 - эквивалентность после конкретизации"""
        raw = self.eq(source_relation=source_rel, candidate_relation=candidate_rel).verdict_json
        return self._score_json(raw)

