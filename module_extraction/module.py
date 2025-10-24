import json
from typing import List
import dspy
from .signatures import ExtractRelationsSig


class RelationExtractor(dspy.Module):
    def __init__(self, cot=False):
        super().__init__()
        self.pred = dspy.ChainOfThought(ExtractRelationsSig) if cot else dspy.Predict(ExtractRelationsSig)

    def forward(self, text: str) -> List[str]:
        out = self.pred(text=text).relations
        try:
            return json.loads(out)
        except Exception:
            # fallback: пытаться выделить по строкам
            return [s.strip("-• ").strip() for s in out.split("\n") if s.strip()]
