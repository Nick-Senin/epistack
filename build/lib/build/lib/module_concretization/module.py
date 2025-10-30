import json
from typing import Dict
import dspy
from .signatures import ConcretizeFromATBSig
from ..utils.common_signatures import CritiqueSig, ReviseSig


class ConcretizerWithReflection(dspy.Module):
    """«Продвинутый reflection model»: конкретизация + критика + правка."""
    def __init__(self, steps: int = 2):
        super().__init__()
        self.steps = steps
        self.make = dspy.ChainOfThought(ConcretizeFromATBSig)
        self.crit = dspy.Predict(CritiqueSig)
        self.rev = dspy.Predict(ReviseSig)

    def forward(self, atb: Dict[str, str], source_context: str) -> str:
        draft = self.make(atb_json=json.dumps(atb, ensure_ascii=False),
                          source_context=source_context).relation
        for _ in range(self.steps):
            critique = self.crit(draft=draft, source=source_context).critique
            draft = self.rev(draft=draft, critique=critique).improved
        return draft.strip()
