import json
from typing import Dict
import dspy
from .signatures import AbstractATBSig
from ..utils.common_signatures import CritiqueSig, ReviseSig
from ..utils.helpers import safe_json_dict


class NaiveATBAbstraction(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.Predict(AbstractATBSig)

    def forward(self, relation: str) -> Dict[str, str]:
        raw = self.pred(relation=relation).atb_json
        return safe_json_dict(raw, keys=("A","T","B"))


class EfficientATBAbstraction(dspy.Module):
    """Более «эффективный» модуль: CoT + короткая внутренняя саморефлексия."""
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(AbstractATBSig)
        self.crit = dspy.Predict(CritiqueSig)
        self.rev = dspy.Predict(ReviseSig)

    def forward(self, relation: str) -> Dict[str, str]:
        draft = self.cot(relation=relation).atb_json
        atb = safe_json_dict(draft, keys=("A","T","B"))
        # мини-рефлексия: сверим обратно на текстовую формулировку
        draft_text = json.dumps(atb, ensure_ascii=False)
        critique = self.crit(draft=draft_text, source=relation).critique
        improved = self.rev(draft=draft_text, critique=critique).improved
        atb2 = safe_json_dict(improved, keys=("A","T","B"))
        return atb2 or atb
