import dspy
from .signatures import AbstractStateTripleSignature
from data_models.state_triple import StateTriple


class NaiveStateTripleAbstraction(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.Predict(AbstractStateTripleSignature)

    def forward(self, state_triple: StateTriple) -> StateTriple:
        result = self.pred(state_triple=state_triple)
        return result.abstract_state_triple
