from dspy import InputField, OutputField, Signature
from data_models.state_triple import StateTriple


class AbstractStateTripleSignature(Signature):
    """Абстрагировать исходную тройку (initial_state, transformation, final_state).
    Убрать детали, которые не важны для выполнения transformation.
    Вернуть абстрактную тройку с сохранением ключевых элементов для трансформации."""
    state_triple: StateTriple = InputField()
    abstract_state_triple: StateTriple = OutputField()

