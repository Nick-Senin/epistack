from dspy import InputField, OutputField, Signature


class ConcretizeFromATBSig(Signature):
    """Сконкретизировать A-T-B обратно в связку с опорой на исходный контекст.
    Вернуть одну строку — конкретную связку."""
    atb_json = InputField()
    source_context = InputField()
    relation = OutputField()

