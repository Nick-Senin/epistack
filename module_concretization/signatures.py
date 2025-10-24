from dspy import InputField, OutputField, Signature


class ConcretizeFromATBSig(Signature):
    """Сконкретизировать A-T-B обратно в связку с опорой на исходный контекст.
    Вернуть одну строку — конкретную связку."""
    atb_json = InputField()
    source_context = InputField()
    relation = OutputField()


class CritiqueSig(Signature):
    """Критиковать текущий черновик (конкретизацию или абстракцию): найти неточности/потери смысла."""
    draft = InputField()
    source = InputField()
    critique = OutputField()


class ReviseSig(Signature):
    """Уточнить/переписать черновик с учётом критики, сохраняя смысл источника."""
    draft = InputField()
    critique = InputField()
    improved = OutputField()

