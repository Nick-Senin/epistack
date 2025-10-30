from dspy import InputField, OutputField, Signature


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

