from dspy import InputField, OutputField, Signature


class AbstractATBSig(Signature):
    """Абстрагировать связку в формат A-T-B (Агент/Действие-Превращение-Результат).
    Вернуть JSON: {"A": "...", "T": "...", "B": "..."}.
    A — кто/что действует; T — какое преобразование/действие; B — результат/состояние."""
    relation = InputField()
    atb_json = OutputField()


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

