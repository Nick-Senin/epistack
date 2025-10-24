from dspy import InputField, OutputField, Signature


class AbstractATBSig(Signature):
    """Абстрагировать связку в формат A-T-B (Агент/Действие-Превращение-Результат).
    Вернуть JSON: {"A": "...", "T": "...", "B": "..."}.
    A — кто/что действует; T — какое преобразование/действие; B — результат/состояние."""
    relation = InputField()
    atb_json = OutputField()

