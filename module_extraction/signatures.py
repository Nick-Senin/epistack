from dspy import InputField, OutputField, Signature


class ExtractRelationsSig(Signature):
    """Извлечь из фрагмента текста список "связок" (коротких утверждений-соотношений).
    Формат ответа: JSON-список строк. Пример: ["X преобразует Y в Z", "..."]"""
    text = InputField()
    relations = OutputField(desc="JSON list of short relation strings")

