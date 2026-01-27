"""
Сигнатуры DSPy для структурированного суммаризатора.
"""
import dspy


class GistSignature(dspy.Signature):
    """
    Краткая выжимка по одному чанку с учетом контекста секции.

    Требования:
    - 2–5 предложений (не 1)
    - обязательно упоминай ключевые сущности/имена/проекты/цифры из чанка (если есть)
    - не добавляй фактов, которых нет в тексте
    """

    parent_headings = dspy.InputField(desc="Текущая цепочка заголовков секции.")
    chunk = dspy.InputField(desc="Текстовый чанк для резюмирования.")
    gist = dspy.OutputField(
        desc="Короткая выжимка (2-5 предложений) о содержании чанка; с ключевыми именами/цифрами."
    )


class ContentHeadingsSignature(dspy.Signature):
    """
    Сформировать список подзаголовков следующего уровня.

    Требования:
    - верни СПИСОК строк (не один текстовый блок)
    - 6–12 заголовков, чтобы покрыть ВСЕ chunk_gists
    - заголовки короткие (2–7 слов), без префикса '#', без нумерации
    - не делай заголовки из 1–2 символов
    """

    parent_headings = dspy.InputField(desc="Текущая цепочка заголовков секции.")
    chunk_gists = dspy.InputField(desc="Список кратких выжимок по чанкам этой секции.")
    content_headings = dspy.OutputField(
        desc="Список подзаголовков следующего уровня (без префикса #)."
    )


class ChunkTopicSignature(dspy.Signature):
    """
    Классифицировать чанк по одной из тем.

    Требования:
    - верни РОВНО одну тему из content_headings (строгое совпадение)
    - если сомневаешься — выбери наиболее близкую по смыслу, но всё равно из списка
    """

    parent_headings = dspy.InputField(desc="Текущая цепочка заголовков секции.")
    chunk = dspy.InputField(desc="Текстовый чанк.")
    content_headings = dspy.InputField(desc="Список допустимых тем для выбора.")
    topic = dspy.OutputField(desc="Одна тема из списка content_headings.")


class SubsectionSignature(dspy.Signature):
    """
    Суммаризовать список чанков в Markdown-раздел.

    Требования к покрытию:
    - покрывай содержательно ВСЕ content_chunks (не фокусируйся на одном фрагменте)
    - сохраняй хронологию/порядок идей, если это интервью/повествование

    Формат:
    - Markdown
    - допускаются только заголовки на 1 уровень глубже, чем последний заголовок из parent_headings
    - затем 8–20 пунктов (буллеты) с ключевыми тезисами/деталями
    """

    parent_headings = dspy.InputField(desc="Цепочка заголовков секции.")
    content_chunks = dspy.InputField(desc="Список чанков текущей секции.")
    subsection = dspy.OutputField(desc="Markdown-текст секции.")
