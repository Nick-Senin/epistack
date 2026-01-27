"""Структурированный суммаризатор на DSPy с семантическим сплиттером."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import dotenv
import dspy

from module_semantic_parallel_splitter.module import SemanticParallelSplitter

from .signatures import (
    ChunkTopicSignature,
    ContentHeadingsSignature,
    GistSignature,
    SubsectionSignature,
)

logger = logging.getLogger(__name__)


def _coerce_str_list(value: Any) -> list[str]:
    """Привести значение (иногда строковое) к list[str].

    В DSPy выходные поля, описанные как "список", иногда приходят строкой
    (JSON или маркированный список). Если строку итерировать напрямую,
    получится список символов ("Р", "е", "з", ...), и оглавление ломается.
    """

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            s = (x if isinstance(x, str) else str(x)).strip()
            if s:
                out.append(s)
        return out

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        # JSON-массив
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                return _coerce_str_list(parsed)
            except Exception:
                pass

        # Построчно / по разделителям
        parts = [p.strip() for p in re.split(r"[\r\n]+", s) if p and p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in re.split(r"[;,]+", s) if p and p.strip()]

        cleaned: list[str] = []
        for p in parts:
            p = re.sub(r"^\s*([-*•]+|\d+[\.\)])\s*", "", p).strip()
            if p:
                cleaned.append(p)
        return cleaned

    s = str(value).strip()
    return [s] if s else []


def configure_module_llm(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    use_global_config: bool = True,
    **kwargs: Any,
) -> dspy.LM:
    """Настройка LLM для структурированного суммаризатора."""

    dotenv.load_dotenv()

    if use_global_config and not any([model, api_base, api_key]):
        openrouter_model = os.getenv("OPENROUTER_MODEL", "").strip()
        if openrouter_model:
            from config.llm import configure_llm

            return configure_llm()

    model = (
        model
        or os.getenv("CEREBRAS_MODEL")
        or "cerebras/qwen-3-235b-a22b-instruct-2507"
    ).strip()
    api_base = (
        api_base or os.getenv("CEREBRAS_API_BASE") or "https://api.cerebras.ai/v1"
    ).strip()
    api_key = (api_key or os.getenv("CEREBRAS_API_KEY") or "").strip()

    if not api_key:
        raise ValueError(
            "Не найден API ключ для LLM. Укажите `CEREBRAS_API_KEY` (и опционально `CEREBRAS_MODEL`, `CEREBRAS_API_BASE`) "
            "или задайте глобальные `OPENROUTER_*` переменные и вызовите `configure_module_llm(use_global_config=True)`."
        )

    lm = dspy.LM(model=model, api_base=api_base, api_key=api_key, **kwargs)
    dspy.configure(lm=lm)
    logger.info("Module LLM configured: %s", lm.model)
    return lm


def _normalize_headings(parent_headings: list[str] | None) -> list[str]:
    headings: list[str] = []
    for i, h in enumerate(parent_headings or []):
        text = (h or "").strip()
        if not text:
            continue
        if text.lstrip().startswith("#"):
            headings.append(text)
        else:
            headings.append(f"{'#' * (i + 1)} {text}")
    if not headings:
        headings = ["# Summary"]
    return headings


def _parallel_predict(
    predictor: dspy.Module,
    inputs_list: list[dict[str, Any]],
    input_fields: list[str],
    num_threads: int,
) -> list[Any]:
    if not inputs_list:
        return []

    examples = [
        dspy.Example(**inputs).with_inputs(*input_fields) for inputs in inputs_list
    ]
    exec_pairs = [(predictor, ex) for ex in examples]
    parallel = dspy.Parallel(
        num_threads=max(1, int(num_threads)),
        max_errors=len(exec_pairs),
        provide_traceback=True,
    )
    results = parallel(exec_pairs)

    preds: list[Any] = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 2:
            preds.append(item[0])
        else:
            preds.append(item)
    return preds


def structure_and_summarize(
    parent_headings: list[str],
    chunks: list[str],
    *,
    num_threads: int = 40,
    max_depth: int = 3,
) -> str:
    logger.info(
        "Старт структурирования: chunks=%s, depth=%s",
        len(chunks or []),
        len(parent_headings or []),
    )
    parent_headings = _normalize_headings(parent_headings)
    chunks = [c for c in (chunks or []) if c and c.strip()]

    if not chunks:
        logger.info("Пустые чанки — возвращаю только заголовки")
        return "\n\n".join(parent_headings)

    # 1) Базовый случай
    if len(chunks) <= 4 or len(parent_headings) >= max_depth:
        logger.info("Базовый случай — пишу раздел без разбиения")
        return dspy.ChainOfThought(SubsectionSignature)(
            parent_headings=parent_headings,
            content_chunks=chunks,
        ).subsection

    # 2) Выжимки по чанкам
    logger.info("Шаг 2: краткие выжимки по чанкам")
    produce_gist = dspy.Predict(GistSignature)
    gist_inputs = [{"parent_headings": parent_headings, "chunk": c} for c in chunks]
    chunk_gists = _parallel_predict(
        produce_gist, gist_inputs, ["parent_headings", "chunk"], num_threads
    )
    gist_texts = [getattr(g, "gist", "") for g in chunk_gists]

    # 3) Заголовки следующего уровня
    logger.info("Шаг 3: генерация заголовков оглавления")
    headers_result = dspy.ChainOfThought(ContentHeadingsSignature)(
        parent_headings=parent_headings,
        chunk_gists=gist_texts,
    )
    headers = _coerce_str_list(getattr(headers_result, "content_headings", None))
    headers = [h.strip() for h in headers if isinstance(h, str) and h.strip()]

    if not headers:
        logger.info("Не удалось получить заголовки — fallback на прямое суммирование")
        return dspy.ChainOfThought(SubsectionSignature)(
            parent_headings=parent_headings, content_chunks=chunks
        ).subsection

    # Если заголовков слишком мало — детерминированное разбиение по порядку чанков
    if len(headers) < 4 and len(chunks) >= 12:
        logger.info(
            "Слишком мало заголовков (%s) для %s чанков — fallback на детерминированные части",
            len(headers),
            len(chunks),
        )
        desired = min(10, max(4, (len(chunks) + 9) // 10))  # ~10 чанков на часть
        chunk_per_part = (len(chunks) + desired - 1) // desired
        summarized_parts: list[str] = []
        for i in range(desired):
            part_chunks = chunks[i * chunk_per_part : (i + 1) * chunk_per_part]
            if not part_chunks:
                continue
            heading = "#" * (len(parent_headings) + 1) + f" Часть {i+1}"
            part = dspy.ChainOfThought(SubsectionSignature)(
                parent_headings=parent_headings + [heading],
                content_chunks=part_chunks,
            ).subsection
            summarized_parts.append(part)

        return "\n\n".join([parent_headings[-1]] + summarized_parts)

    # 4) Классификация чанков по темам
    logger.info("Шаг 4: классификация чанков по заголовкам")
    classify = dspy.ChainOfThought(ChunkTopicSignature)
    classify_inputs = [
        {"parent_headings": parent_headings, "chunk": c, "content_headings": headers}
        for c in chunks
    ]
    topics = _parallel_predict(
        classify,
        classify_inputs,
        ["parent_headings", "chunk", "content_headings"],
        num_threads,
    )

    # 5) Группировка
    logger.info("Шаг 5: группировка чанков по разделам")
    sections: dict[str, list[str]] = {topic: [] for topic in headers}
    for topic_pred, chunk in zip(topics, chunks):
        topic = getattr(topic_pred, "topic", None)
        if topic not in sections:
            topic = headers[0]
        sections[topic].append(chunk)

    # 6) Рекурсивное суммирование
    logger.info("Шаг 6: рекурсивное суммирование секций")
    prefix = "#" * (len(parent_headings) + 1) + " "
    summarized_sections: list[str] = []
    for topic, section_chunks in sections.items():
        if not section_chunks:
            continue
        summarized_sections.append(
            structure_and_summarize(
                parent_headings + [prefix + topic],
                section_chunks,
                num_threads=num_threads,
                max_depth=max_depth,
            )
        )

    logger.info("Шаг 7: сбор итогового текста")
    return "\n\n".join([parent_headings[-1]] + summarized_sections)


def summarize_text(
    input_text: str,
    *,
    parent_heading: str = "Summary",
    splitter_kwargs: dict[str, Any] | None = None,
    num_threads: int = 40,
    max_depth: int = 3,
    output_path: str = "summary.md",
) -> str:
    splitter = SemanticParallelSplitter()
    splitter_kwargs = splitter_kwargs or {}
    split_result = splitter(input_text=input_text, **splitter_kwargs)
    chunks = getattr(split_result, "segments", []) or []

    parent_headings = _normalize_headings([parent_heading])
    result = structure_and_summarize(
        parent_headings,
        chunks,
        num_threads=num_threads,
        max_depth=max_depth,
    )

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    logger.info("Результат записан в файл: %s", output_path)
    return result
