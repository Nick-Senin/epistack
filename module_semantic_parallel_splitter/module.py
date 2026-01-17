"""
Модуль семантической сегментации текста на параллельные блоки
"""

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import dspy
from requests.exceptions import RequestException

from .signatures import SemanticHalverSignature
from .utils import TextMatcher

logger = logging.getLogger(__name__)


@dataclass
class ChunkSplit:
    """Результат разделения одного чанка на две части."""

    first_half: str
    second_half: str
    split_index: int
    failure_reason: str | None = None

    @property
    def is_split(self) -> bool:
        return bool(self.second_half)


@dataclass
class ProcessingStats:
    """Статистика обработки (в духе ноутбука text_splitter.ipynb)."""

    initial_chunks: int
    successfully_split: int
    failed_splits: int
    final_chunks: int
    avg_chunk_size: float
    max_chunk_size: int
    min_chunk_size: int


class SemanticHalver(dspy.Module):
    """DSPy модуль для смыслового разделения текста на две части."""

    def __init__(self):
        super().__init__()
        # Predict обеспечивает более высокую скорость работы
        self.halver = dspy.Predict(SemanticHalverSignature)
        self.matcher = TextMatcher()

    def forward(self, text: str) -> tuple[dspy.Prediction, Any]:
        """
        Выполняет семантическое разделение текста на два блока.

        Args:
            text: Исходный текст для разделения

        Returns:
            dspy.Prediction с полями:
                - first_block: первый смысловой блок
                - split_index: индекс разбиения (0 если не удалось)
                - error: описание ошибки (если была)
        """
        # 1. Предварительная валидация
        if not text or not isinstance(text, str):
            return self._handle_error("EMPTY_INPUT", "текст пустой или не является строкой", logging.WARNING)

        text = text.strip()
        if len(text) < 50:
            logger.info(f"[TOO_SHORT] текст слишком короткий (len={len(text)}), возвращаем без изменений")
            return (
                dspy.Prediction(
                    first_block=text,
                    split_index=len(text),
                    error=None,
                ),
                None,
            )

        # 2. Вызов LLM
        try:
            prediction = self.halver(text=text)
        except (RequestException, TimeoutError, dspy.utils.DSPyTimeoutError) as e:
            return self._handle_error("NETWORK_OR_TIMEOUT", str(e))
        except Exception as e:
            return self._handle_error("UNEXPECTED", f"{type(e).__name__}: {e}")

        # 3. Извлечение и нормализация результата
        first_block = getattr(prediction, "first_block", "")
        if first_block is None:
            first_block = ""
        elif not isinstance(first_block, str):
            first_block = str(first_block)

        first_block = first_block.strip()

        # 4. Сопоставление с исходным текстом
        split_index = 0
        if not first_block:
            logger.warning("[EMPTY_BLOCK] LLM вернул пустой first_block")
        else:
            # Даже если LLM вернул слишком большой блок (иногда он «продолжает» дальше границ чанка),
            # пытаемся сопоставить и/или применить fallback-стратегии в TextMatcher.
            if len(first_block) >= len(text):
                logger.warning(
                    f"[BLOCK_TOO_LARGE] first_block (len={len(first_block)}) >= исходного текста; пробуем matcher/fallback"
                )
            split_index = self.matcher.find_split_index(text, first_block)

        # 5. Финальная валидация индекса
        if split_index >= len(text):
            logger.warning(f"[SPLIT_OVERFLOW] split_index={split_index} >= len(text), сбрасываем в 0")
            split_index = 0

        if split_index > 0:
            logger.info(f"[SUCCESS] Текст разделён: split_index={split_index}")
        else:
            logger.debug("[SPLIT_ZERO] Не удалось найти точку разделения")

        prediction.split_index = split_index
        # dspy.Parallel ожидает пару (prediction, trace)
        return prediction, None

    def _handle_error(self, code: str, message: str, level: int = logging.ERROR) -> tuple[dspy.Prediction, Any]:
        """Вспомогательный метод для единообразной обработки ошибок."""
        full_msg = f"[{code}] {message}"
        logger.log(level, full_msg)
        return (
            dspy.Prediction(
                first_block="",
                split_index=0,
                error=full_msg,
            ),
            None,
        )


class SemanticParallelSplitter(dspy.Module):
    """
    Оркестратор семантической сегментации (pipeline из text_splitter.ipynb).

    Алгоритм:
    1) Первичный pre-split по предложениям до max_chunk_size
    2) Параллельный SemanticHalver для каждого чанка
    3) Склейка second(i)+first(i+1)
    4) Повторное деление чанков > max_chunk_size (с ограничением итераций)
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。\u3002\uFF01\uFF1F])\s+(?=[A-ZА-ЯЁa-zа-яё])")

    def __init__(self, halver: SemanticHalver | None = None):
        super().__init__()
        self.halver = halver or SemanticHalver()

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            try:
                return int(v)
            except ValueError:
                return None
        try:
            return int(value)
        except Exception:
            return None

    def _split_into_sentence_chunks(self, text: str, max_chunk_size: int) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if max_chunk_size <= 0 or len(text) <= max_chunk_size:
            return [text]

        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
        if not sentences:
            return [text]

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= max_chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            # Если предложение само по себе больше max_chunk_size, оставляем как есть
            current = sentence

        if current:
            chunks.append(current.strip())

        return chunks or [text]

    def _split_chunk_semantically(self, chunk: str, min_chunk_len: int) -> ChunkSplit:
        if not chunk:
            return ChunkSplit(first_half="", second_half="", split_index=0, failure_reason="Пустой чанк")
        if min_chunk_len > 0 and len(chunk) < min_chunk_len:
            return ChunkSplit(
                first_half=chunk,
                second_half="",
                split_index=0,
                failure_reason=f"Чанк слишком короткий (<{min_chunk_len} символов)",
            )

        try:
            pred, _trace = self.halver(text=chunk)
        except Exception as e:
            return ChunkSplit(
                first_half=chunk,
                second_half="",
                split_index=0,
                failure_reason=f"Ошибка вызова halver: {type(e).__name__}: {e}",
            )

        split_idx = getattr(pred, "split_index", 0)
        try:
            split_idx = int(split_idx)
        except Exception:
            split_idx = 0

        if not (0 < split_idx < len(chunk)):
            return ChunkSplit(
                first_half=chunk,
                second_half="",
                split_index=0,
                failure_reason=f"Некорректный split_index={split_idx}",
            )

        first_half = chunk[:split_idx].strip()
        second_half = chunk[split_idx:].strip()

        if not second_half:
            return ChunkSplit(
                first_half=chunk,
                second_half="",
                split_index=0,
                failure_reason=f"Вторая часть пуста после split_index={split_idx}",
            )

        return ChunkSplit(first_half=first_half, second_half=second_half, split_index=split_idx)

    def _split_chunks_parallel(self, chunks: list[str], num_threads: int, min_chunk_len: int) -> list[ChunkSplit]:
        if not chunks:
            return []

        examples = [dspy.Example(text=c).with_inputs("text") for c in chunks]
        exec_pairs = [(self.halver, ex) for ex in examples]

        parallel = dspy.Parallel(num_threads=max(1, int(num_threads)), max_errors=len(chunks), provide_traceback=True)
        try:
            results = parallel(exec_pairs)
        except Exception as e:
            details = f"{type(e).__name__}: {e}"
            out: list[ChunkSplit] = []
            for c in chunks:
                r = self._split_chunk_semantically(c, min_chunk_len=min_chunk_len)
                if not r.is_split:
                    r.failure_reason = (r.failure_reason or "Не удалось разделить") + f" | parallel failed: {details}"
                out.append(r)
            return out

        preds: list[Any] = []
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                preds.append(item[0])
            else:
                preds.append(item)

        splits: list[ChunkSplit] = []
        for chunk, pred in zip(chunks, preds):
            if min_chunk_len > 0 and len(chunk) < min_chunk_len:
                splits.append(
                    ChunkSplit(
                        first_half=chunk,
                        second_half="",
                        split_index=0,
                        failure_reason=f"Чанк слишком короткий (<{min_chunk_len} символов)",
                    )
                )
                continue

            split_idx = getattr(pred, "split_index", 0)
            try:
                split_idx = int(split_idx)
            except Exception:
                split_idx = 0

            if 0 < split_idx < len(chunk):
                first_half = chunk[:split_idx].strip()
                second_half = chunk[split_idx:].strip()
                if second_half:
                    splits.append(ChunkSplit(first_half=first_half, second_half=second_half, split_index=split_idx))
                    continue

            splits.append(
                ChunkSplit(
                    first_half=chunk,
                    second_half="",
                    split_index=0,
                    failure_reason=f"Не удалось разделить параллельно: split_index={split_idx}",
                )
            )

        return splits

    @staticmethod
    def _merge_adjacent_halves(splits: list[ChunkSplit]) -> list[str]:
        if not splits:
            return []

        merged: list[str] = []

        if splits[0].first_half:
            merged.append(splits[0].first_half)

        for i in range(len(splits) - 1):
            current_second = splits[i].second_half
            next_first = splits[i + 1].first_half

            if current_second and next_first:
                merged.append(f"{current_second} {next_first}".strip())
            elif current_second:
                merged.append(current_second)
            elif next_first:
                merged.append(next_first)

        if splits[-1].second_half and not splits[-1].second_half.isspace():
            merged.append(splits[-1].second_half)

        return [c for c in merged if c and c.strip()]

    def _resplit_large_chunks(
        self,
        chunks: list[str],
        max_chunk_size: int,
        min_chunk_len: int,
        max_iters: int,
    ) -> list[str]:
        if not chunks:
            return []
        if max_chunk_size <= 0:
            return chunks

        out = chunks[:]
        for _ in range(max(0, int(max_iters))):
            changed = False
            next_out: list[str] = []

            for chunk in out:
                if len(chunk) <= max_chunk_size:
                    next_out.append(chunk)
                    continue

                split_result = self._split_chunk_semantically(chunk, min_chunk_len=min_chunk_len)
                if split_result.is_split:
                    # прогресс: обе части должны быть короче исходного чанка
                    if len(split_result.first_half) < len(chunk) and len(split_result.second_half) < len(chunk):
                        next_out.append(split_result.first_half)
                        next_out.append(split_result.second_half)
                        changed = True
                        continue

                next_out.append(chunk)

            out = next_out
            if not changed:
                break

        return [c for c in out if c and c.strip()]

    @staticmethod
    def _apply_constraints(segments: list[str], max_segments: int | None, min_segment_length: int | None) -> list[str]:
        cleaned = [s.strip() for s in (segments or []) if s and s.strip()]
        if not cleaned:
            return []

        # min_segment_length: сливаем короткие сегменты со следующими
        if min_segment_length is not None and min_segment_length > 0:
            merged: list[str] = []
            buf = ""
            for s in cleaned:
                if not buf:
                    buf = s
                    continue
                if len(buf) < min_segment_length:
                    buf = f"{buf} {s}".strip()
                else:
                    merged.append(buf)
                    buf = s
            if buf:
                merged.append(buf)
            cleaned = merged

        # max_segments: сливаем самые короткие соседние пары
        if max_segments is not None and max_segments > 0:
            max_segments = int(max_segments)
            while len(cleaned) > max_segments and len(cleaned) >= 2:
                best_i = 0
                best_len = len(cleaned[0]) + len(cleaned[1])
                for i in range(len(cleaned) - 1):
                    pair_len = len(cleaned[i]) + len(cleaned[i + 1])
                    if pair_len < best_len:
                        best_len = pair_len
                        best_i = i
                merged_pair = f"{cleaned[best_i]} {cleaned[best_i + 1]}".strip()
                cleaned = cleaned[:best_i] + [merged_pair] + cleaned[best_i + 2 :]

        return cleaned

    def forward(
        self,
        input_text: str | None = None,
        max_segments: int | str | None = None,
        min_segment_length: int | str | None = None,
        max_chunk_size: int | str = 3000,
        num_threads: int | str = 4,
        min_chunk_len: int | str = 100,
        max_resplit_iters: int | str = 3,
    ) -> dspy.Prediction:
        text = (input_text or "").strip()

        max_segments_i = self._coerce_int(max_segments)
        min_segment_length_i = self._coerce_int(min_segment_length)
        max_chunk_size_i = self._coerce_int(max_chunk_size) or 3000
        num_threads_i = self._coerce_int(num_threads) or 4
        min_chunk_len_i = self._coerce_int(min_chunk_len) or 100
        max_resplit_iters_i = self._coerce_int(max_resplit_iters) or 3

        if not text:
            empty_stats = ProcessingStats(
                initial_chunks=0,
                successfully_split=0,
                failed_splits=0,
                final_chunks=0,
                avg_chunk_size=0.0,
                max_chunk_size=0,
                min_chunk_size=0,
            )
            return dspy.Prediction(segments=[], stats=asdict(empty_stats))

        # 1) pre-split по предложениям
        initial_chunks = self._split_into_sentence_chunks(text, max_chunk_size=max_chunk_size_i)

        # 2) параллельный halving
        splits = self._split_chunks_parallel(
            initial_chunks,
            num_threads=num_threads_i,
            min_chunk_len=min_chunk_len_i,
        )
        successfully_split = sum(1 for s in splits if s.is_split)

        # 3) merge halves
        merged_chunks = self._merge_adjacent_halves(splits)

        # 4) resplit oversized
        final_chunks = self._resplit_large_chunks(
            merged_chunks,
            max_chunk_size=max_chunk_size_i,
            min_chunk_len=min_chunk_len_i,
            max_iters=max_resplit_iters_i,
        )

        # 5) ограничения
        final_chunks = self._apply_constraints(
            final_chunks,
            max_segments=max_segments_i,
            min_segment_length=min_segment_length_i,
        )

        sizes = [len(c) for c in final_chunks]
        stats = ProcessingStats(
            initial_chunks=len(initial_chunks),
            successfully_split=successfully_split,
            failed_splits=len(splits) - successfully_split,
            final_chunks=len(final_chunks),
            avg_chunk_size=(sum(sizes) / len(sizes)) if sizes else 0.0,
            max_chunk_size=max(sizes) if sizes else 0,
            min_chunk_size=min(sizes) if sizes else 0,
        )

        return dspy.Prediction(segments=final_chunks, stats=asdict(stats))
