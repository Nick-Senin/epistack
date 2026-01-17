"""
Вспомогательные утилиты для модуля семантического разделения.
"""
import re
import difflib
import logging

logger = logging.getLogger(__name__)

class TextMatcher:
    """Логика поиска и сопоставления подстрок в тексте."""

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Нормализует текст для сопоставления:
        - приводит разные тире к обычному дефису
        - схлопывает пробелы/переводы строк
        - убирает лишние пробелы вокруг пунктуации не трогая саму пунктуацию
        """
        if not text:
            return ""
        # унификация тире/минусов
        text = text.replace("—", "-").replace("–", "-").replace("−", "-")
        # унификация кавычек (минимально)
        text = text.replace("“", '"').replace("”", '"').replace("„", '"').replace("’", "'").replace("‘", "'")
        # схлопываем whitespace
        text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
        return text

    @staticmethod
    def _fallback_structural(full_text: str, preferred: int | None = None) -> int:
        """
        Fallback #1: структурный разрез по двойному переводу строки / одинарному переводу строки
        около preferred (если задан) иначе около середины.
        """
        n = len(full_text)
        if n < 2:
            return 0
        target = preferred if preferred is not None else n // 2
        target = max(1, min(n - 1, target))

        # кандидаты: границы абзацев/строк
        candidates: list[int] = []
        for m in re.finditer(r"\n{2,}", full_text):
            candidates.append(m.end())
        for m in re.finditer(r"\n", full_text):
            candidates.append(m.end())

        # фильтруем слишком ранние/поздние точки
        min_i = max(1, int(n * 0.15))
        max_i = min(n - 1, int(n * 0.85))
        candidates = [c for c in candidates if min_i <= c <= max_i]
        if not candidates:
            return 0

        best = min(candidates, key=lambda c: abs(c - target))
        logger.info(f"[FALLBACK_STRUCTURAL] split_index={best}")
        return best

    @staticmethod
    def _fallback_sentence(full_text: str, preferred: int | None = None) -> int:
        """
        Fallback #2: разрез по концу предложения (., !, ?, …) около preferred/середины.
        """
        n = len(full_text)
        if n < 2:
            return 0
        target = preferred if preferred is not None else n // 2
        target = max(1, min(n - 1, target))

        # находим окончания предложений, учитывая пробел/перевод строки после
        candidates: list[int] = []
        for m in re.finditer(r"[.!?…](?:\s+|$)", full_text):
            candidates.append(m.end())

        min_i = max(1, int(n * 0.15))
        max_i = min(n - 1, int(n * 0.85))
        candidates = [c for c in candidates if min_i <= c <= max_i]
        if not candidates:
            return 0

        best = min(candidates, key=lambda c: abs(c - target))
        logger.info(f"[FALLBACK_SENTENCE] split_index={best}")
        return best

    @staticmethod
    def _fallback_hard(full_text: str, preferred: int | None = None) -> int:
        """
        Fallback #3: жёсткий разрез по ближайшему пробелу около preferred/середины.
        """
        n = len(full_text)
        if n < 2:
            return 0
        target = preferred if preferred is not None else n // 2
        target = max(1, min(n - 1, target))

        # ищем ближайший пробел в окне
        window = max(20, int(n * 0.05))
        lo = max(1, target - window)
        hi = min(n - 1, target + window)
        segment = full_text[lo:hi]
        # ближайший пробел/таб
        space_positions = [m.start() for m in re.finditer(r"[ \t]", segment)]
        if not space_positions:
            split_index = target
        else:
            best_local = min(space_positions, key=lambda p: abs((lo + p) - target))
            split_index = lo + best_local
        split_index = max(1, min(n - 1, split_index))
        logger.info(f"[FALLBACK_HARD] split_index={split_index}")
        return split_index

    @staticmethod
    def find_split_index(full_text: str, candidate_block: str) -> int:
        """
        Находит индекс разделения текста на основе предложенного блока.
        
        Args:
            full_text: Исходный текст
            candidate_block: Текст первого блока, предложенный LLM
            
        Returns:
            int: Индекс конца первого блока в исходном тексте (0 если не найдено)
        """
        if not candidate_block or not full_text:
            return 0
            
        candidate_block = candidate_block.strip()
        if not candidate_block:
            return 0

        # Быстрый путь для коротких фрагментов: если текст маленький, лучше выбрать границу по структуре/предложению.
        if len(full_text) < 200:
            preferred = min(len(full_text) - 1, max(1, len(candidate_block)))
            for fb in (TextMatcher._fallback_structural, TextMatcher._fallback_sentence, TextMatcher._fallback_hard):
                idx = fb(full_text, preferred=preferred)
                if 0 < idx < len(full_text):
                    return idx
        
        # 1. Точное совпадение с начала (самый надежный вариант)
        if full_text.startswith(candidate_block):
            return len(candidate_block)
            
        # 2. Поиск подстроки (если LLM пропустила начало)
        idx = full_text.find(candidate_block)
        if idx != -1:
            return idx + len(candidate_block)

        # 3. Нормализуем и повторяем точный/подстрочный матч на нормализованном тексте.
        # Это часто лечит расхождения по пробелам/тире/кавычкам.
        full_norm = TextMatcher._normalize(full_text)
        cand_norm = TextMatcher._normalize(candidate_block)
        if full_norm and cand_norm:
            if full_norm.startswith(cand_norm):
                # приблизительно возвращаем длину оригинального блока; точная проекция сложна,
                # но для нашей задачи достаточно корректной точки внутри full_text.
                approx = min(len(full_text) - 1, len(candidate_block))
                logger.info(f"[MATCH_NORM_PREFIX] approx_split_index={approx}")
                return approx
            idx_norm = full_norm.find(cand_norm)
            if idx_norm != -1:
                # приближённая проекция: берем долю позиции в нормализованном тексте и переносим на исходный
                frac = (idx_norm + len(cand_norm)) / max(len(full_norm), 1)
                approx = int(frac * len(full_text))
                approx = max(1, min(len(full_text) - 1, approx))
                logger.info(f"[MATCH_NORM_SUBSTR] approx_split_index={approx}")
                return approx

        # 4. Символьный fuzzy-поиск (SequenceMatcher) по ПРЕФИКСУ candidate_block.
        # Ключевой фикс для проблемных чанков: LLM часто «продолжает» дальше чанка,
        # из-за чего full_text содержит только начало candidate_block.
        prefix_len = min(800, len(candidate_block))
        cand_prefix = candidate_block[:prefix_len]
        min_cover_full = 0.80      # насколько большая часть full_text покрыта совпадением
        min_match_len = 180
        matcher = difflib.SequenceMatcher(None, full_text, cand_prefix)
        match = matcher.find_longest_match(0, len(full_text), 0, len(cand_prefix))
        if match.size >= min_match_len:
            cover_full = match.size / max(len(full_text), 1)
            cover_cand = match.size / max(len(cand_prefix), 1)
            # если совпадение почти с начала чанка — считаем это корректным якорем
            if match.a <= 20 and (cover_cand >= 0.70 or cover_full >= min_cover_full):
                split_index = match.a + match.size
                split_index = max(1, min(len(full_text) - 1, split_index))
                logger.info(f"[MATCH_FUZZY_PREFIX] split_index={split_index} (cover_full={cover_full:.2f}, cover_cand={cover_cand:.2f})")
                return split_index

        # 5. Поиск по "хвосту" блока (последние 10 слов)
        # Это помогает, если LLM ошиблась в середине длинного текста, 
        # но верно определила точку разделения в конце.
        words = candidate_block.split()
        if len(words) >= 10:
            tail_words = words[-10:]
            tail_pattern = r"\s+".join([re.escape(w) for w in tail_words])
            try:
                # Ищем хвост в тексте. Если он встречается один раз - это наша точка.
                matches = list(re.finditer(tail_pattern, full_text))
                if len(matches) == 1:
                    logger.info(f"[MATCH_TAIL] Точка разделения найдена по хвосту блока: {matches[0].end()}")
                    return matches[0].end()
                elif len(matches) > 1:
                    # Если несколько совпадений, берем то, что ближе всего к длине candidate_block
                    best_match = min(matches, key=lambda m: abs(m.end() - len(candidate_block)))
                    logger.info(f"[MATCH_TAIL_NEAR] Найдено несколько совпадений хвоста, выбрано ближайшее: {best_match.end()}")
                    return best_match.end()
            except re.error:
                pass

        # 6. "Fuzzy" поиск по всему блоку (игнорируя различия в пробелах)
        # Разбиваем блок на слова, экранируя спецсимволы
        words_escaped = [re.escape(w) for w in words]
        if not words_escaped:
            return 0
            
        pattern_str = r'\s*'.join(words_escaped)
        try:
            match = re.search(pattern_str, full_text)
            if match:
                logger.info(f"[MATCH_FUZZY] Блок найден через fuzzy-поиск, split_index={match.end()}")
                return match.end()
        except re.error as e:
            logger.error(f"[REGEX_ERROR] Ошибка в регулярном выражении при fuzzy поиске: {e}")

        # 7. Если сопоставление не удалось — fallback стратегии.
        # preferred: ожидаем, что first_block заканчивается примерно около len(candidate_block),
        # но ограничиваемся длиной full_text.
        preferred = min(len(full_text) - 1, max(1, len(candidate_block)))
        for fb in (TextMatcher._fallback_structural, TextMatcher._fallback_sentence, TextMatcher._fallback_hard):
            idx = fb(full_text, preferred=preferred)
            if 0 < idx < len(full_text):
                return idx

        return 0
