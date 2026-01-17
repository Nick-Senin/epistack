import re

def robust_find_split_index(text: str, first_block: str) -> int:
    """
    Ищет окончание first_block в text, игнорируя различия в пробельных символах.
    Возвращает индекс разрыва или 0, если совпадение не найдено.
    """
    if not text or not first_block:
        return 0
        
    # Разбиваем блок на слова, экранируя спецсимволы regex
    words = [re.escape(w) for w in first_block.split()]
    
    if not words:
        return 0
        
    # Создаем паттерн: слова, разделенные любыми пробельными символами (включая \n)
    # \s* означает 0 или более пробелов.
    pattern_str = r'\s*'.join(words)
    
    # Компилируем regex
    # Ищем первое вхождение
    match = re.search(pattern_str, text)
    
    if match:
        # Возвращаем конец совпадения
        return match.end()
        
    return 0

# Тест на данных из chunks.json
if __name__ == "__main__":
    # Имитация проблемы: в тексте \n, в блоке пробел
    real_text = "We're going to\ntalk through stuff."
    llm_block = "We're going to talk through stuff."
    
    print(f"Text: {repr(real_text)}")
    print(f"Block: {repr(llm_block)}")
    
    strict_idx = real_text.find(llm_block)
    print(f"Strict find index: {strict_idx}") # Ожидается -1
    
    fuzzy_idx = robust_find_split_index(real_text, llm_block)
    print(f"Fuzzy find index: {fuzzy_idx}")   # Ожидается > 0
    print(f"Result match: {repr(real_text[:fuzzy_idx])}")
