import csv
import json

source_csv = 'datasets/Интеллектуальные коллективы - тренажерка - Sheet1.csv'
target_json = 'datasets/Интеллектуальные коллективы - тренажерка - Sheet1.json'

def clean_text(text):
    if not text:
        return ""
    return text.strip()

def process_csv():
    examples = []
    
    try:
        with open(source_csv, 'r', encoding='utf-8') as f:
            # Используем DictReader для удобства доступа к именованным колонкам
            reader = csv.DictReader(f)
            all_rows = list(reader)
            
            total_rows = len(all_rows)
            i = 0
            
            while i < total_rows:
                current_row = all_rows[i]
                
                # Если нет исходного примера, это может быть вторая/третья строка триплета, 
                # которую мы уже обработали, или пустая строка. Пропускаем.
                original_text = clean_text(current_row.get('Исходный пример', ''))
                if not original_text:
                    i += 1
                    continue
                
                author = clean_text(current_row.get('АВТОР', ''))
                name = clean_text(current_row.get('Название примера', ''))
                
                triplets = []
                
                # Предполагаем до 10 связок
                for k in range(1, 11):
                    link_key = f'СВЯЗКА {k}'
                    metric_key = f'СВЯЗКА {k} - МЕТРИКИ'
                    
                    # Данные для Initial State (из текущей строки i)
                    initial_state = ""
                    if link_key in current_row:
                        initial_state = clean_text(current_row[link_key])
                        
                    # Данные для Transformation (из строки i+1)
                    transformation = ""
                    if i + 1 < total_rows:
                        next_row = all_rows[i + 1]
                        # Проверяем, что это продолжение (нет нового примера)
                        # Но новый пример может быть без текста? Вряд ли.
                        # Будем считать что следующие 2 строки всегда относятся к этому примеру.
                        if link_key in next_row:
                            transformation = clean_text(next_row[link_key])
                            
                    # Данные для Result (из строки i+2)
                    result = ""
                    if i + 2 < total_rows:
                        next_next_row = all_rows[i + 2]
                        if link_key in next_next_row:
                            result = clean_text(next_next_row[link_key])
                    
                    # Воспроизводимость берем из первой строки (или может быть в любой?)
                    # В debug_triplet метрика была в первой строке (8), а во второй был текст.
                    # Будем брать из первой строки.
                    reproducibility = ""
                    if metric_key in current_row:
                        reproducibility = clean_text(current_row[metric_key])
                    
                    # Если есть хотя бы одно непустое поле состояния
                    if initial_state or transformation or result:
                        triplets.append({
                            "начальное состояние": initial_state,
                            "преобразование": transformation,
                            "результат": result,
                            "воспроизводимость": reproducibility
                        })
                
                example_obj = {
                    "Автор примера": author,
                    "Исходный текст примера": original_text,
                    "Название примера": name,
                    "triples": triplets
                }
                
                examples.append(example_obj)
                
                # Мы обработали этот пример. Нужно ли сдвигать индекс i?
                # Цикл while идет по одной строке. Если мы использовали i+1 и i+2, 
                # то на следующей итерации (i+1) мы увидим пустой 'Исходный пример' и пропустим её.
                # Так что просто i += 1 достаточно.
                i += 1
        
        # Запись в JSON
        with open(target_json, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed {len(examples)} examples.")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_csv()
