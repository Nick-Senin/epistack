import csv
import json

file_path = 'datasets/Интеллектуальные коллективы - тренажерка - Sheet1.csv'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if row['Исходный пример']:
                print(f"--- Example {count+1} ---")
                print(f"Link 1 Raw: {row.get('СВЯЗКА 1', '')}")
                print(f"Link 1 Abstraction: {row.get('СВЯЗКА 1 - АБСТРАКЦИЯ', '')}")
                print(f"Link 2 Raw: {row.get('СВЯЗКА 2', '')}")
                count += 1
                if count >= 3:
                    break
except Exception as e:
    print(e)

