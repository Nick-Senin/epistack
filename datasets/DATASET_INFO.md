# Epistack Optimization Dataset

## 📊 Публичный датасет для оптимизации модулей

**URL:** https://huggingface.co/datasets/Nick-Sen/epistack-optimization

### Статистика
- **Train:** 2 примера
- **Test:** 1 пример
- **Доступ:** Публичный (токен не требуется)

### Структура примера

```python
{
    'source_text': str,           # Исходный код
    'title': str,                 # Название модуля
    'extracted_chains': [         # Выделенные связки
        {
            'initial_state': str,
            'transformation': str,
            'result': str
        }
    ],
    'abstracted_chains': [        # Абстрагированные связки
        {
            'initial_state': str,
            'transformation': str,
            'result': str
        }
    ]
}
```

### Быстрое использование

```python
from datasets import load_dataset

# Загрузка (публичный датасет)
ds = load_dataset('Nick-Sen/epistack-optimization')

# Train примеры
train = ds['train']
print(f"Примеров: {len(train)}")
print(f"Первый: {train[0]['title']}")

# В pandas
df = train.to_pandas()

# Для DSPy оптимизации
from epistack_data import for_naming_module
trainset = for_naming_module('Nick-Sen')
```

### Примеры в датасете

#### 1. User Data Processing Pipeline
```python
def process_user_data(raw_data):
    cleaned = remove_nulls(raw_data)
    validated = check_format(cleaned)
    return save_to_db(validated)
```

**Связки:** raw_data → remove_nulls → cleaned → check_format → validated → save_to_db

#### 2. Email Delivery System
```python
class EmailSender:
    def send(self, recipients, message):
        formatted = self.format_html(message)
        attachments = self.prepare_files()
        return self.smtp_send(recipients, formatted, attachments)
```

**Связки:** message → format_html → formatted, files → prepare_files → attachments

#### 3. API Data Fetcher
```python
async def fetch_api_data(url, params):
    response = await http_get(url, params)
    parsed = json.loads(response.text)
    return transform_schema(parsed)
```

**Связки:** url, params → http_get → response → json.loads → parsed → transform_schema

### Добавление новых примеров

```bash
# Через скрипт
python epistack_data/edit_dataset.py

# Или программно
from epistack_data.edit_dataset import add_example

add_example(
    source_text='your code here',
    title='Module Name',
    extracted_chains=[{...}],
    abstracted_chains=[{...}],
    hf_username='Nick-Sen'
)
```

### Оптимизация модулей

Каждый модуль теперь имеет свой `optimize.py` файл.

#### Через CLI

```bash
# Оптимизация всех модулей
python optimize_modules.py Nick-Sen all

# Только naming
python optimize_modules.py Nick-Sen naming

# Только extraction
python optimize_modules.py Nick-Sen extraction

# Только abstraction
python optimize_modules.py Nick-Sen abstraction
```

#### Программно (из каждого модуля)

```python
# Импорт оптимизаторов напрямую из модулей
from epistack.module_naming import optimize as optimize_naming
from epistack.module_extraction import optimize as optimize_extraction
from epistack.module_abstraction import optimize as optimize_abstraction

# Оптимизация конкретного модуля
optimized_namer = optimize_naming('Nick-Sen')
optimized_extractor = optimize_extraction('Nick-Sen')
optimized_abstractor = optimize_abstraction('Nick-Sen')
```

---

Создано: 2025-10-24  
Автор: Nick-Sen  
Лицензия: MIT

