# Epistack Optimization Dataset

Датасет для оптимизации модулей epistack с помощью DSPy.

## Структура датасета

Каждый пример содержит:

- **source_text**: исходный код для анализа
- **title**: название модуля/функциональности
- **extracted_chains**: выделенные связки (initial_state → transformation → result)
- **abstracted_chains**: абстрагированные связки

## Быстрый старт

### 1. Создание датасета

```bash
# Установи зависимости
pip install -r requirements.txt

# Настрой .env файл с токеном HF_TOKEN и HF_USERNAME

# Создай датасет
python epistack_data/create_hf_dataset.py
```

Токен получи здесь: https://huggingface.co/settings/tokens

### 2. Просмотр на Hugging Face

После загрузки: `https://huggingface.co/datasets/your_username/epistack-optimization`

### 3. Добавление примеров

```python
from epistack_data.edit_dataset import add_example

add_example(
    source_text='...',
    title='Module Name',
    extracted_chains=[...],
    abstracted_chains=[...],
    hf_username='Nick-Sen'
)
```

### 4. Использование для оптимизации

```python
from epistack_data import for_naming_module
import dspy

# Загрузка (датасет публичный, токен не нужен)
trainset = for_naming_module('Nick-Sen')

# Оптимизация
optimizer = dspy.BootstrapFewShot(metric=your_metric)
optimized_module = optimizer.compile(module, trainset=trainset)
```

## Доступные функции

### `create_hf_dataset.py`
- Создание начального датасета с примерами
- Загрузка на Hugging Face

### `edit_dataset.py`
- `add_example()` - добавить новый пример
- `view_dataset()` - просмотр примеров

### `use_dataset.py`
- `for_extraction_module()` - для module_extraction
- `for_abstraction_module()` - для module_abstraction
- `for_naming_module()` - для module_naming
- `for_full_pipeline()` - для всего пайплайна

## Структура файлов

```
epistack_data/
├── README.md                      # Это файл
├── create_hf_dataset.py          # Создание датасета
├── edit_dataset.py               # Редактирование
├── use_dataset.py                # Использование
└── epistack_optimization_local/  # Локальная копия (если нет HF_TOKEN)
```

## Приватность

По умолчанию датасет **приватный**. Для публичного:

```python
create_and_upload_dataset(
    hf_username='your_username',
    private=False  # Публичный
)
```

## Пример структуры данных

```python
{
    'source_text': 'def process_data(x):\n    y = clean(x)\n    return save(y)',
    'title': 'Data Processor',
    'extracted_chains': [
        {
            'initial_state': 'x',
            'transformation': 'clean',
            'result': 'y'
        },
        {
            'initial_state': 'y',
            'transformation': 'save',
            'result': 'saved_data'
        }
    ],
    'abstracted_chains': [
        {
            'initial_state': 'raw_input',
            'transformation': 'sanitize',
            'result': 'clean_input'
        },
        {
            'initial_state': 'clean_input',
            'transformation': 'persist',
            'result': 'persisted_output'
        }
    ]
}
```

