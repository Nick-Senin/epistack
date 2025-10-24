# EpiStack

Модульная система реализующая атомарные преобразования со знаниями.  

## Структура проекта

```
epistack/
├── module_extraction/   # Извлечение связок
│   ├── signatures.py    # ExtractRelationsSig
│   ├── module.py        # RelationExtractor
│   └── optimize.py      # Оптимизатор модуля
│
├── module_naming/       # Именование связок
│   ├── signatures.py    # CausalRelationExtractorSignature
│   ├── module.py        # RelationNamer
│   └── optimize.py      # Оптимизатор модуля
│
├── module_abstraction/  # Абстрагирование в A-T-B
│   ├── signatures.py    # AbstractATBSig, CritiqueSig, ReviseSig
│   ├── module.py        # NaiveATBAbstraction, EfficientATBAbstraction
│   ├── metrics.py       # AbstractionMetrics (id:24)
│   └── optimize.py      # Оптимизатор модуля
│
├── module_concretization/  # Конкретизация из A-T-B
│   ├── signatures.py       # ConcretizeFromATBSig, CritiqueSig, ReviseSig
│   ├── module.py           # ConcretizerWithReflection
│   └── metrics.py          # ConcretizationMetrics (id:23)
│
├── utils/               # Вспомогательные функции
│   └── helpers.py       # safe_json_dict, jaccard_like
│
├── config/              # Конфигурация LLM
│   └── llm.py           # configure_llm
│
├── main.py              # Точка входа для запуска
└── optimize_modules.py  # Главный оптимизатор (импортирует из модулей)
```

## Преобразования и их компоненты

### 📊 module_extraction/
- **Сигнатура**: `ExtractRelationsSig`
- **Модуль**: `RelationExtractor`
- **Метрики**: количество извлеченных связок

### 🏷️ module_naming/
- **Сигнатура**: `CausalRelationExtractorSignature`
- **Модуль**: `RelationNamer`

### 🔺 module_abstraction/
- **Сигнатуры**: `AbstractATBSig`, `CritiqueSig`, `ReviseSig`
- **Модули**: `NaiveATBAbstraction`, `EfficientATBAbstraction`
- **Метрики** (id:24):
  - `sufficient_abstraction` - достаточность абстракции
  - `not_over_abstracted` - отсутствие пере-абстрагирования

### 🔻 module_concretization/
- **Сигнатуры**: `ConcretizeFromATBSig`, `CritiqueSig`, `ReviseSig`
- **Модуль**: `ConcretizerWithReflection`
- **Метрики** (id:23):
  - `equivalence_after_concretization` - эквивалентность после конкретизации

## Описание метрик

### Метрики абстрагирования (id:24)
- **sufficient_abstraction**: LLM-as-Judge проверяет, что A-T-B достаточно обобщает связку без потери ключевого смысла
- **not_over_abstracted**: LLM-as-Judge проверяет, что A-T-B не слишком общий

### Метрики конкретизации (id:23)
- **equivalence_after_concretization**: LLM-as-Judge проверяет эквивалентность между исходной связкой и конкретизацией из A-T-B

## Установка

```bash
# Установка в режиме разработки (editable)
pip install -e .

# Проверка импортов
python test_imports.py
```

Подробнее см. [INSTALL.md](INSTALL.md)

## Использование

### Базовый пример

```python
from epistack import configure_llm, RelationNamer

configure_llm()

text = "Исследователь применил метод дистилляции знаний..."
namer = RelationNamer()
result = namer(text=text)
```

### Использование отдельных модулей

```python
# Импорт конкретных модулей
from epistack.module_abstraction import NaiveATBAbstraction, AbstractionMetrics
from epistack.module_naming import RelationNamer
from epistack.module_concretization import ConcretizerWithReflection

# Или через основной пакет
from epistack import (
    RelationNamer,
    NaiveATBAbstraction,
    ConcretizerWithReflection
)
```

## 📊 Датасет для оптимизации

**Публичный датасет**: [Nick-Sen/epistack-optimization](https://huggingface.co/datasets/Nick-Sen/epistack-optimization)

### Оптимизация через CLI

```bash
# Все модули
python optimize_modules.py Nick-Sen all

# Отдельные модули
python optimize_modules.py Nick-Sen naming
python optimize_modules.py Nick-Sen extraction
python optimize_modules.py Nick-Sen abstraction
```

### Оптимизация программно

```python
# Импорт оптимизаторов из модулей
from epistack.module_naming import optimize as optimize_naming
from epistack.module_extraction import optimize as optimize_extraction
from epistack.module_abstraction import optimize as optimize_abstraction

# Оптимизация конкретного модуля
optimized_namer = optimize_naming('Nick-Sen')
optimized_extractor = optimize_extraction('Nick-Sen')
optimized_abstractor = optimize_abstraction('Nick-Sen')
```

Подробнее: [DATASET_INFO.md](DATASET_INFO.md)

## Зависимости

- dspy-ai>=2.4.0
- datasets>=2.14.0
- huggingface_hub>=0.19.0
- pandas>=2.0.0
- python-dotenv>=1.0.0
