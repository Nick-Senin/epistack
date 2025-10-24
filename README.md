# EpiStack

Модульная система для извлечения, абстрагирования и конкретизации связок с метриками качества.

## Структура проекта

```
epistack/
├── extraction/          # Извлечение связок
│   ├── signatures.py    # ExtractRelationsSig
│   └── module.py        # RelationExtractor
│
├── naming/              # Именование связок
│   ├── signatures.py    # NameRelationSig
│   └── module.py        # RelationNamer
│
├── abstraction/         # Абстрагирование в A-T-B
│   ├── signatures.py    # AbstractATBSig, CritiqueSig, ReviseSig
│   ├── module.py        # NaiveATBAbstraction, EfficientATBAbstraction
│   └── metrics.py       # AbstractionMetrics (id:24)
│
├── concretization/      # Конкретизация из A-T-B
│   ├── signatures.py    # ConcretizeFromATBSig, CritiqueSig, ReviseSig
│   ├── module.py        # ConcretizerWithReflection
│   └── metrics.py       # ConcretizationMetrics (id:23)
│
├── evaluation/          # Общие метрики оценки
│   └── metrics.py       # StabilityMetrics (id:22)
│
├── pipeline/            # Полный пайплайн и оптимизация
│   ├── epistack.py      # EpiStack (композитный модуль)
│   └── optimizer.py     # optimize_with_gepa
│
├── utils/               # Вспомогательные функции
│   └── helpers.py       # safe_json_dict, jaccard_like
│
├── config/              # Конфигурация LLM
│   └── llm.py           # configure_llm
│
└── main.py              # Точка входа для запуска
```

## Преобразования и их компоненты

### 📊 extraction/
- **Сигнатура**: `ExtractRelationsSig`
- **Модуль**: `RelationExtractor`
- **Метрики**: количество извлеченных связок

### 🏷️ naming/
- **Сигнатура**: `NameRelationSig`
- **Модуль**: `RelationNamer`

### 🔺 abstraction/
- **Сигнатуры**: `AbstractATBSig`, `CritiqueSig`, `ReviseSig`
- **Модули**: `NaiveATBAbstraction`, `EfficientATBAbstraction`
- **Метрики** (id:24):
  - `sufficient_abstraction` - достаточность абстракции
  - `not_over_abstracted` - отсутствие пере-абстрагирования

### 🔻 concretization/
- **Сигнатуры**: `ConcretizeFromATBSig`, `CritiqueSig`, `ReviseSig`
- **Модуль**: `ConcretizerWithReflection`
- **Метрики** (id:23):
  - `equivalence_after_concretization` - эквивалентность после конкретизации

### ✅ evaluation/
- **Метрики** (id:22):
  - `stability_after_reabstraction` - стабильность после повторного абстрагирования

## Описание метрик

### Метрики абстрагирования (id:24)
- **sufficient_abstraction**: LLM-as-Judge проверяет, что A-T-B достаточно обобщает связку без потери ключевого смысла
- **not_over_abstracted**: LLM-as-Judge проверяет, что A-T-B не слишком общий

### Метрики конкретизации (id:23)
- **equivalence_after_concretization**: LLM-as-Judge проверяет эквивалентность между исходной связкой и конкретизацией из A-T-B

### Метрики стабильности (id:22)
- **stability_after_reabstraction**: проверка через Jaccard-подобие, что повторное абстрагирование конкретизации даёт похожий результат

## Использование

```python
from config import configure_llm
from pipeline import EpiStack

configure_llm()

text = "Ваш текст для анализа..."
pipeline = EpiStack(use_efficient_abstraction=True)
result = pipeline(text=text)

# Результат содержит метрики для каждой связки:
# {
#   "items": [
#     {
#       "relation": "...",
#       "name": "...",
#       "ATB": {"A": "...", "T": "...", "B": "..."},
#       "concretized": "...",
#       "metrics": {
#         "sufficient_abstraction": 0|1,      # id:24
#         "not_over_abstracted": 0|1,         # id:24
#         "equivalence_after_concretization": 0|1,  # id:23
#         "stability_after_reabstraction": 0|1      # id:22
#       }
#     }
#   ]
# }
```

## Зависимости

- dspy-ai>=2.4.0
