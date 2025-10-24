# Установка и использование epistack

## Установка пакета

### 1. Установка в режиме разработки (editable)

```bash
cd /Users/admin/Documents/repos/epistack
pip install -e .
```

**Преимущества:**
- Изменения в коде сразу доступны без переустановки
- Пакет доступен из любой директории
- Идеально для разработки

### 2. Обычная установка

```bash
cd /Users/admin/Documents/repos/epistack
pip install .
```

### 3. Проверка установки

```bash
python test_imports.py
```

Должно вывести:
```
✓ configure_llm
✓ RelationExtractor, ExtractRelationsSig
✓ RelationNamer, CausalRelationExtractorSignature
✓ module_abstraction компоненты
✓ module_concretization компоненты
✓ utils компоненты
✓ Прямой импорт: epistack.module_abstraction
✓ Прямой импорт: epistack.module_naming
✓ Прямой импорт: epistack.module_extraction
✓ Прямой импорт: epistack.module_concretization
✓ Прямой импорт: epistack.config
✓ Прямой импорт: epistack.utils

✅ Все импорты успешны!
```

## Использование

### Вариант 1: Импорт из основного пакета

```python
from epistack import (
    configure_llm,
    RelationNamer,
    NaiveATBAbstraction,
    ConcretizerWithReflection
)

configure_llm()
namer = RelationNamer()
```

### Вариант 2: Импорт отдельных модулей

```python
from epistack.module_naming import RelationNamer
from epistack.module_abstraction import NaiveATBAbstraction, AbstractionMetrics
from epistack.module_concretization import ConcretizerWithReflection

namer = RelationNamer()
abstractor = NaiveATBAbstraction()
```

### Вариант 3: Импорт модуля целиком

```python
from epistack import module_naming
from epistack import module_abstraction

namer = module_naming.RelationNamer()
abstractor = module_abstraction.NaiveATBAbstraction()
```

## Структура пакета

```
epistack/
├── config/                  # Конфигурация LLM
│   └── llm.py
├── module_extraction/       # Извлечение связок
│   ├── module.py
│   └── signatures.py
├── module_naming/           # Именование связок
│   ├── module.py
│   └── signatures.py
├── module_abstraction/      # Абстрагирование A-T-B
│   ├── module.py
│   ├── signatures.py
│   └── metrics.py
├── module_concretization/   # Конкретизация из A-T-B
│   ├── module.py
│   ├── signatures.py
│   └── metrics.py
└── utils/                   # Вспомогательные функции
    └── helpers.py
```

## Доступные компоненты

### config
- `configure_llm()` - конфигурация LLM

### module_extraction
- `RelationExtractor` - класс для извлечения связок
- `ExtractRelationsSig` - сигнатура

### module_naming
- `RelationNamer` - класс для именования связок
- `CausalRelationExtractorSignature` - сигнатура

### module_abstraction
- `NaiveATBAbstraction` - наивная абстракция
- `EfficientATBAbstraction` - эффективная абстракция
- `AbstractATBSig`, `CritiqueSig`, `ReviseSig` - сигнатуры
- `AbstractionMetrics` - метрики

### module_concretization
- `ConcretizerWithReflection` - конкретизация с рефлексией
- `ConcretizeFromATBSig` - сигнатура
- `ConcretizationMetrics` - метрики

### utils
- `safe_json_dict()` - безопасное преобразование в JSON dict
- `jaccard_like()` - Jaccard similarity

## Удаление пакета

```bash
pip uninstall epistack
```

