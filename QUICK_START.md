# Быстрый старт

## Установка

```bash
cd /Users/admin/Documents/repos/epistack
pip install -e .
```

## Проверка

```bash
python test_imports.py
```

Ожидаемый результат:
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

### Простой пример

```python
from epistack import configure_llm, RelationNamer

configure_llm()
namer = RelationNamer()
result = namer(text="Ваш текст...")
print(result)
```

### Использование отдельных модулей

```python
# Вариант 1: импорт из основного пакета
from epistack import NaiveATBAbstraction, RelationNamer

# Вариант 2: прямой импорт модуля
from epistack.module_abstraction import NaiveATBAbstraction
from epistack.module_naming import RelationNamer
```

## Структура импортов

После установки доступны все модули:

```python
epistack/
├── config                   → from epistack.config import configure_llm
├── module_extraction        → from epistack.module_extraction import RelationExtractor
├── module_naming           → from epistack.module_naming import RelationNamer
├── module_abstraction      → from epistack.module_abstraction import NaiveATBAbstraction
├── module_concretization   → from epistack.module_concretization import ConcretizerWithReflection
└── utils                   → from epistack.utils import safe_json_dict, jaccard_like
```

## Запуск примера

```bash
python main.py
```

## Подробная документация

- [INSTALL.md](INSTALL.md) - детальные инструкции по установке
- [README.md](README.md) - полное описание проекта
- [CHANGES.md](CHANGES.md) - журнал изменений

