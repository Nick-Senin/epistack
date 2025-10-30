# Module Extraction

Модуль для извлечения связок из текста.

## Компоненты

- `module.py` - `RelationExtractor` класс
- `signatures.py` - `ExtractRelationsSig`
- `optimize.py` - функция оптимизации модуля

## Оптимизация

```python
from epistack.module_extraction import optimize

# Оптимизация модуля
optimized_extractor = optimize(hf_username='Nick-Sen')
```

### Или через CLI

```bash
# Только extraction модуль
python optimize_modules.py Nick-Sen extraction

# Все модули
python optimize_modules.py Nick-Sen all
```

## Метрика

Проверяет количество найденных связок с допуском ±1.

