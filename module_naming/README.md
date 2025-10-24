# Module Naming

Модуль для именования связок.

## Компоненты

- `module.py` - `RelationNamer` класс
- `signatures.py` - `CausalRelationExtractorSignature` 
- `optimize.py` - функция оптимизации модуля

## Оптимизация

```python
from epistack.module_naming import optimize

# Оптимизация модуля
optimized_namer = optimize(hf_username='Nick-Sen')
```

### Или через CLI

```bash
# Только naming модуль
python optimize_modules.py Nick-Sen naming

# Все модули
python optimize_modules.py Nick-Sen all
```

## Метрика

Проверяет, что название содержится в предсказании или наоборот.

