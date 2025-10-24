# Module Abstraction

Модуль для абстрагирования связок в формат A-T-B.

## Компоненты

- `module.py` - `NaiveATBAbstraction`, `EfficientATBAbstraction` классы
- `signatures.py` - `AbstractATBSig`, `CritiqueSig`, `ReviseSig`
- `metrics.py` - `AbstractionMetrics`
- `optimize.py` - функция оптимизации модуля

## Оптимизация

```python
from epistack.module_abstraction import optimize

# Оптимизация модуля
optimized_abstractor = optimize(hf_username='Nick-Sen')
```

### Или через CLI

```bash
# Только abstraction модуль
python optimize_modules.py Nick-Sen abstraction

# Все модули
python optimize_modules.py Nick-Sen all
```

## Метрика

Проверяет, что абстракции отличаются от конкретных названий.

