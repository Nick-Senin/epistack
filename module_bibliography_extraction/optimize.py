"""
Оптимизация модуля извлечения библиографической информации
"""
import dspy
from .module import BibliographyExtraction
from .metrics import BibliographyMetric


def optimize(dataset=None, max_metric_calls=50, optimizer_type='gepa'):
    """
    Оптимизация модуля с использованием DSPy оптимизаторов
    
    Args:
        dataset: Датасет для оптимизации (list of dspy.Example)
                 Каждый пример должен содержать поля:
                 - text: входной текст
                 - title, author, publisher, year, place: ожидаемые значения
        max_metric_calls: Максимальное количество вызовов метрики
        optimizer_type: Тип оптимизатора ('gepa', 'mipro', 'bootstrap')
        
    Returns:
        Оптимизированный модуль BibliographyExtraction
        
    Example:
        >>> import dspy
        >>> from module_bibliography_extraction import optimize
        >>> 
        >>> # Создаем датасет
        >>> dataset = [
        ...     dspy.Example(
        ...         text="Война и мир. Лев Толстой. М.: Наука, 1869.",
        ...         title="Война и мир",
        ...         author="Лев Толстой",
        ...         publisher="Наука",
        ...         year="1869",
        ...         place="М."
        ...     ).with_inputs("text")
        ... ]
        >>> 
        >>> # Оптимизируем
        >>> optimized_module = optimize(dataset, max_metric_calls=50, optimizer_type='gepa')
        >>> 
        >>> # Сохраняем
        >>> optimized_module.save('optimized_module.json')
    """
    # Проверка наличия датасета
    if dataset is None:
        raise ValueError(
            "Необходим датасет для оптимизации.\n"
            "Пример создания датасета:\n"
            "dataset = [\n"
            "    dspy.Example(\n"
            "        text='Текст с библиографией',\n"
            "        title='Название',\n"
            "        author='Автор',\n"
            "        publisher='Издательство',\n"
            "        year='Год',\n"
            "        place='Место'\n"
            "    ).with_inputs('text')\n"
            "]"
        )
    
    # Разделение на train/val
    split_idx = int(len(dataset) * 0.8)
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # Создание метрики
    metric = BibliographyMetric()
    
    # Выбор и настройка оптимизатора
    if optimizer_type == 'gepa':
        # GEPA - эволюционная оптимизация промптов с рефлексией
        print("🔧 Используется оптимизатор GEPA")
        reflection_lm = dspy.settings.lm
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max_metric_calls,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
            candidate_selection_strategy='pareto',
            skip_perfect_score=True,
            track_stats=True,
            seed=42
        )
    elif optimizer_type == 'mipro':
        # MIPRO - оптимизация промптов и примеров
        print("🔧 Используется оптимизатор MIPRO")
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=10,
            init_temperature=1.0
        )
    elif optimizer_type == 'bootstrap':
        # Bootstrap - генерация примеров для few-shot
        print("🔧 Используется оптимизатор Bootstrap")
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
    
    # Создание и компиляция модуля
    print("🚀 Начинается оптимизация...")
    module = BibliographyExtraction()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print(f"✅ Модуль оптимизирован с {optimizer_type.upper()}")
    return optimized

