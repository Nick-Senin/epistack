"""
Оптимизация модуля TransformationMarker
"""
import dspy
from .module import TransformationMarker
from .metrics import TransformationMarkerMetric


def optimize(dataset=None, max_metric_calls=50, optimizer_type='gepa'):
    """
    Оптимизация модуля TransformationMarker с использованием DSPy оптимизаторов
    
    Args:
        dataset: Датасет для оптимизации (list of dspy.Example).
                Каждый пример должен содержать:
                - text: исходный текст
                - transformations: массив преобразований (list[str])
                - expected_marked_text (опционально): эталонный результат
        max_metric_calls: Максимальное количество вызовов метрики (для GEPA)
        optimizer_type: Тип оптимизатора ('gepa', 'mipro', 'bootstrap')
        
    Returns:
        Оптимизированный модуль TransformationMarker
        
    Example:
        >>> dataset = [
        ...     dspy.Example(
        ...         text="Данные проходят нормализацию и фильтрацию",
        ...         transformations=["нормализация", "фильтрация"],
        ...         expected_marked_text="Данные проходят **нормализацию** и **фильтрацию**"
        ...     ).with_inputs("text", "transformations")
        ... ]
        >>> optimized = optimize(dataset, max_metric_calls=20)
    """
    # Проверка датасета
    if dataset is None:
        raise ValueError(
            "Необходим датасет для оптимизации.\n"
            "Каждый пример должен содержать:\n"
            "  - text: исходный текст\n"
            "  - transformations: список преобразований\n"
            "  - expected_marked_text (опционально): эталонный результат\n\n"
            "Пример:\n"
            "  dataset = [\n"
            "      dspy.Example(\n"
            "          text='Ваш текст',\n"
            "          transformations=['преобразование1', 'преобразование2'],\n"
            "          expected_marked_text='Текст с **выделениями**'\n"
            "      ).with_inputs('text', 'transformations')\n"
            "  ]"
        )
    
    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Датасет должен быть непустым списком dspy.Example")
    
    # Разделение на train/val
    split_idx = int(len(dataset) * 0.8)
    if split_idx == 0:
        split_idx = 1
    
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:] if split_idx < len(dataset) else dataset[:1]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val примеров")
    
    # Создание метрики
    metric = TransformationMarkerMetric(use_similarity=True)
    
    # Выбор и настройка оптимизатора
    if optimizer_type == 'gepa':
        # GEPA - эволюционная оптимизация промптов с рефлексией
        print(f"🔧 Используется GEPA оптимизатор (max_metric_calls={max_metric_calls})")
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
        print(f"🔧 Используется MIPRO оптимизатор")
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=10,
            init_temperature=1.0
        )
    elif optimizer_type == 'bootstrap':
        # Bootstrap - генерация примеров для few-shot
        print(f"🔧 Используется Bootstrap оптимизатор")
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
    
    # Создание и компиляция модуля
    print("🚀 Начинаем оптимизацию...")
    module = TransformationMarker()
    
    try:
        optimized = optimizer.compile(
            module,
            trainset=trainset,
            valset=valset
        )
        
        print(f"✅ Модуль успешно оптимизирован с {optimizer_type.upper()}")
        return optimized
        
    except Exception as e:
        print(f"❌ Ошибка при оптимизации: {e}")
        raise


def create_example_dataset():
    """
    Создаёт примерный датасет для тестирования оптимизации
    
    Returns:
        list: Список dspy.Example с примерами
    """
    examples = [
        dspy.Example(
            text="Первым этапом обработки является нормализация данных. "
                 "Затем применяется фильтрация выбросов. "
                 "В конце выполняется агрегация результатов.",
            transformations=["нормализация", "фильтрация", "агрегация"],
            expected_marked_text="Первым этапом обработки является **нормализация** данных. "
                                "Затем применяется **фильтрация** выбросов. "
                                "В конце выполняется **агрегация** результатов."
        ).with_inputs("text", "transformations"),
        
        dspy.Example(
            text="В процессе трансформации данных мы масштабируем признаки "
                 "и кодируем категориальные переменные.",
            transformations=["масштабирование", "кодирование"],
            expected_marked_text="В процессе трансформации данных мы **масштабируем признаки** "
                                "и **кодируем категориальные переменные**."
        ).with_inputs("text", "transformations"),
        
        dspy.Example(
            text="Модель обучается на тренировочных данных, затем тестируется на валидационной выборке.",
            transformations=["обучение", "тестирование"],
            expected_marked_text="Модель **обучается на тренировочных данных**, затем **тестируется на валидационной выборке**."
        ).with_inputs("text", "transformations"),
    ]
    
    return examples

