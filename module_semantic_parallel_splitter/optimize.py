"""
Оптимизация модуля семантической сегментации для SemanticHalver
"""
import os
import json
import dspy
from .module import SemanticHalver
from .config import configure_module_llm

DSPY_LM_DOCS_URL = "https://dspy.ai"


def _lm_output_to_text(output):
    """
    Нормализует ответ LLM к строке.

    GEPA ожидает строку и вызывает .strip(). Некоторые провайдеры возвращают dict/list.
    """
    if isinstance(output, dict):
        return (output.get("text") or output.get("completion") or "").strip()
    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, dict):
            return (first.get("text") or first.get("completion") or "").strip()
        return str(first).strip()
    return str(output).strip()


class _TextOnlyLM:
    """Обёртка для LLM, чтобы GEPA всегда получал строку."""

    def __init__(self, lm):
        self._lm = lm
        self.model = getattr(lm, "model", "unknown")

    def __call__(self, prompt, **kwargs):
        return _lm_output_to_text(self._lm(prompt, **kwargs))


def create_reflection_lm(
    model=None,
    api_base=None,
    api_key=None,
    **kwargs
):
    """
    Создаёт LM для reflection (оценки и улучшения промптов).

    Returns:
        _TextOnlyLM: LLM-обёртка, возвращающая строку
    """
    # См. документацию DSPy по LM:
    # https://dspy.ai
    model = (model or os.getenv("REFLECTION_MODEL") or os.getenv("CEREBRAS_MODEL") or "cerebras/zai-glm-4.7").strip()
    api_base = (api_base or os.getenv("REFLECTION_API_BASE") or os.getenv("CEREBRAS_API_BASE") or "https://api.cerebras.ai/v1").strip()
    api_key = (api_key or os.getenv("REFLECTION_API_KEY") or os.getenv("CEREBRAS_API_KEY") or "").strip()

    if not api_key:
        raise ValueError(
            "Не найден API ключ для reflection-LM. Укажите `REFLECTION_API_KEY` или `CEREBRAS_API_KEY`."
        )

    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key=api_key,
        **kwargs
    )

    wrapped_lm = _TextOnlyLM(lm)
    print(f"🔧 Reflection LM configured: {wrapped_lm.model}")
    return wrapped_lm


class SemanticHalverMetric:
    """
    Метрика для оценки качества SemanticHalver.

    Оценивает точность определения первого смыслового блока.

    Для GEPA метрика должна возвращать dspy.Prediction с полями:
    - score: оценка от 0.0 до 1.0
    - feedback: текстовый фидбек для улучшения
    """

    def __init__(self):
        """Инициализация метрики."""
        pass

    def __call__(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        """
        Оценка качества разделения текста.

        Args:
            example: Пример из датасета с полями:
                - text: полный текст (part1 + part2)
                - ground_truth_first_block: ожидаемый первый блок (part1)
            pred: Предсказание модуля с полями:
                - first_block: предсказанный первый блок
                - split_index: индекс разделения
            trace: Опциональный трейс выполнения
            pred_name: Имя предиктора (для GEPA)
            pred_trace: Трейс предиктора (для GEPA)

        Returns:
            float: Оценка от 0.0 до 1.0 (или dspy.Prediction для GEPA)
        """
        try:
            # Получаем ground truth и предсказание
            ground_truth = example.ground_truth_first_block.strip()
            predicted = pred.first_block.strip()

            # Если предсказание пустое - низкая оценка
            if not predicted:
                return self._make_result(0.0, "Предсказанный блок пустой")

            # Полное совпадение - максимальная оценка
            if ground_truth == predicted:
                return self._make_result(1.0, "Идеальное совпадение")

            # Проверяем, что предсказанная строка является подстрокой исходного текста
            full_text = example.text
            if predicted not in full_text:
                # LLM нарушил инструкцию "точная подстрока"
                return self._make_result(
                    0.0,
                    f"LLM нарушил инструкцию 'точная подстрока'. "
                    f"Предсказанный блок не найден в исходном тексте. "
                    f"Ожидалось: {ground_truth[:100]}... "
                    f"Получено: {predicted[:100]}..."
                )

            # Вычисляем overlap на основе позиций
            gt_len = len(ground_truth)
            pred_len = len(predicted)

            # Если длины сильно отличаются - низкая оценка
            len_diff = abs(gt_len - pred_len) / max(gt_len, pred_len)
            if len_diff > 0.5:  # Если разница в длине > 50%
                return self._make_result(
                    0.1,
                    f"Слишком большая разница в длине. "
                    f"Ожидалось: {gt_len} символов, получено: {pred_len} символов"
                )

            # Вычисляем overlap по символам
            # Находим максимальную общую подстроку
            max_overlap = 0
            for i in range(min(gt_len, pred_len)):
                if ground_truth[:i] == predicted[:i]:
                    max_overlap = i
                else:
                    break

            overlap_ratio = max_overlap / max(gt_len, pred_len)

            # Учитываем также разницу в длине
            score = overlap_ratio * (1 - len_diff)

            # Формируем фидбек
            feedback = (
                f"Overlap: {overlap_ratio:.2%}, разница длины: {len_diff:.2%}. "
                f"Ожидалось блок длиной {gt_len}, получено {pred_len}. "
                f"Совпало первых {max_overlap} символов."
            )

            return self._make_result(max(0.0, min(1.0, score)), feedback)

        except Exception as e:
            return self._make_result(0.0, f"Ошибка при вычислении метрики: {e}")

    def _make_result(self, score, feedback):
        """
        Создаёт результат в правильном формате.

        Для GEPA возвращаем dspy.Prediction с feedback,
        для других оптимизаторов просто float.
        """
        return dspy.Prediction(score=score, feedback=feedback)


def load_dataset(filepath=None, limit=15):
    """
    Загрузка датасета из JSON файла.

    Args:
        filepath: Путь к JSON файлу с датасетом.
                   Если None, используется путь по умолчанию.
        limit: Максимальное число чанков для загрузки (None = без лимита).

    Returns:
        list[dspy.Example]: Список примеров для обучения
    """
    if filepath is None:
        # Путь по умолчанию к датасету
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            "splitting datasets",
            "split_chunks.json"
        )

    print(f"📂 Загрузка датасета из: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    examples = []
    for item in data:
        # Пропускаем записи без part1 или part2
        if 'part1' not in item or 'part2' not in item:
            continue

        # Объединяем part1 и part2 для получения полного текста
        full_text = item['part1'] + ' ' + item['part2']

        # Создаём dspy.Example
        # text - вход, ground_truth_first_block - ожидаемый выход
        example = dspy.Example(
            text=full_text,
            ground_truth_first_block=item['part1']
        ).with_inputs('text')

        examples.append(example)

    print(f"✅ Загружено {len(examples)} примеров")
    return examples


def optimize(
    dataset=None,
    max_metric_calls=50,
    optimizer_type='gepa',
    metric=None,
    valset_ratio=0.2,
    **kwargs
):
    """
    Оптимизация модуля SemanticHalver.

    Args:
        dataset: Датасет с примерами для оптимизации.
                 Если None, датасет загружается из файла по умолчанию.
        max_metric_calls: Максимальное количество вызовов метрики (не для GEPA)
        optimizer_type: Тип оптимизатора ('gepa', 'mipro', 'bootstrap')
        metric: Метрика для оценки (если None, используется SemanticHalverMetric)
        **kwargs: Дополнительные параметры для оптимизатора
                  Для GEPA: auto="light"|"medium"|"heavy", num_threads, valset
        valset_ratio: Доля данных для валидации, если valset не передан

    Returns:
        Оптимизированный модуль
    """
    # Настройка LLM
    configure_module_llm()

    # Загружаем датасет если не предоставлен
    if dataset is None:
        dataset = load_dataset()

    if not dataset:
        raise ValueError("Датасет пуст или не загружен")

    # Создаём базовый модуль
    student = SemanticHalver()

    # Создаём метрику если не предоставлена
    if metric is None:
        metric = SemanticHalverMetric()

    # Выбор оптимизатора
    if optimizer_type == 'gepa':
        # GEPA: Genealogical Effective Prompt Optimization
        reflection_lm = kwargs.pop('reflection_lm', None)
        if reflection_lm is None:
            reflection_lm = create_reflection_lm()

        # Извлекаем параметры GEPA
        auto = kwargs.pop('auto', 'light')
        num_threads = kwargs.pop('num_threads', 4)

        optimizer = dspy.GEPA(
            metric=metric,
            auto=auto,
            num_threads=num_threads,
            reflection_lm=reflection_lm,
            **kwargs
        )
    elif optimizer_type == 'mipro':
        # MIPRO: Multi-step Instruction Proposal Optimization
        num_trials = kwargs.pop('num_trials', 10)

        optimizer = dspy.MIPROv2(
            student=student,
            metric=metric,
            num_trials=num_trials,
            max_metric_calls=max_metric_calls,
            **kwargs
        )
    elif optimizer_type == 'bootstrap':
        # Bootstrap: Few-shot примеры
        max_labeled_demos = kwargs.pop('max_labeled_demos', 8)
        max_rounds = kwargs.pop('max_rounds', 1)

        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            **kwargs
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")

    # Подготовка train/val для GEPA
    valset = kwargs.pop('valset', None)
    trainset = dataset
    if optimizer_type == 'gepa' and valset is None:
        # Простой детерминированный сплит: первые N для valset
        val_count = max(1, int(len(dataset) * valset_ratio))
        valset = dataset[:val_count]
        trainset = dataset[val_count:] or dataset

    # Запуск оптимизации
    print(f"🚀 Запуск оптимизации с {optimizer_type.upper()}...")
    print(f"📊 Размер датасета: {len(dataset)}")
    if optimizer_type == 'gepa' and valset is not None:
        print(f"🧪 Размер valset: {len(valset)}; trainset: {len(trainset)}")

    if optimizer_type == 'gepa':
        # GEPA использует другой API: program positional, trainset, valset
        optimized_module = optimizer.compile(
            student,  # positional argument, not keyword
            trainset=trainset,
            valset=valset
        )
    else:
        # MIPRO и BootstrapFewShot используют student, trainset
        print(f"🔄 Максимальные вызовы метрики: {max_metric_calls}")
        optimized_module = optimizer.compile(
            student=student,
            trainset=dataset
        )

    print("✅ Оптимизация завершена")
    return optimized_module


def save_optimized_module(module, filepath):
    """
    Сохранение оптимизированного модуля.

    Args:
        module: Оптимизированный модуль
        filepath: Путь для сохранения
    """
    # DSPy сохраняет оптимизированные prompt'ы
    module.save(filepath)
    print(f"💾 Модуль сохранён в: {filepath}")


def load_optimized_module(module_class, filepath):
    """
    Загрузка оптимизированного модуля.

    Args:
        module_class: Класс модуля (например, SemanticHalver)
        filepath: Путь к сохранённому модулю

    Returns:
        Загруженный модуль
    """
    module = module_class()
    module.load(filepath)
    print(f"📂 Модуль загружен из: {filepath}")
    return module


if __name__ == "__main__":
    # Запуск по умолчанию при прямом выполнении файла/модуля.
    optimizer_type = os.getenv("OPTIMIZER_TYPE", "gepa").strip().lower()
    kwargs = {}
    if optimizer_type == "mipro":
        num_trials = os.getenv("NUM_TRIALS")
        if num_trials:
            kwargs["num_trials"] = int(num_trials)
    elif optimizer_type == "bootstrap":
        max_labeled_demos = os.getenv("MAX_LABELED_DEMOS")
        if max_labeled_demos:
            kwargs["max_labeled_demos"] = int(max_labeled_demos)
        max_rounds = os.getenv("MAX_ROUNDS")
        if max_rounds:
            kwargs["max_rounds"] = int(max_rounds)

    max_metric_calls = os.getenv("MAX_METRIC_CALLS")
    max_metric_calls = int(max_metric_calls) if max_metric_calls else 50

    optimized = optimize(
        optimizer_type=optimizer_type,
        max_metric_calls=max_metric_calls,
        **kwargs
    )
    default_save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "artifacts",
        "semantic_halver_optimized.json"
    )
    os.makedirs(os.path.dirname(default_save_path), exist_ok=True)
    save_optimized_module(optimized, default_save_path)
