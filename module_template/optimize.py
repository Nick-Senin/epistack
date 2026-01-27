"""
Модуль оптимизации DSPy программ

Предоставляет функции для компиляции и оптимизации DSPy программ
с использованием различных оптимизаторов: MIPROv2, GEPA, BootstrapFinetune
"""
import os
import dspy
import dotenv
from typing import Callable, Optional, Any


def configure_optimizer(
    optimizer_type: str = "mipro",
    metric: Optional[Callable] = None,
    auto: str = "medium",
    num_threads: int = 24,
    track_stats: bool = False,
    **kwargs
) -> Any:
    """
    Настройка и создание оптимизатора DSPy

    Args:
        optimizer_type: Тип оптимизатора ('mipro', 'gepa', 'bootstrap')
        metric: Функция метрики для оценки качества
        auto: Уровень оптимизации ('light', 'medium', 'heavy')
        num_threads: Количество потоков для параллельной обработки
        track_stats: Отслеживать статистику (для GEPA)
        **kwargs: Дополнительные параметры для оптимизатора

    Returns:
        Сконфигурированный оптимизатор
    """
    dotenv.load_dotenv()

    if optimizer_type == "mipro":
        from dspy.teleprompt import MIPROv2
        return dspy.MIPROv2(
            metric=metric,
            auto=auto,
            num_threads=num_threads,
            **kwargs
        )

    elif optimizer_type == "gepa":
        return dspy.GEPA(
            metric=metric,
            auto=auto,
            num_threads=num_threads,
            track_stats=track_stats,
            **kwargs
        )

    elif optimizer_type == "bootstrap":
        dspy.settings.experimental = True
        return dspy.BootstrapFinetune(
            num_threads=num_threads,
            metric=metric,
            **kwargs
        )

    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")


def optimize_program(
    program: dspy.Program,
    trainset: list,
    valset: Optional[list] = None,
    optimizer_type: str = "mipro",
    metric: Optional[Callable] = None,
    max_bootstrapped_demos: int = 2,
    max_labeled_demos: int = 2,
    save_path: Optional[str] = None,
    **optimizer_kwargs
) -> dspy.Program:
    """
    Компиляция и оптимизация DSPy программы

    Args:
        program: DSPy программа для оптимизации
        trainset: Обучающий набор данных
        valset: Валидационный набор (обязателен для GEPA)
        optimizer_type: Тип оптимизатора ('mipro', 'gepa', 'bootstrap')
        metric: Функция метрики для оценки
        max_bootstrapped_demos: Макс. количество bootstrapped демо
        max_labeled_demos: Макс. количество labeled демо
        save_path: Путь для сохранения оптимизированной программы
        **optimizer_kwargs: Дополнительные параметры оптимизатора

    Returns:
        Оптимизированная программа
    """
    optimizer = configure_optimizer(
        optimizer_type=optimizer_type,
        metric=metric,
        **optimizer_kwargs
    )

    # Компиляция с параметрами по умолчанию
    compile_kwargs = {
        "trainset": trainset,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    # Добавляем valset если требуется (для GEPA)
    if valset is not None:
        compile_kwargs["valset"] = valset

    optimized_program = optimizer.compile(program.deepcopy(), **compile_kwargs)

    # Сохранение если указан путь
    if save_path:
        optimized_program.save(save_path)

    return optimized_program


def load_optimized_program(path: str) -> dspy.Program:
    """
    Загрузка ранее оптимизированной программы

    Args:
        path: Путь к сохранённой программе

    Returns:
        Загруженная оптимизированная программа
    """
    return dspy.Program.load(path)


# TODO: Добавить специфичные метрики для модуля
# Например:
# def semantic_split_metric(gold, pred, trace=None):
#     """Метрика для оценки качества семантической сегментации"""
#     pass
