"""
Оптимизация модуля именования
"""
import os
import json
from pathlib import Path
from typing import Optional

import dotenv
import dspy
from epistack_data import for_naming_module
from .module import RelationNamer
from .metrics import create_metric


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "kollektives_dataset.json"


def _clean_text(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return None


def _load_kollektives_dataset(dataset_path: Path):
    """
    Загружает локальный датасет kollektives_dataset.json и преобразует его в dspy.Example.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    try:
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {dataset_path}: {exc}") from exc

    examples = []
    skipped = 0

    for record in raw_records:
        title = _clean_text(record.get("Название примера") or record.get("title"))
        source_text = _clean_text(
            record.get("Исходный пример") or record.get("source_text") or record.get("case")
        )

        if not title or not source_text:
            skipped += 1
            continue

        examples.append(
            dspy.Example(
                source_text=source_text,
                title=title,
            ).with_inputs("source_text")
        )

    if not examples:
        raise ValueError(
            f"В {dataset_path} не найдено записей с заполненными кейсами и названиями."
        )

    print(
        f"✅ Загружено {len(examples)} примеров из {dataset_path} "
        f"(пропущено {skipped} неполных записей)"
    )
    return examples


OPTIMIZATION_MODEL_ID = "openrouter/moonshotai/kimi-k2-thinking"


def _configure_optimization_lm():
    """
    Настраивает LLM конкретно для процедуры оптимизации.
    """
    dotenv.load_dotenv()
    api_base = (
        os.getenv("OPENROUTER_API_BASE")
        or os.getenv("OPENROUTER_BASE")
        or "https://openrouter.ai/api/v1"
    )
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    lm = dspy.LM(
        model=OPTIMIZATION_MODEL_ID,
        api_base=api_base,
        api_key=api_key,
    )
    dspy.configure(lm=lm)
    return lm


def optimize(
    hf_username: Optional[str] = None,
    max_metric_calls: int = 50,
    dataset_path: Optional[str] = None,
):
    """
    Оптимизация модуля именования с использованием GEPA
    
    Args:
        hf_username: HuggingFace username для загрузки датасета (None = локальный)
        max_metric_calls: Максимальное количество вызовов метрики
        dataset_path: Путь к kollektives_dataset.json (используется при локальной загрузке)
        
    Returns:
        Оптимизированный модуль RelationNamer
    """
    # Загрузка датасета (локальный JSON или HF)
    if hf_username:
        dataset = for_naming_module(hf_username)
    else:
        target_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        dataset = _load_kollektives_dataset(target_path)
    
    if len(dataset) < 2:
        raise ValueError("Для оптимизации требуется минимум 2 примера.")
    
    # Разделяем на train/val (80/20)
    split_idx = max(1, int(len(dataset) * 0.8))
    trainset = dataset[:split_idx]
    valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # Настраиваем специализированную модель и метрику LLM as Judge
    optimization_lm = _configure_optimization_lm()
    metric = create_metric()
    
    # Создаем LM для рефлексии (используем текущую настроенную LM)
    reflection_lm = optimization_lm
    
    # GEPA оптимизация с рефлективной эволюцией промптов
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
    
    module = RelationNamer()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print("✅ RelationNamer оптимизирован с GEPA")
    return optimized

