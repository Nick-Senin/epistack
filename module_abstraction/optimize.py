"""
Оптимизация модуля абстракции троек состояний с помощью GEPA.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import dotenv
import dspy

from data_models.state_triple import StateTriple
from epistack_data import for_abstraction_module
from .metrics import StateTripleSimilarityMetric
from .module import NaiveStateTripleAbstraction


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "abstraction_dataset.json"
OPTIMIZATION_MODEL_ID = "openrouter/moonshotai/kimi-k2-thinking"


def _clean_text(value: Any) -> Optional[str]:
    """
    Нормализует строку: str -> strip -> None если пусто.
    """
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def _make_state_triple(
    initial: Optional[str],
    transformation: Optional[str],
    final_state: Optional[str],
) -> Optional[StateTriple]:
    """
    Собирает StateTriple при наличии всех трёх компонент.
    """
    if not (initial and transformation and final_state):
        return None
    triple: StateTriple = {
        "initial_state": initial,
        "transformation": transformation,
        "final_state": final_state,
    }
    return triple


def _example_from_triples(source: StateTriple, target: StateTriple) -> dspy.Example:
    """
    Конвертирует пару троек в DSPy пример.
    """
    return (
        dspy.Example(
            state_triple=source,
            abstract_state_triple=target,
        ).with_inputs("state_triple")
    )


def _local_record_examples(record: Dict[str, Any]) -> Tuple[List[dspy.Example], int]:
    """
    Формирует примеры из одной записи локального датасета.
    Возвращает (примеры, пропущено).
    """
    examples: List[dspy.Example] = []
    skipped = 0
    index = 1

    while True:
        keys = (
            f"СВЯЗКА {index} - initial_state",
            f"СВЯЗКА {index} - transformation",
            f"СВЯЗКА {index} - result",
            f"СВЯЗКА {index} - АБСТРАКЦИЯ - initial_state",
            f"СВЯЗКА {index} - АБСТРАКЦИЯ - transformation",
            f"СВЯЗКА {index} - АБСТРАКЦИЯ - result",
        )
        has_data = any(_clean_text(record.get(key)) for key in keys)
        if not has_data:
            break

        source = _make_state_triple(
            _clean_text(record.get(keys[0])),
            _clean_text(record.get(keys[1])),
            _clean_text(record.get(keys[2])),
        )
        target = _make_state_triple(
            _clean_text(record.get(keys[3])),
            _clean_text(record.get(keys[4])),
            _clean_text(record.get(keys[5])),
        )

        if source and target:
            examples.append(_example_from_triples(source, target))
        else:
            skipped += 1

        index += 1

    return examples, skipped


def _load_local_dataset(dataset_path: Path) -> List[dspy.Example]:
    """
    Загружает локальный abstraction_dataset.json и превращает в примеры DSPy.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    try:
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {dataset_path}: {exc}") from exc

    examples: List[dspy.Example] = []
    skipped_total = 0

    for record in raw_records:
        record_examples, skipped = _local_record_examples(record or {})
        examples.extend(record_examples)
        skipped_total += skipped

    if not examples:
        raise ValueError(
            f"В {dataset_path} не найдено полных пар (исходная тройка + абстракция)."
        )

    print(
        f"✅ Загружено {len(examples)} троек из {dataset_path} "
        f"(пропущено {skipped_total} неполных связок)"
    )
    return examples


def _state_triple_from_mapping(payload: Mapping[str, Any]) -> Optional[StateTriple]:
    """
    Преобразует словарь в StateTriple, учитывая разные варианты ключей.
    """
    initial = _clean_text(payload.get("initial_state") or payload.get("initial"))
    transformation = _clean_text(payload.get("transformation") or payload.get("action"))
    final_state = _clean_text(
        payload.get("final_state") or payload.get("result") or payload.get("final")
    )
    return _make_state_triple(initial, transformation, final_state)


def _hf_examples(hf_username: str) -> List[dspy.Example]:
    """
    Собирает примеры из HF-датасета (extracted_chains/abstracted_chains).
    """
    dataset = for_abstraction_module(hf_username)
    examples: List[dspy.Example] = []
    skipped = 0

    for item in dataset:
        extracted = getattr(item, "extracted_chains", None) or []
        abstracted = getattr(item, "abstracted_chains", None) or []
        limit = min(len(extracted), len(abstracted))

        for idx in range(limit):
            source = _state_triple_from_mapping(extracted[idx])
            target = _state_triple_from_mapping(abstracted[idx])
            if source and target:
                examples.append(_example_from_triples(source, target))
            else:
                skipped += 1

    if not examples:
        raise ValueError(
            "HF-датасет не содержит совпадающих троек (extracted_chains vs abstracted_chains)."
        )

    print(f"✅ Подготовлено {len(examples)} троек из HF датасета ({skipped} пропущено)")
    return examples


def _load_examples(
    hf_username: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> List[dspy.Example]:
    """
    Определяет источник данных: HF или локальный JSON.
    """
    if hf_username:
        return _hf_examples(hf_username)
    target_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    return _load_local_dataset(target_path)


def _split_dataset(examples: List[dspy.Example]) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Делит датасет на train/val (80/20) с гарантией минимум 1 пример в каждой части.
    """
    if len(examples) < 2:
        raise ValueError("Для оптимизации требуется минимум 2 примера.")
    split_idx = max(1, int(len(examples) * 0.8))
    trainset = examples[:split_idx]
    valset = examples[split_idx:]
    if not valset:
        valset = trainset[-1:]
        trainset = trainset[:-1]
    print(f"📊 Датасет троек: {len(trainset)} train, {len(valset)} val")
    return trainset, valset


def _configure_optimization_lm() -> dspy.LM:
    """
    Настраивает LLM под GEPA-оптимизацию.
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
    max_metric_calls: int = 75,
    dataset_path: Optional[str] = None,
    reflection_minibatch_size: int = 3,
) -> NaiveStateTripleAbstraction:
    """
    GEPA-оптимизация модуля NaiveStateTripleAbstraction.
    
    Args:
        hf_username: HuggingFace username (если None — используем локальный JSON).
        max_metric_calls: Ограничение на количество вызовов метрики.
        dataset_path: Путь к локальному abstraction_dataset.json.
        reflection_minibatch_size: Размер минибатча для отражения GEPA.
        
    Returns:
        Оптимизированный модуль NaiveStateTripleAbstraction.
    """
    examples = _load_examples(hf_username=hf_username, dataset_path=dataset_path)
    trainset, valset = _split_dataset(examples)

    optimization_lm = _configure_optimization_lm()
    metric = StateTripleSimilarityMetric()

    optimizer = dspy.GEPA(
        metric=metric,
        max_metric_calls=max_metric_calls,
        reflection_lm=optimization_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        track_stats=True,
        seed=42,
    )

    module = NaiveStateTripleAbstraction()
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset,
    )

    print("✅ NaiveStateTripleAbstraction оптимизирован с GEPA")
    return optimized

