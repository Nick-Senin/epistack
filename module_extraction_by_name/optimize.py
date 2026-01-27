"""
Оптимизация модуля извлечения связок с использованием GEPA
"""
import os
import json
import dotenv
import dspy
from pathlib import Path
from typing import Optional, List, Dict, Any

from epistack_data import for_extraction_module
from config import configure_llm
from .module import StateTransformationExtractor
from .metrics import create_metric, metric

# -------------------------------------------------------------------------------------------
# Константы и настройки
# -------------------------------------------------------------------------------------------

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "Интеллектуальные коллективы - тренажерка - Sheet1.json"
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
    # dspy.configure(lm=lm)  # Не устанавливаем глобально, только возвращаем для рефлексии
    return lm


# -------------------------------------------------------------------------------------------
# Загрузка данных (локально или через HF)
# -------------------------------------------------------------------------------------------

def _clean_text(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return None


def _parse_chains(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Парсит плоскую структуру (СВЯЗКА X - ...) или список triples в список словарей.
    """
    chains = []
    
    # Вариант 1: Список triples (новый формат)
    if "triples" in record and isinstance(record["triples"], list):
        for item in record["triples"]:
            initial = _clean_text(item.get("начальное состояние") or item.get("initial_state"))
            transformation = _clean_text(item.get("преобразование") or item.get("transformation"))
            final = _clean_text(item.get("результат") or item.get("final_state") or item.get("result"))
            
            if initial and transformation and final:
                chains.append({
                    "initial_state": initial,
                    "transformation": transformation,
                    "final_state": final
                })
        if chains:
            return chains

    # Вариант 2: Плоская структура (старый формат)
    # Проверяем до 10 связок (обычно их 1-4)
    for i in range(1, 11):
        prefix = f"СВЯЗКА {i}"
        
        # Ключи могут быть разными, пробуем основные варианты
        initial = _clean_text(record.get(f"{prefix} - initial_state"))
        transformation = _clean_text(record.get(f"{prefix} - transformation"))
        # В датасете часто используется 'result', но модель ожидает 'final_state'
        result = _clean_text(record.get(f"{prefix} - result") or record.get(f"{prefix} - final_state"))
        
        # Если есть хотя бы частичное заполнение, пытаемся добавить
        if initial or transformation or result:
            if initial and transformation and result:
                chains.append({
                    "initial_state": initial,
                    "transformation": transformation,
                    "final_state": result
                })
    return chains


def _get_mock_dataset() -> List[dspy.Example]:
    """
    Возвращает тестовый (mock) датасет для проверки пайплайна оптимизации.
    Используется, если основной датасет недоступен.
    """
    print("⚠️ ИСПОЛЬЗУЕТСЯ MOCK ДАТАСЕТ (реальный файл не найден или не указан) ⚠️")
    
    mock_data = [
        {
            "source_text": "Компания решила внедрить новую CRM-систему. Сотрудники прошли обучение и начали вносить данные. В результате продажи выросли на 20% за квартал.",
            "extracted_chains": [
                {
                    "initial_state": "В компании отсутствует эффективная система управления клиентами (CRM)",
                    "transformation": "Компания внедряет новую CRM-систему и обучает сотрудников работе с ней",
                    "final_state": "Продажи компании выросли на 20% благодаря систематизации данных"
                }
            ]
        },
        {
            "source_text": "Студент готовился к экзамену всю ночь, читая конспекты. Утром он выпил крепкий кофе. На экзамене он чувствовал себя бодрым, но забыл часть материала из-за усталости.",
            "extracted_chains": [
                {
                    "initial_state": "Студент имеет пробелы в знаниях перед экзаменом",
                    "transformation": "Студент интенсивно готовится всю ночь",
                    "final_state": "Студент устал и забыл часть материала"
                },
                {
                    "initial_state": "Студент чувствует сонливость утром",
                    "transformation": "Студент выпивает крепкий кофе",
                    "final_state": "Студент чувствует временную бодрость"
                }
            ]
        },
        {
            "source_text": "Разработчик обнаружил критический баг в продакшене. Он откатил последний релиз и начал искать ошибку в логах. Через час работа сервиса была восстановлена.",
            "extracted_chains": [
                {
                    "initial_state": "В продакшене обнаружен критический баг, сервис работает некорректно",
                    "transformation": "Разработчик откатывает релиз и анализирует логи",
                    "final_state": "Работа сервиса восстановлена, причина сбоя локализована"
                }
            ]
        }
    ]
    
    return [
        dspy.Example(
            source_text=item["source_text"],
            extracted_chains=item["extracted_chains"]
        ).with_inputs("source_text")
        for item in mock_data
    ]


def _load_local_dataset(dataset_path: Path) -> List[dspy.Example]:
    """
    Загружает локальный датасет и преобразует его в формат для module_extraction.
    Если файл не найден, возвращает MOCK датасет.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"ℹ️ Файл датасета не найден по пути: {dataset_path}")
        return _get_mock_dataset()

    try:
        raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"❌ Ошибка чтения JSON в {dataset_path}: {exc}")
        return _get_mock_dataset()

    examples = []
    skipped = 0

    for record in raw_records:
        source_text = _clean_text(
            record.get("Исходный пример") 
            or record.get("source_text") 
            or record.get("case") 
            or record.get("Исходный текст примера")
        )
        
        # Парсим цепочки
        extracted_chains = _parse_chains(record)

        if not source_text or not extracted_chains:
            skipped += 1
            continue

        examples.append(
            dspy.Example(
                source_text=source_text,
                extracted_chains=extracted_chains,
            ).with_inputs("source_text")
        )

    if not examples:
        print(f"⚠️ В {dataset_path} не найдено корректных примеров.")
        return _get_mock_dataset()

    print(
        f"✅ Загружено {len(examples)} примеров из {dataset_path} "
        f"(пропущено {skipped} записей)"
    )
    return examples


# -------------------------------------------------------------------------------------------
# Основная функция оптимизации
# -------------------------------------------------------------------------------------------

def optimize(
    hf_username: Optional[str] = None,
    max_metric_calls: int = 50,
    dataset_path: Optional[str] = None,
):
    """
    Оптимизация модуля извлечения связок с использованием GEPA.
    
    Args:
        hf_username: HuggingFace username для загрузки датасета (None = локальный)
        max_metric_calls: Максимальное количество вызовов метрики
        dataset_path: Путь к JSON файлу датасета (для локальной загрузки)
        
    Returns:
        Оптимизированный модуль StateTransformationExtractor
    """
    
    # 1. Загрузка данных
    if hf_username:
        print(f"📥 Загрузка датасета из HuggingFace ({hf_username})...")
        dataset = for_extraction_module(hf_username)
    else:
        target_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        print(f"📂 Попытка загрузки локального датасета: {target_path}...")
        dataset = _load_local_dataset(target_path)
    
    if len(dataset) < 2:
        print("⚠️ Слишком мало данных, переключаемся на MOCK датасет")
        dataset = _get_mock_dataset()
        
    # Разделяем на train/val (80/20)
    # Для маленького mock датасета можно использовать весь для train и val, или дублировать
    if len(dataset) <= 3:
         trainset = dataset
         valset = dataset
    else:
        split_idx = max(1, int(len(dataset) * 0.8))
        trainset = dataset[:split_idx]
        valset = dataset[split_idx:]
    
    print(f"📊 Датасет: {len(trainset)} train, {len(valset)} val")
    
    # 2. Настройка LLM и Метрики
    # Основная модель из config (глобальная настройка)
    configure_llm()
    
    # Модель для рефлексии (специальная, Moonshot)
    reflection_lm = _configure_optimization_lm()
    
    # Создаем метрику через фабрику
    # Используем функцию metric, которая совместима с GEPA и возвращает score/feedback
    gepa_metric = metric
    
    # 3. Запуск GEPA
    print("🚀 Запуск GEPA оптимизации...")
    
    optimizer = dspy.GEPA(
        metric=gepa_metric,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,  # Размер батча для генерации гипотез
        candidate_selection_strategy='pareto',
        skip_perfect_score=True,
        track_stats=True,
        seed=42
    )
    
    module = StateTransformationExtractor()
    
    # Компиляция (оптимизация)
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        valset=valset
    )
    
    print("✅ StateTransformationExtractor успешно оптимизирован")
    return optimized
