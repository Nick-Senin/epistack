"""
Использование датасета для оптимизации модулей epistack
"""
from datasets import load_dataset
import dspy
import os


def load_epistack_dataset(hf_username: str, hf_token: str = None, split: str = 'train'):
    """
    Загружает датасет epistack с HF
    
    Args:
        hf_username: username на HF
        hf_token: HF токен (или из HF_TOKEN env)
        split: 'train' или 'test'
    
    Returns:
        Dataset
    """
    repo_id = f"{hf_username}/epistack-optimization"
    
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    try:
        ds = load_dataset(repo_id, token=hf_token, split=split)
        print(f"✅ Загружено {len(ds)} примеров из {repo_id} ({split})")
        return ds
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None


def to_dspy_examples(dataset, input_fields: list[str], output_fields: list[str]):
    """
    Конвертирует HF датасет в DSPy примеры
    
    Args:
        dataset: HF Dataset
        input_fields: поля для входа (напр. ['source_text'])
        output_fields: поля для выхода (напр. ['title', 'extracted_chains'])
    
    Returns:
        List[dspy.Example]
    """
    examples = []
    
    for item in dataset:
        example_dict = {}
        
        # Входные поля
        for field in input_fields:
            if field in item:
                example_dict[field] = item[field]
        
        # Выходные поля
        for field in output_fields:
            if field in item:
                example_dict[field] = item[field]
        
        example = dspy.Example(**example_dict).with_inputs(*input_fields)
        examples.append(example)
    
    return examples


# === Примеры использования для разных модулей ===

def for_extraction_module(hf_username: str):
    """Датасет для оптимизации module_extraction"""
    ds = load_epistack_dataset(hf_username, split='train')
    
    # source_text -> extracted_chains
    trainset = to_dspy_examples(
        ds,
        input_fields=['source_text'],
        output_fields=['extracted_chains']
    )
    
    return trainset


def for_abstraction_module(hf_username: str):
    """Датасет для оптимизации module_abstraction"""
    ds = load_epistack_dataset(hf_username, split='train')
    
    # extracted_chains -> abstracted_chains
    trainset = to_dspy_examples(
        ds,
        input_fields=['source_text', 'extracted_chains'],
        output_fields=['abstracted_chains']
    )
    
    return trainset


def for_naming_module(hf_username: str):
    """Датасет для оптимизации module_naming"""
    ds = load_epistack_dataset(hf_username, split='train')
    
    # source_text -> title
    trainset = to_dspy_examples(
        ds,
        input_fields=['source_text'],
        output_fields=['title']
    )
    
    return trainset


def for_full_pipeline(hf_username: str):
    """Датасет для оптимизации всего пайплайна"""
    ds = load_epistack_dataset(hf_username, split='train')
    
    # source_text -> title, extracted_chains, abstracted_chains
    trainset = to_dspy_examples(
        ds,
        input_fields=['source_text'],
        output_fields=['title', 'extracted_chains', 'abstracted_chains']
    )
    
    return trainset


if __name__ == '__main__':
    HF_USERNAME = "your_username"  # <-- ИЗМЕНИ ЭТО
    
    # Пример загрузки для разных модулей
    print("\n=== Extraction Module ===")
    extraction_trainset = for_extraction_module(HF_USERNAME)
    print(f"Примеров: {len(extraction_trainset)}")
    print(f"Первый пример input: {extraction_trainset[0].source_text[:50]}...")
    
    print("\n=== Abstraction Module ===")
    abstraction_trainset = for_abstraction_module(HF_USERNAME)
    print(f"Примеров: {len(abstraction_trainset)}")
    
    print("\n=== Naming Module ===")
    naming_trainset = for_naming_module(HF_USERNAME)
    print(f"Примеров: {len(naming_trainset)}")
    
    print("\n=== Full Pipeline ===")
    full_trainset = for_full_pipeline(HF_USERNAME)
    print(f"Примеров: {len(full_trainset)}")

