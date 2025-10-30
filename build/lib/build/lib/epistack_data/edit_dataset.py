"""
Добавление новых примеров в датасет
"""
from datasets import load_dataset, Dataset
import os


def add_example(
    source_text: str,
    title: str,
    extracted_chains: list[dict],
    abstracted_chains: list[dict],
    hf_username: str,
    hf_token: str = None
):
    """
    Добавляет новый пример в датасет
    
    Args:
        source_text: исходный код
        title: название примера
        extracted_chains: список словарей с ключами initial_state, transformation, result
        abstracted_chains: абстрагированные связки
        hf_username: username на HF
        hf_token: HF токен
    """
    repo_id = f"{hf_username}/epistack-optimization"
    
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    # Загружаем существующий датасет
    try:
        ds = load_dataset(repo_id, token=hf_token)
        print(f"✅ Загружен датасет: {repo_id}")
    except:
        print("❌ Не удалось загрузить датасет")
        return
    
    # Новый пример
    new_example = {
        'source_text': source_text,
        'title': title,
        'extracted_chains': extracted_chains,
        'abstracted_chains': abstracted_chains
    }
    
    # Добавляем в train
    train_data = ds['train'].to_dict()
    for key in train_data:
        if key in new_example:
            train_data[key].append(new_example[key])
    
    new_train = Dataset.from_dict(train_data)
    ds['train'] = new_train
    
    print(f"✅ Добавлен новый пример: {title}")
    print(f"   Всего в train: {len(ds['train'])} примеров")
    
    # Загружаем обратно на HF
    ds.push_to_hub(repo_id, token=hf_token)
    print(f"✅ Датасет обновлен на HF")


def view_dataset(hf_username: str, hf_token: str = None, n: int = 5):
    """Просмотр первых n примеров"""
    repo_id = f"{hf_username}/epistack-optimization"
    
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    ds = load_dataset(repo_id, token=hf_token)
    
    print(f"Датасет: {repo_id}")
    print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}")
    print("\n" + "="*80)
    
    for i, example in enumerate(ds['train'].select(range(min(n, len(ds['train']))))):
        print(f"\n[{i+1}] {example['title']}")
        print(f"Код:\n{example['source_text'][:100]}...")
        print(f"Связок: {len(example['extracted_chains'])}")
        print("-"*80)


if __name__ == '__main__':
    HF_USERNAME = "your_username"  # <-- ИЗМЕНИ ЭТО
    
    # Пример добавления нового примера
    add_example(
        source_text='''def calculate_discount(price, user_tier):
    rate = get_tier_rate(user_tier)
    discount = apply_rate(price, rate)
    return round_to_cents(discount)''',
        
        title='Discount Calculator',
        
        extracted_chains=[
            {
                'initial_state': 'user_tier',
                'transformation': 'get_tier_rate',
                'result': 'rate'
            },
            {
                'initial_state': 'price, rate',
                'transformation': 'apply_rate',
                'result': 'discount'
            },
            {
                'initial_state': 'discount',
                'transformation': 'round_to_cents',
                'result': 'final_discount'
            }
        ],
        
        abstracted_chains=[
            {
                'initial_state': 'user_attribute',
                'transformation': 'lookup_parameter',
                'result': 'parameter_value'
            },
            {
                'initial_state': 'value, parameter',
                'transformation': 'compute',
                'result': 'computed_value'
            },
            {
                'initial_state': 'computed_value',
                'transformation': 'normalize',
                'result': 'normalized_output'
            }
        ],
        
        hf_username=HF_USERNAME
    )
    
    # Просмотр датасета
    view_dataset(HF_USERNAME, n=3)

