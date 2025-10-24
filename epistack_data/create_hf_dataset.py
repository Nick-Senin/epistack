"""
Создание и загрузка датасета на Hugging Face для оптимизации модулей epistack
"""
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd
import os


def create_dataset_structure():
    """Создает структуру датасета с примерами"""
    
    examples = [
        {
            'source_text': '''def process_user_data(raw_data):
    cleaned = remove_nulls(raw_data)
    validated = check_format(cleaned)
    return save_to_db(validated)''',
            
            'title': 'User Data Processing Pipeline',
            
            'extracted_chains': [
                {
                    'initial_state': 'raw_data',
                    'transformation': 'remove_nulls',
                    'result': 'cleaned'
                },
                {
                    'initial_state': 'cleaned',
                    'transformation': 'check_format',
                    'result': 'validated'
                },
                {
                    'initial_state': 'validated',
                    'transformation': 'save_to_db',
                    'result': 'stored_data'
                }
            ],
            
            'abstracted_chains': [
                {
                    'initial_state': 'unprocessed_input',
                    'transformation': 'sanitize',
                    'result': 'clean_input'
                },
                {
                    'initial_state': 'clean_input',
                    'transformation': 'validate',
                    'result': 'valid_input'
                },
                {
                    'initial_state': 'valid_input',
                    'transformation': 'persist',
                    'result': 'stored_output'
                }
            ]
        },
        
        {
            'source_text': '''class EmailSender:
    def send(self, recipients, message):
        formatted = self.format_html(message)
        attachments = self.prepare_files()
        return self.smtp_send(recipients, formatted, attachments)''',
            
            'title': 'Email Delivery System',
            
            'extracted_chains': [
                {
                    'initial_state': 'message',
                    'transformation': 'format_html',
                    'result': 'formatted'
                },
                {
                    'initial_state': 'files',
                    'transformation': 'prepare_files',
                    'result': 'attachments'
                },
                {
                    'initial_state': 'recipients, formatted, attachments',
                    'transformation': 'smtp_send',
                    'result': 'send_result'
                }
            ],
            
            'abstracted_chains': [
                {
                    'initial_state': 'raw_content',
                    'transformation': 'format',
                    'result': 'formatted_content'
                },
                {
                    'initial_state': 'raw_resources',
                    'transformation': 'prepare',
                    'result': 'ready_resources'
                },
                {
                    'initial_state': 'destination, content, resources',
                    'transformation': 'deliver',
                    'result': 'delivery_status'
                }
            ]
        },
        
        {
            'source_text': '''async def fetch_api_data(url, params):
    response = await http_get(url, params)
    parsed = json.loads(response.text)
    return transform_schema(parsed)''',
            
            'title': 'API Data Fetcher',
            
            'extracted_chains': [
                {
                    'initial_state': 'url, params',
                    'transformation': 'http_get',
                    'result': 'response'
                },
                {
                    'initial_state': 'response.text',
                    'transformation': 'json.loads',
                    'result': 'parsed'
                },
                {
                    'initial_state': 'parsed',
                    'transformation': 'transform_schema',
                    'result': 'transformed_data'
                }
            ],
            
            'abstracted_chains': [
                {
                    'initial_state': 'endpoint, parameters',
                    'transformation': 'request',
                    'result': 'raw_response'
                },
                {
                    'initial_state': 'raw_response',
                    'transformation': 'parse',
                    'result': 'structured_data'
                },
                {
                    'initial_state': 'structured_data',
                    'transformation': 'transform',
                    'result': 'normalized_output'
                }
            ]
        }
    ]
    
    return examples


def prepare_for_hf(examples):
    """Преобразует данные в формат для HF"""
    data = {
        'source_text': [],
        'title': [],
        'extracted_chains': [],
        'abstracted_chains': []
    }
    
    for ex in examples:
        data['source_text'].append(ex['source_text'])
        data['title'].append(ex['title'])
        data['extracted_chains'].append(ex['extracted_chains'])
        data['abstracted_chains'].append(ex['abstracted_chains'])
    
    return data


def create_and_upload_dataset(hf_username, hf_token=None, private=True):
    """
    Создает и загружает датасет на Hugging Face
    
    Args:
        hf_username: имя пользователя на HF
        hf_token: токен (или None для использования из окружения)
        private: приватный датасет или публичный
    """
    
    # Получаем примеры
    examples = create_dataset_structure()
    data = prepare_for_hf(examples)
    
    # Создаем датасет
    dataset = Dataset.from_dict(data)
    
    # Разделяем на train/test (80/20)
    dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Информация
    print(f"Создан датасет:")
    print(f"  Train: {len(dataset_dict['train'])} примеров")
    print(f"  Test: {len(dataset_dict['test'])} примеров")
    print(f"\nСтруктура:")
    print(dataset_dict['train'].features)
    print(f"\nПервый пример:")
    print(dataset_dict['train'][0])
    
    # Загрузка на HF
    repo_id = f"{hf_username}/epistack-optimization"
    
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    if hf_token:
        print(f"\nЗагрузка на Hugging Face: {repo_id}")
        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            token=hf_token
        )
        print(f"✅ Датасет загружен: https://huggingface.co/datasets/{repo_id}")
    else:
        print("\n⚠️  HF_TOKEN не найден. Датасет создан локально.")
        print("Для загрузки на HF:")
        print("  1. Получи токен: https://huggingface.co/settings/tokens")
        print("  2. export HF_TOKEN='your_token'")
        print(f"  3. Запусти снова")
        
        # Сохраняем локально
        dataset_dict.save_to_disk('datasets/epistack_optimization_local')
        print(f"\n💾 Сохранено локально: datasets/epistack_optimization_local")
    
    return dataset_dict


def load_dataset_example(hf_username, hf_token=None):
    """Пример загрузки датасета"""
    from datasets import load_dataset
    
    repo_id = f"{hf_username}/epistack-optimization"
    
    try:
        # Загрузка с HF
        ds = load_dataset(repo_id, token=hf_token)
        print(f"✅ Датасет загружен с HF: {repo_id}")
    except:
        # Загрузка локальной версии
        ds = load_dataset('datasets/epistack_optimization_local')
        print("✅ Загружен локальный датасет")
    
    return ds


if __name__ == '__main__':
    # Читаем из .env
    from dotenv import load_dotenv
    load_dotenv()
    
    HF_USERNAME = os.getenv('HF_USERNAME', 'Nick-Sen')
    
    # Создание и загрузка
    dataset = create_and_upload_dataset(
        hf_username=HF_USERNAME,
        private=False  # Публичный датасет
    )
    
    print("\n" + "="*60)
    print("Готово! Используй датасет:")
    print("="*60)
    print(f"""
from datasets import load_dataset

# Загрузка
ds = load_dataset('{HF_USERNAME}/epistack-optimization', token='your_token')

# В pandas
df = ds['train'].to_pandas()

# Для DSPy
import dspy
trainset = [dspy.Example(**x).with_inputs('source_text') for x in ds['train']]
""")

