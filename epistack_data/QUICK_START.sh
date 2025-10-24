#!/bin/bash
# Быстрый старт с датасетом epistack

echo "=== Epistack Dataset Quick Start ==="
echo ""

# 1. Установка зависимостей
echo "1. Установка зависимостей..."
pip install -q datasets huggingface_hub

# 2. Получение токена
echo ""
echo "2. Настройка HF токена"
echo "   Получи токен здесь: https://huggingface.co/settings/tokens"
echo ""
read -p "Введи HF токен: " HF_TOKEN
export HF_TOKEN=$HF_TOKEN

# 3. Введи username
echo ""
read -p "Введи твой HF username: " HF_USERNAME

# 4. Создание датасета
echo ""
echo "3. Создание датасета..."
cat > /tmp/create_dataset_tmp.py << EOF
import os
os.environ['HF_USERNAME'] = '$HF_USERNAME'
os.environ['HF_TOKEN'] = '$HF_TOKEN'

import sys
sys.path.insert(0, '.')

from datasets.create_hf_dataset import create_and_upload_dataset
create_and_upload_dataset('$HF_USERNAME', '$HF_TOKEN', private=True)
EOF

python /tmp/create_dataset_tmp.py

# 5. Готово
echo ""
echo "✅ Готово!"
echo ""
echo "Датасет доступен:"
echo "  https://huggingface.co/datasets/$HF_USERNAME/epistack-optimization"
echo ""
echo "Использование:"
echo "  python datasets/use_dataset.py"
echo "  python optimize_modules.py $HF_USERNAME"
echo ""

