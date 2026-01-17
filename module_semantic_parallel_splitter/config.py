"""
Конфигурация LLM для модуля семантической сегментации

Этот файл позволяет переопределить настройки LLM для данного модуля,
или использовать глобальные настройки из config/llm.py
"""
import os
import dspy
import dotenv


def configure_module_llm(
    model=None,
    api_base=None,
    api_key=None,
    use_global_config=True,
    **kwargs
):
    """
    Настройка LLM для модуля семантической сегментации

    Args:
        model: Название модели (если None, берется из env)
        api_base: API endpoint (если None, берется из env)
        api_key: API ключ (если None, берется из env)
        use_global_config: Использовать глобальный конфиг как fallback
        **kwargs: Дополнительные параметры для dspy.LM

    Returns:
        Сконфигурированная LM
    """
    dotenv.load_dotenv()

    if use_global_config and not any([model, api_base, api_key]):
        openrouter_model = os.getenv("OPENROUTER_MODEL", "").strip()
        if openrouter_model:
            from config.llm import configure_llm

            return configure_llm()

    # По умолчанию используем Cerebras (можно переопределить через параметры/переменные окружения)
    model = (model or os.getenv("CEREBRAS_MODEL") or "cerebras/gpt-oss-120b").strip()
    api_base = (api_base or os.getenv("CEREBRAS_API_BASE") or "https://api.cerebras.ai/v1").strip()
    api_key = (api_key or os.getenv("CEREBRAS_API_KEY") or "").strip()

    if not api_key:
        raise ValueError(
            "Не найден API ключ для LLM. Укажите `CEREBRAS_API_KEY` (и опционально `CEREBRAS_MODEL`, `CEREBRAS_API_BASE`) "
            "или задайте глобальные `OPENROUTER_*` переменные и вызовите `configure_module_llm(use_global_config=True)`."
        )

    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key=api_key,
        **kwargs
    )
    dspy.configure(lm=lm)
    print(f"🔧 Module LLM configured: {lm.model}")
    return lm
