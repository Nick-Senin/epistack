"""
Конфигурация LLM для конкретного модуля

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
    Настройка LLM для модуля
    
    Args:
        model: Название модели (если None, берется из .env)
        api_base: API endpoint (если None, берется из .env)
        api_key: API ключ (если None, берется из .env)
        use_global_config: Использовать глобальный конфиг как fallback
        **kwargs: Дополнительные параметры для dspy.LM
        
    Returns:
        Сконфигурированная LM
    """
    dotenv.load_dotenv()
    
    # TODO: Настройте специфичные для модуля переменные окружения
    # Например, если нужна особая модель для этого модуля:
    # MODULE_MODEL = os.getenv("MODULE_TEMPLATE_MODEL")
    
    # Используем глобальный конфиг как fallback
    if use_global_config:
        try:
            from config.llm import configure_llm
            configure_llm()
            
            # Если нужны специфичные настройки, переопределяем
            if model or api_base or api_key:
                lm = dspy.LM(
                    model=model or os.getenv("OPENROUTER_MODEL", ""),
                    api_base=api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                    api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
                    **kwargs
                )
                dspy.configure(lm=lm)
                return lm
            
            return dspy.settings.lm
            
        except ImportError:
            print("⚠️  Глобальный конфиг не найден, используем локальные настройки")
    
    # TODO: Настройте параметры LLM специфично для вашего модуля
    lm = dspy.LM(
        model=model or os.getenv("OPENROUTER_MODEL", ""),
        api_base=api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
        **kwargs
    )
    
    dspy.configure(lm=lm)
    return lm


# TODO: Добавьте дополнительные функции конфигурации если нужно
# Например:
# def configure_judge_llm():
#     """Отдельная LM для метрики LLM as Judge"""
#     pass


