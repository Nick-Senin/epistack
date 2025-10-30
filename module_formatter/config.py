"""
Конфигурация LLM для модуля форматирования текста

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
    Настройка LLM для модуля форматирования
    
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
    
    # Локальная конфигурация
    lm = dspy.LM(
        model=model or os.getenv("OPENROUTER_MODEL", ""),
        api_base=api_base or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
        **kwargs
    )
    
    dspy.configure(lm=lm)
    return lm
