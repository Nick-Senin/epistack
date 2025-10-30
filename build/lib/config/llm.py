import os
import dspy
import dotenv

def configure_llm():
    dotenv.load_dotenv()

    openrouter_api_base = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "")

    lm = dspy.LM(
        model=openrouter_model,
        api_base=openrouter_api_base,
        api_key=openrouter_api_key,
    )
    dspy.configure(lm=lm)
