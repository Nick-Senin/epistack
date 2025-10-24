import json
from config import configure_llm
from module_naming import RelationNamer


if __name__ == "__main__":
    configure_llm()

    text = """Исследователь применил метод дистилляции знаний, чтобы перевести сложную модель в более компактную,
    сохранив точность на валидации. Затем он автоматизировал предобработку данных с помощью пайплайна."""

    namer = RelationNamer()
    result = namer(text=text)
    print(json.dumps(result.toDict(), ensure_ascii=False, indent=2))

