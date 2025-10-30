from typing import List
import dspy
from dspy import InputField, OutputField, Signature
from data_models import StateTriple

# -------------------------------------------------------------------------------------------
# Модуль для анализа состояний и преобразований (State Transformation Analyzer)

class StateTransformationAnalyzerSignature(dspy.Signature):
    """
    Твоя задача для каждого решения или причинно-следственной связки содержащейся в переданном названии 
    выделить из исходного текста:
    - Начальное состояние
    - Преобразование  
    - Конечное состояние

    Формат ответа должен быть строго следующим (в виде массива):
    [{'initial_state': 'начальное состояние связки', 'transformation': 'преобразование связки', 'final_state': 'конечное состояние связки'}, ...]
    """

    relation_title = dspy.InputField(desc="Название связки с причинно-следственными отношениями")
    source_text = dspy.InputField(desc="Исходный текст для анализа состояний")

    state_analysis: List[StateTriple] = dspy.OutputField(
        desc="Явный массив троек [{'initial_state': ..., 'transformation': ..., 'final_state': ...}, ...] для каждой найденной связки"
    )



