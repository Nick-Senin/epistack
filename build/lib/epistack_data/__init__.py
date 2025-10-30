"""
Epistack Dataset Management
"""
from .use_dataset import (
    load_epistack_dataset,
    to_dspy_examples,
    for_extraction_module,
    for_abstraction_module,
    for_naming_module,
    for_full_pipeline
)

__all__ = [
    'load_epistack_dataset',
    'to_dspy_examples',
    'for_extraction_module',
    'for_abstraction_module',
    'for_naming_module',
    'for_full_pipeline'
]

