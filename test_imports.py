"""
Тест импортов после установки пакета epistack.
Запустите после: pip install -e .
"""

if __name__ == "__main__":
    try:
        # Тест основных импортов
        from epistack import configure_llm
        print("✓ configure_llm")
        
        from epistack import RelationExtractor, ExtractRelationsSig
        print("✓ RelationExtractor, ExtractRelationsSig")
        
        from epistack import RelationNamer, CausalRelationExtractorSignature
        print("✓ RelationNamer, CausalRelationExtractorSignature")
        
        from epistack import (
            NaiveATBAbstraction,
            EfficientATBAbstraction,
            AbstractATBSig,
            CritiqueSig,
            ReviseSig,
            AbstractionMetrics
        )
        print("✓ module_abstraction компоненты")
        
        from epistack import (
            ConcretizerWithReflection,
            ConcretizeFromATBSig,
            ConcretizationMetrics
        )
        print("✓ module_concretization компоненты")
        
        from epistack import safe_json_dict, jaccard_like
        print("✓ utils компоненты")
        
        # Тест прямых импортов модулей
        from epistack.module_abstraction import NaiveATBAbstraction
        print("✓ Прямой импорт: epistack.module_abstraction")
        
        from epistack.module_naming import RelationNamer
        print("✓ Прямой импорт: epistack.module_naming")
        
        from epistack.module_extraction_by_name import RelationExtractor
        print("✓ Прямой импорт: epistack.module_extraction_by_name")
        
        from epistack.module_concretization import ConcretizerWithReflection
        print("✓ Прямой импорт: epistack.module_concretization")
        
        from epistack.config import configure_llm
        print("✓ Прямой импорт: epistack.config")
        
        from epistack.utils import safe_json_dict, jaccard_like
        print("✓ Прямой импорт: epistack.utils")
        
        print("\n✅ Все импорты успешны!")
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("\nУстановите пакет командой: pip install -e .")

