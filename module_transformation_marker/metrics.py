"""
Метрики для оценки качества работы TransformationMarker
"""
import dspy
import re


class TransformationMarkerMetric:
    """
    Метрика для оценки качества выделения текста, связанного с преобразованиями.
    
    Проверяет:
    1. Корректность markdown синтаксиса (**...**)
    2. Наличие выделений в тексте
    3. Сохранность исходного текста (без лишних изменений)
    4. Если есть эталонный результат (expected_marked_text), сравнивает с ним
    """
    
    def __init__(self, use_similarity=True):
        """
        Args:
            use_similarity: Использовать ли сравнение с эталоном (если доступен)
        """
        self.use_similarity = use_similarity
    
    def __call__(self, example, pred, trace=None):
        """
        Оценка качества предсказания
        
        Args:
            example: Пример из датасета (должен содержать text, transformations,
                    опционально expected_marked_text)
            pred: Предсказание модуля (должно содержать marked_text)
            trace: Опциональный трейс выполнения
            
        Returns:
            dspy.Prediction: Оценка с score (0.0-1.0) и feedback
        """
        try:
            marked_text = pred.marked_text
            original_text = example.text if hasattr(example, 'text') else ""
            
            # Список для хранения найденных проблем
            feedback_parts = []
            score_components = []
            
            # 1. Проверка корректности markdown синтаксиса
            markdown_score = self._check_markdown_syntax(marked_text, feedback_parts)
            score_components.append(markdown_score)
            
            # 2. Проверка наличия выделений
            has_markings_score = self._check_has_markings(marked_text, feedback_parts)
            score_components.append(has_markings_score)
            
            # 3. Проверка сохранности исходного текста
            if original_text:
                preservation_score = self._check_text_preservation(
                    original_text, marked_text, feedback_parts
                )
                score_components.append(preservation_score)
            
            # 4. Сравнение с эталоном (если есть)
            if self.use_similarity and hasattr(example, 'expected_marked_text'):
                similarity_score = self._compare_with_expected(
                    marked_text, example.expected_marked_text, feedback_parts
                )
                score_components.append(similarity_score * 2)  # Больший вес для точности
            
            # Итоговая оценка - среднее взвешенное
            final_score = sum(score_components) / len(score_components)
            
            # Формируем финальный feedback
            if final_score >= 0.8:
                feedback = f"✅ Отличный результат (score: {final_score:.2f})"
            elif final_score >= 0.6:
                feedback = f"✓ Хороший результат (score: {final_score:.2f})"
            else:
                feedback = f"⚠️  Требует улучшения (score: {final_score:.2f})"
            
            if feedback_parts:
                feedback += "\n" + "\n".join(feedback_parts)
            
            return dspy.Prediction(
                score=final_score,
                feedback=feedback
            )
            
        except Exception as e:
            return dspy.Prediction(
                score=0.0,
                feedback=f"❌ Ошибка при оценке: {str(e)}"
            )
    
    def _check_markdown_syntax(self, text, feedback_parts):
        """Проверка корректности markdown синтаксиса"""
        # Ищем все вхождения ** 
        asterisks = text.count('**')
        
        # Должно быть чётное количество (открывающие и закрывающие)
        if asterisks % 2 != 0:
            feedback_parts.append("⚠️  Некорректный markdown: нечётное количество **")
            return 0.0
        
        # Проверяем что есть хотя бы одна пара
        if asterisks == 0:
            feedback_parts.append("ℹ️  Нет выделений жирным шрифтом")
            return 0.5
        
        return 1.0
    
    def _check_has_markings(self, text, feedback_parts):
        """Проверка наличия выделений"""
        pattern = r'\*\*(.+?)\*\*'
        markings = re.findall(pattern, text)
        
        if not markings:
            feedback_parts.append("⚠️  Нет выделенных фрагментов")
            return 0.0
        
        feedback_parts.append(f"✓ Найдено выделений: {len(markings)}")
        return 1.0
    
    def _check_text_preservation(self, original, marked, feedback_parts):
        """Проверка что исходный текст сохранён (без учёта **)"""
        # Удаляем все ** из marked текста
        cleaned_marked = marked.replace('**', '')
        
        # Сравниваем с оригиналом
        if cleaned_marked.strip() == original.strip():
            return 1.0
        
        # Вычисляем процент сходства
        similarity = self._text_similarity(original, cleaned_marked)
        
        if similarity < 0.9:
            feedback_parts.append(
                f"⚠️  Текст изменён (сходство: {similarity:.0%}). "
                "Должен быть сохранён исходный текст."
            )
        
        return similarity
    
    def _compare_with_expected(self, predicted, expected, feedback_parts):
        """Сравнение с эталонным результатом"""
        if predicted.strip() == expected.strip():
            feedback_parts.append("✅ Точное совпадение с эталоном")
            return 1.0
        
        # Вычисляем сходство
        similarity = self._text_similarity(predicted, expected)
        
        if similarity < 0.8:
            feedback_parts.append(
                f"ℹ️  Отличается от эталона (сходство: {similarity:.0%})"
            )
        
        return similarity
    
    def _text_similarity(self, text1, text2):
        """Простое вычисление сходства текстов (можно улучшить)"""
        # Простая метрика на основе общих символов
        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Используем отношение длин как приблизительную метрику
        common_ratio = min(len1, len2) / max(len1, len2)
        
        # Проверяем сколько символов совпадают на одинаковых позициях
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        position_similarity = matches / max(len1, len2)
        
        return (common_ratio + position_similarity) / 2

