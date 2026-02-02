"""
Validation service package for PDF content quality assurance.

Refactored from monolithic validation_service.py (1,128 lines) into focused modules:
- problem_detector.py: 13 problem detection patterns
- similarity_calculator.py: Similarity scoring algorithms
- content_normalizer.py: Text normalization utilities
- validation_orchestrator.py: Main ValidationService orchestration

Code refactoring: This package eliminates the Single Responsibility Principle violation
by splitting a 1,128-line file into 4 focused classes (~250 lines each).
"""
from .validation_orchestrator import ValidationService, ValidationResult, CrossValidationReport
from .problem_detector import ProblemDetector
from .similarity_calculator import SimilarityCalculator
from .content_normalizer import ContentNormalizer

__all__ = [
    'ValidationService',
    'ValidationResult',
    'CrossValidationReport',
    'ProblemDetector',
    'SimilarityCalculator',
    'ContentNormalizer',
]
