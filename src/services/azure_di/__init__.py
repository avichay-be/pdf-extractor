"""
Azure Document Intelligence client components.

This package provides modular components for Azure Document Intelligence:
- Client: API communication
- TableMerger: Table merging logic across pages
- TableValidator: Numerical validation for table continuity
"""
from .client import AzureDocumentIntelligenceClient

__all__ = ["AzureDocumentIntelligenceClient"]
