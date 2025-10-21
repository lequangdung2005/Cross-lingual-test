"""
UniXcoder-based Retrieval System for Few-Shot Unit Test Generation

DEPRECATED: This module is kept for backward compatibility.
Please import from the retriever package directly:

    from retriever import (
        UniXcoderEmbedder,
        CodeExampleDatabase,
        FewShotTestGenerationPipeline,
        CodeExample,
        RetrievalResult,
        create_pipeline
    )

The code has been refactored into separate modules for better maintainability:
- retriever.models: Data models (CodeExample, RetrievalResult)
- retriever.embedder: UniXcoder embedding functionality
- retriever.database: Code example database with retrieval
- retriever.pipeline: Complete pipeline implementation
- retriever.utils: Helper functions
"""

# Import from refactored modules for backward compatibility
from .models import CodeExample, RetrievalResult
from .embedder import UniXcoderEmbedder
from .database import CodeExampleDatabase
from .pipeline import FewShotTestGenerationPipeline
from .utils import create_pipeline

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Re-export all classes for backward compatibility
__all__ = [
    'CodeExample',
    'RetrievalResult',
    'UniXcoderEmbedder',
    'CodeExampleDatabase',
    'FewShotTestGenerationPipeline',
    'create_pipeline'
]




if __name__ == "__main__":
    # Example usage demonstration
    print("Few-Shot Unit Test Generation Pipeline")
    print("=" * 80)
    print("\nThis module provides a RAG-based pipeline for unit test generation.")
    print("\nKey components:")
    print("  1. UniXcoderEmbedder - Generate code embeddings")
    print("  2. CodeExampleDatabase - Store and retrieve example pairs")
    print("  3. FewShotTestGenerationPipeline - Complete end-to-end pipeline")
    print("\nSee documentation for usage examples.")
