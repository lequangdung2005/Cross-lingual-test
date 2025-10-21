"""
Few-Shot Unit Test Generation Pipeline with RAG

A modular retrieval-augmented generation pipeline using UniXcoder embeddings.
"""

from .embedder import UniXcoderEmbedder
from .database import CodeExampleDatabase
from .pipeline import FewShotTestGenerationPipeline
from .models import CodeExample, RetrievalResult
from .utils import create_pipeline

__version__ = "1.0.0"

__all__ = [
    "UniXcoderEmbedder",
    "CodeExampleDatabase",
    "FewShotTestGenerationPipeline",
    "CodeExample",
    "RetrievalResult",
    "create_pipeline",
]
