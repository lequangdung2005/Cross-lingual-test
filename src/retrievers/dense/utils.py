"""
Utility functions for easy pipeline setup.
"""

from typing import Tuple

from .embedder import DensecoderEmbedder
from .database import CodeExampleDatabase
from ..fewshot_pipeline import FewShotTestGenerationPipeline


def create_pipeline(
    model_name: str = "microsoft/unixcoder-base",
    device: str = None,
    top_k: int = 5,
    similarity_threshold: float = 0.5
) -> Tuple[FewShotTestGenerationPipeline, CodeExampleDatabase]:
    """
    Create a complete pipeline with embedder and database.
    
    Args:
        model_name: Densecoder model name
        device: Device to run on
        top_k: Number of examples to retrieve
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (pipeline, database)
    """
    embedder = DensecoderEmbedder(model_name=model_name, device=device)
    database = CodeExampleDatabase(embedder)
    pipeline = FewShotTestGenerationPipeline(
        embedder=embedder,
        database=database,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    
    return pipeline, database
