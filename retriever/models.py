"""
Data models for the few-shot test generation pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class CodeExample:
    """Represents a focal method and its corresponding unit test."""
    focal_method: str
    unit_test: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"CodeExample(focal_length={len(self.focal_method)}, test_length={len(self.unit_test)})"


@dataclass
class RetrievalResult:
    """Represents a retrieval result with similarity score."""
    example: CodeExample
    similarity_score: float
    rank: int
