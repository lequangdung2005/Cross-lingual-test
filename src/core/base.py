"""
Abstract base classes for retriever and database components.

This module defines the interfaces that all retriever and database implementations
must follow, enabling easy swapping of different RAG strategies (dense retrieval,
BM25, Graph RAG, hybrid, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
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
    def to_dict(self):
        return {
            "focal_method": self.focal_method,
            "unit_test": self.unit_test,
            "metadata": self.metadata
        }

@dataclass
class RetrievalResult:
    """Represents a retrieval result with similarity score."""
    example: CodeExample
    similarity_score: float
    rank: int
    retrieval_method: str = "unknown"  # e.g., "dense", "bm25", "graph", "hybrid"


class BaseEmbedder(ABC):
    """
    Abstract base class for code embedders.
    
    Any embedding model (UniXcoder, CodeBERT, GraphCodeBERT, etc.)
    should implement this interface.
    """
    
    @abstractmethod
    def embed(self, code: str, **kwargs) -> np.ndarray:
        """
        Generate embedding for a single code snippet.
        
        Args:
            code: Code snippet to embed
            **kwargs: Additional model-specific parameters
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def embed_batch(self, codes: List[str], **kwargs) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets.
        
        Args:
            codes: List of code snippets
            **kwargs: Additional model-specific parameters
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass


class BaseDatabase(ABC):
    """
    Abstract base class for code example databases.
    
    Different storage and retrieval strategies (vector DB, graph DB,
    hybrid, etc.) should implement this interface.
    """
    
    @abstractmethod
    def add_example(self, focal_method: str, unit_test: str, metadata: Optional[Dict] = None):
        """
        Add a single example to the database.
        
        Args:
            focal_method: Source code of the focal method
            unit_test: Corresponding unit test code
            metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    def add_examples_bulk(self, examples: List[Tuple[str, str, Optional[Dict]]]):
        """
        Add multiple examples at once.
        
        Args:
            examples: List of (focal_method, unit_test, metadata) tuples
        """
        pass
    
    @abstractmethod
    def build_index(self, **kwargs):
        """
        Build the retrieval index.
        
        Args:
            **kwargs: Implementation-specific parameters
        """
        pass
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve similar examples for a query.
        
        Args:
            query: Query code
            top_k: Number of results to return
            **kwargs: Implementation-specific parameters
            
        Returns:
            List of retrieval results ranked by similarity
        """
        pass
    
    @abstractmethod
    def save_index(self, path: str):
        """
        Save the database index to disk.
        
        Args:
            path: Path to save the index
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str):
        """
        Load the database index from disk.
        
        Args:
            path: Path to load the index from
        """
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of examples in the database."""
        pass
    
    @property
    @abstractmethod
    def retrieval_method(self) -> str:
        """Return the retrieval method identifier (e.g., 'dense', 'graph', 'hybrid')."""
        pass


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.
    
    This separates the retrieval logic from storage, allowing for
    complex retrieval strategies (e.g., hybrid retrieval combining
    multiple methods).
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        database: BaseDatabase,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve examples using this strategy.
        
        Args:
            query: Query code
            database: Database to retrieve from
            top_k: Number of results to return
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of retrieval results
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the retrieval method name."""
        pass


class BasePipeline(ABC):
    """
    Abstract base class for test generation pipelines.
    
    Different pipeline strategies can be implemented while maintaining
    a consistent interface.
    """
    
    @abstractmethod
    def run(
        self,
        focal_method: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            focal_method: The focal method to generate tests for
            **kwargs: Pipeline-specific parameters
            
        Returns:
            Dictionary with pipeline results
        """
        pass
    
    @abstractmethod
    def process_query(self, focal_method: str) -> Dict[str, Any]:
        """Process and validate the query."""
        pass
    
    @abstractmethod
    def retrieve_examples(self, focal_method: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve relevant examples."""
        pass
    

