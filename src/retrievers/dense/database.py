"""
Code example database with embedding-based retrieval.
"""
import tqdm
import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

from src.core.base import CodeExample, RetrievalResult, BaseDatabase, BaseEmbedder

logger = logging.getLogger(__name__)


class CodeExampleDatabase(BaseDatabase):
    """
    Database of code examples with their embeddings for retrieval.
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize the database.
        
        Args:
            embedder: Embedder instance (e.g., UniXcoderEmbedder)
        """
        self.embedder = embedder
        self.examples: List[CodeExample] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def add_example(self, focal_method: str, unit_test: str, metadata: Optional[Dict] = None):
        """
        Add a focal method and unit test pair to the database.
        
        Args:
            focal_method: Source code of the focal method
            unit_test: Corresponding unit test code
            metadata: Additional metadata (e.g., language, project, etc.)
        """
        example = CodeExample(
            focal_method=focal_method,
            unit_test=unit_test,
            metadata=metadata or {}
        )
        self.examples.append(example)
        logger.info(f"Added example {len(self.examples)}: {example}")
    
    def add_examples_bulk(self, examples: List[Tuple[str, str, Optional[Dict]]]):
        """
        Add multiple examples at once.
        
        Args:
            examples: List of (focal_method, unit_test, metadata) tuples
        """
        for item in examples:
            if hasattr(item, "focal_method"):
                focal = getattr(item, "focal_method", None)
                test  = getattr(item, "unit_test", None)
                meta  = getattr(item, "metadata", None)
                self.add_example(focal, test, meta)

    
    def build_index(self, batch_size: int = 8):
        """
        Build the embedding index for all examples.
        This should be called after adding all examples.
        
        Args:
            batch_size: Batch size for embedding generation
        """
        if not self.examples:
            logger.warning("No examples to index")
            return
        
        logger.info(f"Building index for {len(self.examples)} examples...")
        
        # Generate embeddings for all focal methods
        focal_methods = [ex.focal_method for ex in self.examples]
        self.embeddings = self.embedder.embed_batch(focal_methods, batch_size=batch_size)
        
        # Store embeddings in examples
        for i, example in enumerate(self.examples):
            example.embedding = self.embeddings[i]
        
        logger.info(f"Index built successfully. Embedding shape: {self.embeddings.shape}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_metric: str = "cosine"
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k most similar examples for a query.
        
        Args:
            query: Query code snippet (focal method to generate tests for)
            top_k: Number of examples to retrieve
            similarity_metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        if self.embeddings is None or len(self.examples) == 0:
            logger.error("Index not built or no examples available")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Calculate similarities
        if similarity_metric == "cosine":
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
        elif similarity_metric == "euclidean":
            similarities = -self._euclidean_distance(query_embedding, self.embeddings)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Get top-k indices
        top_k = min(top_k, len(self.examples))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Create results
        results = [
            RetrievalResult(
                example=self.examples[idx],
                similarity_score=float(similarities[idx]),
                rank=rank + 1
            )
            for rank, idx in enumerate(top_indices)
        ]
        
        logger.info(f"Retrieved {len(results)} examples for query")
        return results
    
    @staticmethod
    def _cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and database embeddings."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        return np.dot(database_norm, query_norm)
    
    @staticmethod
    def _euclidean_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance between query and database embeddings."""
        return np.linalg.norm(database - query, axis=1)
    
    def save_index(self, path: str):
        """
        Save the database and embeddings to disk.
        
        Args:
            path: Path to save the database
        """
        import pickle
        
        save_data = {
            'examples': self.examples,
            'embeddings': self.embeddings
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Database saved to {path}")
    
    def load_index(self, path: str):
        """
        Load the database and embeddings from disk.
        
        Args:
            path: Path to load the database from
        """
        import pickle
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.examples = save_data['examples']
        self.embeddings = save_data['embeddings']
        
        logger.info(f"Database loaded from {path} with {len(self.examples)} examples")
    
    @property
    def size(self) -> int:
        """Get the number of examples in the database."""
        return len(self.examples)
    
    @property
    def retrieval_method(self) -> str:
        """Get the retrieval method name."""
        return "dense_vector_similarity"
