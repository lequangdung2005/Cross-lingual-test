"""
UniXcoder embedding model wrapper.
"""

import torch
import numpy as np
from typing import List
import logging
import tqdm
from src.core.base import BaseEmbedder

logger = logging.getLogger(__name__)


class UniXcoderEmbedder(BaseEmbedder):
    """
    Wrapper for UniXcoder model to generate code embeddings.
    
    UniXcoder is a unified cross-modal pre-trained model that supports 
    both code and text understanding.
    """
    
    def __init__(self, model_name: str = "microsoft/unixcoder-base", device: str = None):
        """
        Initialize UniXcoder embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self._model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the UniXcoder model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel
            logger.info(f"Loading UniXcoder model: {self._model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed(self, code: str, max_length: int = 512) -> np.ndarray:
        """
        Generate embedding for a code snippet.
        
        Args:
            code: Code snippet to embed
            max_length: Maximum token length
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (pooler output)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.squeeze()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_batch(self, codes: List[str], max_length: int = 1024, batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets.
        
        Args:
            codes: List of code snippets
            max_length: Maximum token length
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        embeddings = []

        for i in tqdm.tqdm(range(0, len(codes), batch_size)):
            batch = codes[i:i + batch_size]
            
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {i//batch_size}: {e}")
                raise
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.model is None:
            self._load_model()
        # UniXcoder base model has 768 dimensions
        return self.model.config.hidden_size
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        """Set the model name."""
        self._model_name = value
