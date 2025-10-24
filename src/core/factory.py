"""
Factory functions for creating retriever components.

This module provides factory functions to easily create and switch between
different retrieval implementations (dense, BM25, Graph RAG, hybrid).
"""

from typing import Optional, Dict, Any
import logging

from .base import BaseEmbedder, BaseDatabase, BasePipeline

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating retrieval components."""
    
    _embedder_registry: Dict[str, type] = {}
    _database_registry: Dict[str, type] = {}
    _pipeline_registry: Dict[str, type] = {}
    
    @classmethod
    def register_embedder(cls, name: str, embedder_class: type):
        """
        Register a new embedder implementation.
        
        Args:
            name: Name identifier for the embedder
            embedder_class: Class implementing BaseEmbedder
        """
        cls._embedder_registry[name] = embedder_class
        logger.info(f"Registered embedder: {name}")
    
    @classmethod
    def register_database(cls, name: str, database_class: type):
        """
        Register a new database implementation.
        
        Args:
            name: Name identifier for the database
            database_class: Class implementing BaseDatabase
        """
        cls._database_registry[name] = database_class
        logger.info(f"Registered database: {name}")
    
    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: type):
        """
        Register a new pipeline implementation.
        
        Args:
            name: Name identifier for the pipeline
            pipeline_class: Class implementing BasePipeline
        """
        cls._pipeline_registry[name] = pipeline_class
        logger.info(f"Registered pipeline: {name}")
    
    @classmethod
    def create_embedder(
        cls,
        method: str = "unixcoder",
        **kwargs
    ) -> BaseEmbedder:
        """
        Create an embedder instance.
        
        Args:
            method: Embedder method name
                - "unixcoder": UniXcoder dense embeddings (default)
                - Future: "codebert", "graphcodebert", "custom"
            **kwargs: Additional arguments passed to embedder constructor
            
        Returns:
            Embedder instance
            
        Raises:
            ValueError: If method is not recognized
        """
        if method not in cls._embedder_registry:
            available = ', '.join(cls._embedder_registry.keys())
            raise ValueError(
                f"Unknown embedder method: '{method}'. "
                f"Available methods: {available}"
            )
        
        embedder_class = cls._embedder_registry[method]
        logger.info(f"Creating embedder: {method}")
        return embedder_class(**kwargs)
    
    @classmethod
    def create_database(
        cls,
        db_type: str = "dense_vector",
        embedder: Optional[BaseEmbedder] = None,
        **kwargs
    ) -> BaseDatabase:
        """
        Create a database instance.
        
        Args:
            db_type: Database type
                - "dense_vector": Dense vector similarity search (default)
                - Future: "graph", "bm25", "hybrid"
            embedder: Embedder instance (required for some database types)
            **kwargs: Additional arguments passed to database constructor
            
        Returns:
            Database instance
            
        Raises:
            ValueError: If db_type is not recognized
        """
        if db_type not in cls._database_registry:
            available = ', '.join(cls._database_registry.keys())
            raise ValueError(
                f"Unknown database type: '{db_type}'. "
                f"Available types: {available}"
            )
        
        database_class = cls._database_registry[db_type]
        logger.info(f"Creating database: {db_type}")
        
        # Pass embedder if required
        if embedder is not None:
            kwargs['embedder'] = embedder
        
        return database_class(**kwargs)
    
    @classmethod
    def create_pipeline(
        cls,
        pipeline_type: str = "few_shot",
        embedder: Optional[BaseEmbedder] = None,
        database: Optional[BaseDatabase] = None,
        **kwargs
    ) -> BasePipeline:
        """
        Create a pipeline instance.
        
        Args:
            pipeline_type: Pipeline type
                - "few_shot": Standard few-shot RAG pipeline (default)
                - Future: "graph_rag", "hybrid", "custom"
            embedder: Embedder instance
            database: Database instance
            **kwargs: Additional arguments passed to pipeline constructor
            
        Returns:
            Pipeline instance
            
        Raises:
            ValueError: If pipeline_type is not recognized
        """
        if pipeline_type not in cls._pipeline_registry:
            available = ', '.join(cls._pipeline_registry.keys())
            raise ValueError(
                f"Unknown pipeline type: '{pipeline_type}'. "
                f"Available types: {available}"
            )
        
        pipeline_class = cls._pipeline_registry[pipeline_type]
        logger.info(f"Creating pipeline: {pipeline_type}")
        
        # Pass components if provided
        if embedder is not None:
            kwargs['embedder'] = embedder
        if database is not None:
            kwargs['database'] = database
        
        return pipeline_class(**kwargs)
    
    @classmethod
    def create_full_pipeline(
        cls,
        method: str = "unixcoder",
        db_type: str = "dense_vector",
        pipeline_type: str = "few_shot",
        embedder_kwargs: Optional[Dict[str, Any]] = None,
        database_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None
    ) -> BasePipeline:
        """
        Create a complete pipeline with all components.
        
        This is a convenience method that creates embedder, database,
        and pipeline in one call.
        
        Args:
            method: Embedder method name
            db_type: Database type
            pipeline_type: Pipeline type
            embedder_kwargs: Arguments for embedder
            database_kwargs: Arguments for database
            pipeline_kwargs: Arguments for pipeline
            
        Returns:
            Fully initialized pipeline
            
        Example:
            >>> pipeline = RetrieverFactory.create_full_pipeline(
            ...     method="unixcoder",
            ...     db_type="dense_vector",
            ...     pipeline_type="few_shot",
            ...     pipeline_kwargs={"top_k": 5}
            ... )
        """
        embedder_kwargs = embedder_kwargs or {}
        database_kwargs = database_kwargs or {}
        pipeline_kwargs = pipeline_kwargs or {}
        
        # Create embedder
        embedder = cls.create_embedder(method, **embedder_kwargs)
        
        # Create database with embedder
        database = cls.create_database(db_type, embedder=embedder, **database_kwargs)
        
        # Create pipeline with embedder and database
        pipeline = cls.create_pipeline(
            pipeline_type,
            embedder=embedder,
            database=database,
            **pipeline_kwargs
        )
        
        logger.info(
            f"Created full pipeline: {method} + {db_type} + {pipeline_type}"
        )
        
        return pipeline
    
    @classmethod
    def list_available_methods(cls) -> Dict[str, list]:
        """
        List all available methods.
        
        Returns:
            Dictionary with keys: 'embedders', 'databases', 'pipelines'
        """
        return {
            'embedders': list(cls._embedder_registry.keys()),
            'databases': list(cls._database_registry.keys()),
            'pipelines': list(cls._pipeline_registry.keys())
        }


# Auto-register built-in implementations
def _register_builtin_implementations():
    """Register built-in retriever implementations."""
    try:
        from retrievers.dense.embedder import UniXcoderEmbedder
        RetrieverFactory.register_embedder("unixcoder", UniXcoderEmbedder)
    except ImportError:
        logger.warning("Could not register UniXcoderEmbedder")
    
    try:
        from retrievers.dense.database import CodeExampleDatabase
        RetrieverFactory.register_database("dense_vector", CodeExampleDatabase)
    except ImportError:
        logger.warning("Could not register CodeExampleDatabase")
    
    try:
        from retrievers.fewshot_pipeline import FewShotTestGenerationPipeline
        RetrieverFactory.register_pipeline("few_shot", FewShotTestGenerationPipeline)
    except ImportError:
        logger.warning("Could not register FewShotTestGenerationPipeline")


# Register on module import
_register_builtin_implementations()
