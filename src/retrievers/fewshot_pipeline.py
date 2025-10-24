"""
Few-shot retrieval pipeline.

This pipeline focuses on retrieving relevant code examples
for a given focal method using RAG (Retrieval-Augmented Generation).
"""

from typing import List, Optional, Dict, Any
import numpy as np
import logging

from src.core.base import RetrievalResult, BasePipeline, BaseEmbedder, BaseDatabase

logger = logging.getLogger(__name__)


class FewShotTestGenerationPipeline(BasePipeline):
    """
    Pipeline for retrieving relevant code examples using RAG.
    
    Pipeline stages:
    1. Query Processing: Embed the input focal method
    2. Retrieval: Find top-k similar examples from the database
    3. Validation: Check quality and relevance of retrieved examples
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        database: BaseDatabase,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the pipeline.
        
        Args:
            embedder: Embedder instance (e.g., UniXcoderEmbedder)
            database: Code example database
            top_k: Number of examples to retrieve
            similarity_threshold: Minimum similarity score for valid examples
        """
        self.embedder = embedder
        self.database = database
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def process_query(self, focal_method: str) -> Dict[str, Any]:
        """
        Process the query focal method and generate embedding.
        
        Args:
            focal_method: The focal method to retrieve examples for
            
        Returns:
            Dictionary with query embedding and metadata
        """
        logger.info("Processing query...")
        
        query_embedding = self.embedder.embed(focal_method)
        
        result = {
            'focal_method': focal_method,
            'embedding': query_embedding,
            'embedding_shape': query_embedding.shape,
            'status': 'success'
        }
        
        logger.info(f"Query processed. Embedding shape: {query_embedding.shape}")
        return result
    
    def retrieve_examples(
        self, 
        focal_method: str, 
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant examples using RAG.
        
        Args:
            focal_method: The focal method to retrieve examples for
            top_k: Number of examples to retrieve (overrides default)
            
        Returns:
            List of retrieved examples with similarity scores
        """
        logger.info("Retrieving examples...")
        
        k = top_k or self.top_k
        results = self.database.retrieve(focal_method, top_k=k)
        
        logger.info(f"Retrieved {len(results)} examples")
        for result in results:
            logger.info(f"  Rank {result.rank}: similarity={result.similarity_score:.4f}")
        
        return results
    
    def validate_retrieval(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Validate the quality of retrieved examples.
        
        Args:
            results: Retrieved examples
            
        Returns:
            Validation report with statistics and filtered results
        """
        logger.info("Validating retrieval results...")
        
        if not results:
            logger.warning("No results to validate")
            return []
        
        # Filter by similarity threshold
        valid_results = [r for r in results if r.similarity_score >= self.similarity_threshold]
        
        # avg_similarity = np.mean([r.similarity_score for r in results])
        # valid_avg = np.mean([r.similarity_score for r in valid_results]) if valid_results else 0.0
        
        report = valid_results
        
        logger.info(f"Validation complete: {len(valid_results)}/{len(results)} examples passed")
        # logger.info(f"Average similarity: {avg_similarity:.4f}")
        # logger.info(f"Valid average similarity: {valid_avg:.4f}")

        if not report:
            logger.warning("No examples met the similarity threshold")
        
        return report
    
    def run(
        self,
        focal_method: str,
        top_k: Optional[int] = None,
        return_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete retrieval pipeline for a given focal method.
        
        Args:
            focal_method: The focal method to retrieve examples for
            top_k: Number of examples to retrieve (overrides default)
            return_validation: If True, include validation report
            
        Returns:
            Dictionary containing retrieval results and metadata
        """
        logger.info("=" * 80)
        logger.info("Starting Few-Shot Retrieval Pipeline")
        logger.info("=" * 80)
        
        pipeline_result = {
            'focal_method': focal_method,
        }
        
        try:

            # Stage 1: Retrieve Examples
            retrieval_results = self.retrieve_examples(focal_method, top_k)
            # pipeline_result['pipeline_stages']['retrieval'] = {
            #     'results': retrieval_results,
            #     'count': len(retrieval_results)
            # }
            # pipeline_result['retrieval_results'] = retrieval_results
            
            # Stage 2: Validate Retrieval (optional)
            if return_validation:
                validation_report = self.validate_retrieval(retrieval_results)
                # pipeline_result['pipeline_stages']['validation'] = validation_report
                pipeline_result['results'] = validation_report
            else:
                pipeline_result['results'] = retrieval_results
            pipeline_result['status'] = 'success'
            
            logger.info("=" * 80)
            logger.info("Retrieval pipeline completed successfully")
            logger.info(f"Retrieved {len(retrieval_results)} examples")
            logger.info("=" * 80)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_result['status'] = 'error'
            pipeline_result['error'] = str(e)
            return pipeline_result
    
    