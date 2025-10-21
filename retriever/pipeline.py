"""
Complete few-shot test generation pipeline.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import logging

from .models import RetrievalResult
from .embedder import UniXcoderEmbedder
from .database import CodeExampleDatabase

logger = logging.getLogger(__name__)


class FewShotTestGenerationPipeline:
    """
    Complete pipeline for few-shot unit test generation using RAG.
    
    Pipeline stages:
    1. Query Processing: Embed the input focal method
    2. Retrieval: Find top-k similar examples from the database
    3. Prompt Construction: Build few-shot prompt with retrieved examples
    4. Validation: Check quality and relevance of retrieved examples
    """
    
    def __init__(
        self,
        embedder: UniXcoderEmbedder,
        database: CodeExampleDatabase,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the pipeline.
        
        Args:
            embedder: UniXcoder embedder instance
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
        Stage 1: Process the query focal method and generate embedding.
        
        Args:
            focal_method: The focal method to generate tests for
            
        Returns:
            Dictionary with query embedding and metadata
        """
        logger.info("Stage 1: Processing query...")
        
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
        Stage 2: Retrieve relevant examples using RAG.
        
        Args:
            focal_method: The focal method to generate tests for
            top_k: Number of examples to retrieve (overrides default)
            
        Returns:
            List of retrieved examples with similarity scores
        """
        logger.info("Stage 2: Retrieving examples...")
        
        k = top_k or self.top_k
        results = self.database.retrieve(focal_method, top_k=k)
        
        logger.info(f"Retrieved {len(results)} examples")
        for result in results:
            logger.info(f"  Rank {result.rank}: similarity={result.similarity_score:.4f}")
        
        return results
    
    def validate_retrieval(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Stage 3: Validate the quality of retrieved examples.
        
        Args:
            results: Retrieved examples
            
        Returns:
            Validation report with statistics and filtered results
        """
        logger.info("Stage 3: Validating retrieval results...")
        
        if not results:
            logger.warning("No results to validate")
            return {
                'valid_results': [],
                'filtered_count': 0,
                'avg_similarity': 0.0,
                'passed': False,
                'message': 'No examples retrieved'
            }
        
        # Filter by similarity threshold
        valid_results = [r for r in results if r.similarity_score >= self.similarity_threshold]
        
        avg_similarity = np.mean([r.similarity_score for r in results])
        valid_avg = np.mean([r.similarity_score for r in valid_results]) if valid_results else 0.0
        
        report = {
            'valid_results': valid_results,
            'total_retrieved': len(results),
            'filtered_count': len(results) - len(valid_results),
            'avg_similarity': float(avg_similarity),
            'valid_avg_similarity': float(valid_avg),
            'passed': len(valid_results) > 0,
            'similarity_threshold': self.similarity_threshold
        }
        
        logger.info(f"Validation complete: {len(valid_results)}/{len(results)} examples passed")
        logger.info(f"Average similarity: {avg_similarity:.4f}")
        
        if not report['passed']:
            logger.warning("No examples met the similarity threshold")
        
        return report
    
    def construct_few_shot_prompt(
        self,
        focal_method: str,
        retrieval_results: List[RetrievalResult],
        instruction: str = "Generate comprehensive unit tests for the following method."
    ) -> str:
        """
        Stage 4: Construct few-shot prompt with retrieved examples.
        
        Args:
            focal_method: The focal method to generate tests for
            retrieval_results: Retrieved examples to use as few-shot examples
            instruction: Instruction text for the generation model
            
        Returns:
            Complete few-shot prompt string
        """
        logger.info("Stage 4: Constructing few-shot prompt...")
        
        prompt_parts = [
            "# Task: Unit Test Generation",
            "",
            instruction,
            "",
            "# Examples:",
            ""
        ]
        
        # Add each retrieved example
        for i, result in enumerate(retrieval_results, 1):
            prompt_parts.extend([
                f"## Example {i} (Similarity: {result.similarity_score:.4f})",
                "",
                "### Focal Method:",
                "```",
                result.example.focal_method.strip(),
                "```",
                "",
                "### Unit Test:",
                "```",
                result.example.unit_test.strip(),
                "```",
                ""
            ])
        
        # Add the query
        prompt_parts.extend([
            "# Your Task:",
            "",
            "### Focal Method:",
            "```",
            focal_method.strip(),
            "```",
            "",
            "### Unit Test:",
            "```",
            "# Generate unit test here",
            "```"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Prompt constructed with {len(retrieval_results)} examples")
        logger.info(f"Total prompt length: {len(prompt)} characters")
        
        return prompt
    
    def run(
        self,
        focal_method: str,
        instruction: Optional[str] = None,
        return_prompt_only: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline for a given focal method.
        
        Args:
            focal_method: The focal method to generate tests for
            instruction: Custom instruction for the generation model
            return_prompt_only: If True, only return the prompt without validation
            
        Returns:
            Dictionary containing all pipeline outputs and metadata
        """
        logger.info("=" * 80)
        logger.info("Starting Few-Shot Test Generation Pipeline")
        logger.info("=" * 80)
        
        pipeline_result = {
            'focal_method': focal_method,
            'pipeline_stages': {}
        }
        
        try:
            # Stage 1: Process Query
            query_result = self.process_query(focal_method)
            pipeline_result['pipeline_stages']['query_processing'] = query_result
            
            # Stage 2: Retrieve Examples
            retrieval_results = self.retrieve_examples(focal_method)
            pipeline_result['pipeline_stages']['retrieval'] = {
                'results': retrieval_results,
                'count': len(retrieval_results)
            }
            
            # Stage 3: Validate Retrieval
            validation_report = self.validate_retrieval(retrieval_results)
            pipeline_result['pipeline_stages']['validation'] = validation_report
            
            # Decide whether to proceed
            if not validation_report['passed'] and not return_prompt_only:
                logger.warning("Validation failed. Consider adjusting threshold or adding more examples.")
                pipeline_result['status'] = 'failed_validation'
                pipeline_result['message'] = 'No examples met similarity threshold'
                return pipeline_result
            
            # Use valid results or all results based on validation
            examples_to_use = (validation_report['valid_results'] 
                             if validation_report['passed'] 
                             else retrieval_results)
            
            # Stage 4: Construct Prompt
            default_instruction = "Generate comprehensive unit tests for the following method."
            prompt = self.construct_few_shot_prompt(
                focal_method,
                examples_to_use,
                instruction or default_instruction
            )
            
            pipeline_result['pipeline_stages']['prompt_construction'] = {
                'prompt': prompt,
                'prompt_length': len(prompt),
                'examples_used': len(examples_to_use)
            }
            
            pipeline_result['status'] = 'success'
            pipeline_result['few_shot_prompt'] = prompt
            
            logger.info("=" * 80)
            logger.info("Pipeline completed successfully")
            logger.info("=" * 80)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_result['status'] = 'error'
            pipeline_result['error'] = str(e)
            return pipeline_result
