"""
Few-shot retrieval pipeline.

This pipeline focuses on retrieving relevant code examples
for a given focal method using RAG (Retrieval-Augmented Generation).
"""

from typing import List, Optional, Dict, Any
import numpy as np
import logging
from difflib import SequenceMatcher
from collections import deque

from tree_sitter import Language, Parser
import tree_sitter_go as ts_go
import tree_sitter_julia as ts_julia
import tree_sitter_rust as ts_rust
import tree_sitter_java as ts_java

from src.core.base import RetrievalResult, BasePipeline, BaseEmbedder, BaseDatabase

logger = logging.getLogger(__name__)


# Initialize tree-sitter parsers
JULIA_LANG = Language(ts_julia.language())
JULIA_PARSER = Parser(JULIA_LANG)

RUST_LANG = Language(ts_rust.language())
RUST_PARSER = Parser(RUST_LANG)

GO_LANG = Language(ts_go.language())
GO_PARSER = Parser(GO_LANG)

JAVA_LANG = Language(ts_java.language())
JAVA_PARSER = Parser(JAVA_LANG)


def _find_nodes_by_type(node, type_names):
    """
    Find all descendant nodes matching any of the given type names.
    
    Args:
        node: Root node to search from
        type_names: List of node type names to match
        
    Returns:
        List of matching nodes
    """
    result = []
    queue = deque([node])
    
    while queue:
        current = queue.popleft()
        if current.type in type_names:
            result.append(current)
        for child in current.children:
            queue.append(child)
    
    return result


def _get_direct_child_by_type(node, type_name):
    """
    Get direct children nodes of a specific type.
    
    Args:
        node: Parent node
        type_name: Type name to match
        
    Returns:
        List of matching direct children
    """
    return [child for child in node.children if child.type == type_name]


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
        similarity_threshold: float = 0.5,
        rerank_weights: Optional[Dict[str, float]] = None,
        rerank_pool_size: int = 50
    ):
        """
        Initialize the pipeline.
        
        Args:
            embedder: Embedder instance (e.g., UniXcoderEmbedder)
            database: Code example database
            top_k: Number of examples to return after reranking
            similarity_threshold: Minimum similarity score for valid examples
            rerank_weights: Weights for reranking scores
                - 'embedding': Weight for embedding similarity (default: 0.6)
                - 'method_name': Weight for method name similarity (default: 0.2)
                - 'signature': Weight for signature similarity (default: 0.2)
            rerank_pool_size: Number of candidates to retrieve before reranking (default: 50)
        """
        self.embedder = embedder
        self.database = database
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rerank_pool_size = rerank_pool_size
        
        # Default reranking weights
        default_weights = {
            'embedding': 0.6,
            'method_name': 0.2,
            'signature': 0.2
        }
        self.rerank_weights = rerank_weights or default_weights
        
        # Normalize weights to sum to 1.0
        total = sum(self.rerank_weights.values())
        self.rerank_weights = {k: v/total for k, v in self.rerank_weights.items()}
        
        logger.info(f"Reranking weights: {self.rerank_weights}")
        logger.info(f"Rerank pool size: {self.rerank_pool_size}, Final top_k: {self.top_k}")
    
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
    
    def _detect_language(self, code: str) -> Optional[str]:
        """
        Detect programming language from code snippet.
        
        Args:
            code: Code snippet
            
        Returns:
            Language name ('rust', 'go', 'julia', 'java') or None
        """
        # Try parsing with each language
        parsers = [
            ('rust', RUST_PARSER),
            ('go', GO_PARSER),
            ('julia', JULIA_PARSER),
            ('java', JAVA_PARSER)
        ]
        
        for lang, parser in parsers:
            try:
                tree = parser.parse(bytes(code, 'utf8'))
                if tree.root_node.has_error:
                    continue
                # Check if there are actual function/method definitions
                function_nodes = self._get_function_nodes(tree.root_node, lang)
                if function_nodes:
                    return lang
            except Exception:
                continue
        
        # Fallback: look for language-specific keywords
        if 'fn ' in code or 'impl ' in code or 'pub fn' in code:
            return 'rust'
        elif 'func ' in code or 'package ' in code:
            return 'go'
        elif 'function ' in code and 'end' in code:
            return 'julia'
        elif ('public ' in code or 'private ' in code) and ('class ' in code or 'void ' in code or 'int ' in code):
            return 'java'
        
        return None
    
    def _get_function_nodes(self, root_node, lang: str):
        """
        Get function/method definition nodes for a specific language.
        
        Args:
            root_node: Root AST node
            lang: Programming language
            
        Returns:
            List of function definition nodes
        """
        if lang == 'rust':
            return _find_nodes_by_type(root_node, ['function_item', 'function_signature_item'])
        elif lang == 'go':
            return _find_nodes_by_type(root_node, ['function_declaration', 'method_declaration'])
        elif lang == 'julia':
            return _find_nodes_by_type(root_node, ['function_definition', 'short_function_definition'])
        elif lang == 'java':
            return _find_nodes_by_type(root_node, ['method_declaration'])
        return []
    
    def _extract_method_name(self, code: str) -> str:
        """
        Extract method/function name from code snippet using tree-sitter.
        
        Args:
            code: Code snippet
            
        Returns:
            Method name or empty string if not found
        """
        # Detect language
        lang = self._detect_language(code)
        if not lang:
            return ""
        
        # Parse code
        parser_map = {
            'rust': RUST_PARSER,
            'go': GO_PARSER,
            'julia': JULIA_PARSER,
            'java': JAVA_PARSER
        }
        
        parser = parser_map.get(lang)
        if not parser:
            return ""
        
        try:
            tree = parser.parse(bytes(code, 'utf8'))
            root = tree.root_node
            
            # Get function nodes
            function_nodes = self._get_function_nodes(root, lang)
            if not function_nodes:
                return ""
            
            # Extract name from first function
            func_node = function_nodes[0]
            
            if lang == 'rust':
                # Look for identifier after 'fn' keyword
                identifiers = _get_direct_child_by_type(func_node, 'identifier')
                if identifiers:
                    return identifiers[0].text.decode('utf8')
                    
            elif lang == 'go':
                # Function name is an identifier child
                identifiers = _get_direct_child_by_type(func_node, 'identifier')
                if identifiers:
                    return identifiers[0].text.decode('utf8')
                # For methods, look for field_identifier
                field_ids = _get_direct_child_by_type(func_node, 'field_identifier')
                if field_ids:
                    return field_ids[0].text.decode('utf8')
                    
            elif lang == 'julia':
                # Look for identifier in function definition
                identifiers = _find_nodes_by_type(func_node, ['identifier'])
                if identifiers:
                    return identifiers[0].text.decode('utf8')
            
            elif lang == 'java':
                # Java method name is an identifier child
                identifiers = _get_direct_child_by_type(func_node, 'identifier')
                if identifiers:
                    return identifiers[0].text.decode('utf8')
            
        except Exception as e:
            logger.debug(f"Failed to extract method name: {e}")
        
        return ""
    
    def _extract_signature(self, code: str) -> str:
        """
        Extract number of parameters from function signature using tree-sitter.
        Returns a string representation of the parameter count for similarity comparison.
        
        Args:
            code: Code snippet
            
        Returns:
            String with parameter count (e.g., "3_params") or empty string if not found
        """
        # Detect language
        lang = self._detect_language(code)
        if not lang:
            return ""
        
        # Parse code
        parser_map = {
            'rust': RUST_PARSER,
            'go': GO_PARSER,
            'julia': JULIA_PARSER,
            'java': JAVA_PARSER
        }
        
        parser = parser_map.get(lang)
        if not parser:
            return ""
        
        try:
            tree = parser.parse(bytes(code, 'utf8'))
            root = tree.root_node
            
            # Get function nodes
            function_nodes = self._get_function_nodes(root, lang)
            if not function_nodes:
                return ""
            
            # Extract parameter count from first function
            func_node = function_nodes[0]
            param_count = 0
            
            if lang == 'rust':
                # Look for parameters node
                params = _get_direct_child_by_type(func_node, 'parameters')
                if params:
                    # Count parameter nodes
                    param_nodes = _get_direct_child_by_type(params[0], 'parameter')
                    param_count = len(param_nodes)
                    
            elif lang == 'go':
                # Look for parameter_list node
                params = _get_direct_child_by_type(func_node, 'parameter_list')
                if params:
                    # Count parameter_declaration nodes
                    param_nodes = _get_direct_child_by_type(params[0], 'parameter_declaration')
                    param_count = len(param_nodes)
                    
            elif lang == 'julia':
                # Look for parameter_list or typed_parameter
                # Julia uses argument_list in the signature node
                signature_nodes = _find_nodes_by_type(func_node, ['signature'])
                if signature_nodes:
                    arg_list = _find_nodes_by_type(signature_nodes[0], ['argument_list'])
                    if arg_list:
                        # Count typed_expression nodes (parameters)
                        typed_exprs = _get_direct_child_by_type(arg_list[0], 'typed_expression')
                        param_count = len(typed_exprs)
                        # Also count plain identifiers (untyped parameters)
                        identifiers = _get_direct_child_by_type(arg_list[0], 'identifier')
                        param_count += len(identifiers)
            
            elif lang == 'java':
                # Look for formal_parameters node
                params = _get_direct_child_by_type(func_node, 'formal_parameters')
                if params:
                    # Count formal_parameter nodes
                    param_nodes = _get_direct_child_by_type(params[0], 'formal_parameter')
                    param_count = len(param_nodes)
            
            # Return normalized representation
            return f"{param_count}_params"
            
        except Exception as e:
            logger.debug(f"Failed to extract signature: {e}")
        
        # Fallback: return empty
        return ""    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using SequenceMatcher.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0
        
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _calculate_param_similarity(self, sig1: str, sig2: str) -> float:
        """
        Calculate similarity based on parameter count.
        Format expected: "N_params" where N is the number of parameters.
        
        Args:
            sig1: First signature (parameter count string)
            sig2: Second signature (parameter count string)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not sig1 or not sig2:
            return 0.0
        
        try:
            # Extract parameter counts
            count1 = int(sig1.split('_')[0])
            count2 = int(sig2.split('_')[0])
            
            # Exact match gets 1.0
            if count1 == count2:
                return 1.0
            
            # Close matches get partial credit
            # Similarity decreases as difference increases
            diff = abs(count1 - count2)
            max_count = max(count1, count2)
            
            if max_count == 0:
                return 1.0  # Both have 0 parameters
            
            # Linear decay: 1.0 at diff=0, decreasing by 0.2 per parameter difference
            similarity = max(0.0, 1.0 - (diff * 0.2))
            return similarity
            
        except (ValueError, IndexError):
            # Fallback to string similarity if format is unexpected
            return SequenceMatcher(None, sig1.lower(), sig2.lower()).ratio()
    
    def rerank_results(
        self, 
        focal_method: str, 
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using multiple similarity metrics.
        
        Combines:
        - Embedding similarity (from initial retrieval)
        - Method name similarity (string matching)
        - Signature similarity (parameter matching)
        
        Args:
            focal_method: The query focal method
            results: Initial retrieval results
            top_k: Number of top results to return after reranking (uses self.top_k if None)
            
        Returns:
            Reranked list of RetrievalResult objects with updated similarity scores
        """
        logger.info("Reranking retrieval results...")
        
        if not results:
            logger.warning("No results to rerank")
            return []
        
        k = top_k or self.top_k
        
        # Extract query features
        query_method_name = self._extract_method_name(focal_method)
        query_signature = self._extract_signature(focal_method)

        
        reranked_results = []
        
        for result in results:
            candidate_code = result.example.focal_method
            
            # Get embedding similarity (original score)
            embedding_sim = result.similarity_score
            
            # Calculate method name similarity
            candidate_method_name = self._extract_method_name(candidate_code)
            method_name_sim = self._calculate_string_similarity(query_method_name, candidate_method_name)
            
            # Calculate signature similarity (based on parameter count)
            candidate_signature = self._extract_signature(candidate_code)
            signature_sim = self._calculate_param_similarity(query_signature, candidate_signature)
            
            # Compute weighted combined score
            combined_score = (
                self.rerank_weights['embedding'] * embedding_sim +
                self.rerank_weights['method_name'] * method_name_sim +
                self.rerank_weights['signature'] * signature_sim
            )
            
            # Create new result with updated score
            reranked_result = RetrievalResult(
                example=result.example,
                similarity_score=combined_score,
                rank=result.rank,  # Will be updated after sorting
                retrieval_method=f"{result.retrieval_method}_reranked"
            )
            
            # Store component scores in metadata for debugging
            if reranked_result.example.metadata is None:
                reranked_result.example.metadata = {}
            
            reranked_result.example.metadata.update({
                'rerank_scores': {
                    'embedding': float(embedding_sim),
                    'method_name': float(method_name_sim),
                    'signature': float(signature_sim),
                    'combined': float(combined_score)
                },
                'extracted_features': {
                    'method_name': candidate_method_name,
                    'signature': candidate_signature
                }
            })
            
            reranked_results.append(reranked_result)
        
        # Sort by combined score (descending)
        reranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for idx, result in enumerate(reranked_results, start=1):
            result.rank = idx
        
        # Filter by similarity threshold
        valid_results = [r for r in reranked_results if r.similarity_score >= self.similarity_threshold]
        
        # Return top_k results
        final_results = valid_results[:k]
        
        logger.info(f"Reranked {len(results)} candidates -> {len(valid_results)} valid -> returning top {len(final_results)}")
        
        return final_results
    
    def validate_retrieval(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Deprecated: Use rerank_results instead.
        
        This method is kept for backward compatibility.
        """
        logger.warning("validate_retrieval is deprecated, using rerank_results instead")
        return results
    
    def run(
        self,
        focal_method: str,
        top_k: Optional[int] = None,
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete retrieval pipeline for a given focal method.
        
        Args:
            focal_method: The focal method to retrieve examples for
            top_k: Number of examples to return after reranking (overrides default)
            enable_reranking: If True, apply reranking with multiple similarity metrics
            
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
            # Stage 1: Retrieve larger pool of candidates
            # Use rerank_pool_size for initial retrieval if reranking is enabled
            if enable_reranking:
                retrieval_k = self.rerank_pool_size
                logger.info(f"Retrieving {retrieval_k} candidates for reranking")
            else:
                retrieval_k = top_k or self.top_k
                logger.info(f"Retrieving {retrieval_k} candidates (no reranking)")
            
            retrieval_results = self.retrieve_examples(focal_method, retrieval_k)
            
            # Stage 2: Rerank results and get top_k (optional)
            if enable_reranking:
                reranked_results = self.rerank_results(focal_method, retrieval_results, top_k)
                pipeline_result['results'] = reranked_results
            else:
                # Filter by threshold without reranking
                filtered_results = [r for r in retrieval_results if r.similarity_score >= self.similarity_threshold]
                # Limit to top_k
                k = top_k or self.top_k
                pipeline_result['results'] = filtered_results[:k]
            
            pipeline_result['status'] = 'success'
            
            logger.info("=" * 80)
            logger.info("Retrieval pipeline completed successfully")
            logger.info(f"Final results: {len(pipeline_result['results'])} examples")
            logger.info("=" * 80)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_result['status'] = 'error'
            pipeline_result['error'] = str(e)
            return pipeline_result
    
    