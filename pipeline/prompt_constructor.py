"""
Benchmark prompt constructor module for Few-Shot Test Generation Pipeline.

This module handles constructing prompts for benchmarks with multiple test cases.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def construct_benchmark_prompt(
    database_path: str,
    benchmark: List[Dict[str, Any]],
    benchmark_query_key: str,
    output_file: Optional[str] = None,
    top_k: int = 3,
    similarity_threshold: float = 0.4,
    show_details: bool = True,
    benchmark_format: str = "standard"
) -> Optional[Dict[str, Any]]:
    """
    Construct prompts for a benchmark with multiple test cases.
    
    Args:
        database_path: Path to saved database (.pkl)
        benchmark_queries: List of benchmark queries, each containing:
            - id: Unique identifier for the test case
            - focal_method: The focal method code
            - metadata: Optional metadata (language, description, etc.)
        output_file: Path to save output (optional)
        top_k: Number of examples to retrieve per query
        similarity_threshold: Minimum similarity threshold
        show_details: Whether to show detailed statistics
        benchmark_format: Format of benchmark output (standard, jsonl, csv)
        
    Returns:
        Benchmark results dict if successful, None otherwise
    """
    try:
        from retriever import create_pipeline
        
        logger.info("=" * 80)
        logger.info("CONSTRUCTING BENCHMARK PROMPTS")
        logger.info("=" * 80)
        
        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline, database = create_pipeline(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Load database
        logger.info(f"Loading database from: {database_path}")
        database.load_index(database_path)
        logger.info(f"✓ Loaded database with {len(database.examples)} examples")

        logger.info(f"\nProcessing {len(benchmark)} benchmark queries...")

        # Process each benchmark query
        benchmark_results = []
        successful = 0
        failed = 0
        
        for idx, item in enumerate(benchmark):
            query_id = item.get('id', f'query_{idx}')
            focal_method = item.get(benchmark_query_key)
            metadata = item.get('metadata', {})

            if not focal_method:
                logger.warning(f"Skipping query {query_id}: No focal_method provided")
                failed += 1
                continue

            logger.info(f"\n[{idx}/{len(benchmark)}] Processing {query_id}...")

            # Run pipeline
            result = pipeline.run(focal_method)
            
            if result['status'] == 'success':
                successful += 1
                
                # Compile benchmark result
                benchmark_result = {
                    'id': query_id,
                    'focal_method': focal_method,
                    'metadata': metadata,
                    'prompt': result['few_shot_prompt'],
                    'retrieval_info': {
                        'examples_retrieved': result['pipeline_stages']['retrieval']['count'],
                        'examples_used': result['pipeline_stages']['prompt_construction']['examples_used'],
                        'avg_similarity': result['pipeline_stages']['validation']['avg_similarity']
                    }
                }
                
                benchmark_results.append(benchmark_result)
                
                if show_details:
                    logger.info(f"  ✓ Success - {benchmark_result['retrieval_info']['examples_used']} examples used")
                    logger.info(f"    Avg similarity: {benchmark_result['retrieval_info']['avg_similarity']:.4f}")
            else:
                failed += 1
                logger.warning(f"  ✗ Failed: {result.get('message', 'Unknown error')}")
                
                # Store failed result
                benchmark_results.append({
                    'id': query_id,
                    'focal_method': focal_method,
                    'metadata': metadata,
                    'status': 'failed',
                    'error': result.get('message', 'Unknown error')
                })
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK PROMPT CONSTRUCTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\n📊 Summary:")
        logger.info(f"  • Total queries: {len(benchmark)}")
        logger.info(f"  • Successful: {successful}")
        logger.info(f"  • Failed: {failed}")
        logger.info(f"  • Success rate: {successful/len(benchmark)*100:.1f}%")

        # Save results
        if output_file:
            output_path = Path(output_file)
            
            if benchmark_format == "jsonl":
                # Save as JSONL (one result per line)
                with open(output_file, 'w') as f:
                    for result in benchmark_results:
                        f.write(json.dumps(result) + '\n')
                logger.info(f"\n✓ Benchmark results saved to: {output_file} (JSONL format)")
                
            elif benchmark_format == "csv":
                # Save as CSV
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'focal_method', 'prompt', 'status', 'avg_similarity'])
                    writer.writeheader()
                    for result in benchmark_results:
                        writer.writerow({
                            'id': result['id'],
                            'focal_method': result['focal_method'][:100] + '...' if len(result['focal_method']) > 100 else result['focal_method'],
                            'prompt': result.get('prompt', '')[:200] + '...' if result.get('prompt') and len(result.get('prompt', '')) > 200 else result.get('prompt', ''),
                            'status': result.get('status', 'success'),
                            'avg_similarity': result.get('retrieval_info', {}).get('avg_similarity', 'N/A')
                        })
                logger.info(f"\n✓ Benchmark results saved to: {output_file} (CSV format)")
                
            else:  # standard JSON format
                with open(output_file, 'w') as f:
                    json.dump({
                        'benchmark_info': {
                            'total_queries': len(benchmark),
                            'successful': successful,
                            'failed': failed,
                            'success_rate': successful/len(benchmark)*100
                        },
                        'results': benchmark_results
                    }, f, indent=2)
                logger.info(f"\n✓ Benchmark results saved to: {output_file} (JSON format)")
        
        return {
            'benchmark_info': {
                'total_queries': len(benchmark),
                'successful': successful,
                'failed': failed
            },
            'results': benchmark_results
        }
        
    except Exception as e:
        logger.error(f"Failed to construct benchmark prompts: {e}")
        import traceback
        traceback.print_exc()
        return None


def construct_single_prompt(
    database_path: str,
    query_code: Optional[str] = None,
    query_file: Optional[str] = None,
    output_file: Optional[str] = None,
    top_k: int = 3,
    similarity_threshold: float = 0.4,
    show_details: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Construct prompt for a single query (backward compatibility).
    
    Args:
        database_path: Path to saved database (.pkl)
        query_code: Query code as string
        query_file: Path to file containing query code
        output_file: Path to save output (optional)
        top_k: Number of examples to retrieve
        similarity_threshold: Minimum similarity threshold
        show_details: Whether to show detailed statistics
        
    Returns:
        Pipeline result dict if successful, None otherwise
    """
    try:
        from retriever import create_pipeline
        
        # Load query code
        if query_file:
            logger.info(f"Reading query from file: {query_file}")
            with open(query_file, 'r') as f:
                query_code = f.read()
        elif not query_code:
            logger.error("Either query_code or query_file must be provided")
            return None
        
        logger.info("=" * 80)
        logger.info("CONSTRUCTING PROMPT")
        logger.info("=" * 80)
        
        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline, database = create_pipeline(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Load database
        logger.info(f"Loading database from: {database_path}")
        database.load_index(database_path)
        logger.info(f"✓ Loaded database with {len(database.examples)} examples")
        
        # Show query
        logger.info(f"\nQuery code:")
        logger.info("-" * 80)
        logger.info(query_code)
        logger.info("-" * 80)
        
        # Run pipeline
        logger.info("\nRunning pipeline...")
        result = pipeline.run(query_code)
        
        if result['status'] == 'success':
            logger.info("=" * 80)
            logger.info("✓ PROMPT CONSTRUCTION SUCCESSFUL")
            logger.info("=" * 80)
            
            # Show statistics
            if show_details:
                validation = result['pipeline_stages']['validation']
                retrieval = result['pipeline_stages']['retrieval']
                prompt_construction = result['pipeline_stages']['prompt_construction']
                
                logger.info(f"\n📊 Statistics:")
                logger.info(f"  • Examples in database: {len(database.examples)}")
                logger.info(f"  • Examples retrieved: {retrieval['count']}")
                logger.info(f"  • Average similarity: {validation['avg_similarity']:.4f}")
                logger.info(f"  • Examples used in prompt: {prompt_construction['examples_used']}")
                logger.info(f"  • Prompt length: {len(result['few_shot_prompt'])} characters")
            
            # Show prompt
            prompt = result['few_shot_prompt']
            logger.info(f"\n📝 Generated Few-Shot Prompt:")
            logger.info("=" * 80)
            print(prompt)  # Print to stdout for easy piping
            logger.info("=" * 80)
            
            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                
                # Save full result as JSON
                if output_path.suffix == '.json':
                    with open(output_file, 'w') as f:
                        json.dump({
                            'query': query_code,
                            'prompt': prompt,
                            'statistics': {
                                'retrieved': retrieval['count'],
                                'used': prompt_construction['examples_used'],
                                'avg_similarity': validation['avg_similarity']
                            },
                            'result': result
                        }, f, indent=2)
                    logger.info(f"\n✓ Full result saved to: {output_file}")
                
                # Save just the prompt as text
                else:
                    with open(output_file, 'w') as f:
                        f.write(prompt)
                    logger.info(f"\n✓ Prompt saved to: {output_file}")
            
            return result
            
        else:
            logger.error(f"Pipeline failed: {result.get('message', 'Unknown error')}")
            if 'error' in result:
                logger.error(f"Error details: {result['error']}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to construct prompt: {e}")
        import traceback
        traceback.print_exc()
        return None
