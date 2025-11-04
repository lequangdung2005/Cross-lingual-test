#!/usr/bin/env python3
"""
Test Runner Script - Few-Shot Retrieval Pipeline

This script builds a retrieval database from training data and generates
few-shot prompts for benchmark test cases across multiple programming languages.

Usage:
    python test_runner.py [--embedder MODEL_NAME] [--max-examples N]
"""

import sys
import os
import json
from pathlib import Path
import logging
import argparse

import datasets

from src.core.factory import RetrieverFactory
from src.core.base import CodeExample
from src.retrievers.dense.database import CodeExampleDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration defaults
TRAINING_DATA_PATH = "/kaggle/input/method2test2/reformat_test.jsonl"
BENCHMARK_REPO = "Tessera2025/Tessera2025"
OUTPUT_DIR = "src/data/constructed_prompt"


def _register_builtin_implementations():
    """Register built-in retriever implementations."""
    try:
        from src.retrievers.dense.embedder import DensecoderEmbedder
        # Will be registered with the embedder name from config
        logger.info("DensecoderEmbedder available for registration")
    except ImportError:
        logger.warning("Could not import DensecoderEmbedder")
    
    try:
        from src.retrievers.dense.database import CodeExampleDatabase
        RetrieverFactory.register_database("dense_vector", CodeExampleDatabase)
        logger.info("Registered CodeExampleDatabase")
    except ImportError:
        logger.warning("Could not register CodeExampleDatabase")
    
    try:
        from src.retrievers.fewshot_pipeline import FewShotTestGenerationPipeline
        RetrieverFactory.register_pipeline("few_shot", FewShotTestGenerationPipeline)
        logger.info("Registered FewShotTestGenerationPipeline")
    except ImportError:
        logger.warning("Could not register FewShotTestGenerationPipeline")


def load_training_data(file_path: str, max_examples: int = None) -> list:
    """
    Load training data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        max_examples: Maximum number of examples to load (None = load all)
        
    Returns:
        List of CodeExample objects
    """
    examples = []
    print(f"Loading training data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    
                    # Extract focal_method and unit_test
                    focal_method = data.get('focal_method') 
                    unit_test = data.get('unit_test') 
                    
                    if focal_method and unit_test:
                        example = CodeExample(
                            focal_method=focal_method,
                            unit_test=unit_test,
                            metadata=data.get('metadata', {})
                        )
                        examples.append(example)
                    
                    # Progress indicator
                    if (i + 1) % 1000 == 0:
                        print(f"  Loaded {i + 1} examples...")
                        
                except json.JSONDecodeError:
                    print(f"  Warning: Skipping invalid JSON at line {i + 1}")
                    continue
    
    except FileNotFoundError:
        print(f"  Error: File not found: {file_path}")
        return []
    
    print(f"✓ Loaded {len(examples)} training examples")
    return examples


def load_benchmark(repo_path: str) -> tuple:
    """
    Load benchmark test cases from HuggingFace dataset.
    
    Args:
        repo_path: HuggingFace dataset repository path
        
    Returns:
        Tuple of (benchmark_rust, benchmark_go, benchmark_julia) lists
    """
    try:
        dataset = datasets.load_dataset(repo_path, trust_remote_code=True)
        benchmark_rust = dataset["rust"].to_list()
        benchmark_go = dataset["go"].to_list()
        benchmark_julia = dataset["julia"].to_list()
        
        return benchmark_rust, benchmark_go, benchmark_julia
    
    except Exception as e:
        print(f"  Error loading benchmark: {e}")
        return [], [], []


def to_jsonable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, list):
        return [to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def save_benchmark(benchmark, output_path: str):
    """
    Save benchmark results to JSONL file.
    
    Args:
        benchmark: List of benchmark results
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jsonable_benchmark = [to_jsonable(d) for d in benchmark]

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in jsonable_benchmark:
            f.write(json.dumps(item) + "\n")


def build_or_load_database(pipeline, database_save_path: str, training_data_path: str, max_examples: int = None):
    """
    Build or load the retrieval database.
    
    Args:
        pipeline: The retrieval pipeline
        database_save_path: Path to save/load the database
        training_data_path: Path to training data JSONL
        max_examples: Maximum examples to load (None = all)
    """
    print("Steps 2-4: Loading or Building retrieval database...")
    print("-" * 80)

    # Check if database already exists
    if os.path.exists(database_save_path):
        print(f"✓ Found existing database at: {database_save_path}")
        print("Loading database index...")
        
        try:
            pipeline.database.load_index(database_save_path)
            print(f"✓ Database loaded successfully with {pipeline.database.size} examples")
            return True
        except Exception as e:
            print(f"⚠ Error loading database: {e}")
            print("Building new database instead...")

    print(f"✗ No existing database found at: {database_save_path}")
    print("Building new database...")
    
    # Load training data
    training_examples = load_training_data(training_data_path, max_examples=max_examples)
    
    if not training_examples:
        print("⚠ No training data loaded. Please check the file path and format.")
        return False
    
    # Build database
    print(f"Adding {len(training_examples)} examples to database...")
    pipeline.database.add_examples_bulk(training_examples)
    
    print("Building index (this may take a few minutes)...")
    pipeline.database.build_index(batch_size=64)
    
    print(f"✓ Database built with {pipeline.database.size} examples")
    
    # Save database
    print("Saving database index...")
    os.makedirs(os.path.dirname(database_save_path), exist_ok=True)
    pipeline.database.save_index(database_save_path)
    print(f"✓ Database saved to: {database_save_path}")
    
    return True


def retrieve_context(pipeline, benchmark_cases: list, lang: str, embedder_name: str, output_dir: str):
    """
    Retrieve few-shot prompts for benchmark cases.
    
    Args:
        pipeline: The retrieval pipeline
        benchmark_cases: List of benchmark test cases
        lang: Programming language (rust/go/julia)
        embedder_name: Name of the embedder model
        output_dir: Base output directory
        
    Returns:
        Number of prompts generated
    """
    output_path = os.path.join(output_dir, embedder_name, lang, "data_with_fewshot.jsonl")
    results = []
    
    for i, case in enumerate(benchmark_cases, 1):
        case_id = case.get('id', f'case_{i}')
        focal_method = case.get('focal_code')
        
        if not focal_method:
            print(f"  ⚠ Skipping case {case_id}: no focal_method found")
            continue
            
        # Retrieve prompt using pipeline
        prompt = pipeline.run(focal_method)
        
        result = {"id": case_id, "retrieved_context": prompt}
        results.append(result)
        
        # Progress indicator
        if i % 50 == 0:
            print(f"  Processed {i}/{len(benchmark_cases)} cases for {lang}...")
    
    # Save prompts to file
    save_benchmark(results, output_path)
    print(f"  ✓ Saved {len(results)} prompts to: {output_path}")
    
    return len(results)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Few-Shot Test Generation Pipeline")
    parser.add_argument(
        '--embedder',
        type=str,
        default='Salesforce/SFR-Embedding-Code-400M_R',
        help='Embedder model name'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum training examples to load (default: all)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of similar examples to retrieve'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.0,
        help='Minimum similarity threshold for retrieval (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    EMBEDDER_NAME = args.embedder
    DATABASE_SAVE_PATH = f"src/data/database/{EMBEDDER_NAME}/database_index.pkl"
    
    print("=" * 80)
    print("Few-Shot Test Generation Pipeline")
    print("=" * 80)
    print(f"Embedder: {EMBEDDER_NAME}")
    print(f"Database: {DATABASE_SAVE_PATH}")
    print(f"Top-K: {args.top_k}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print("=" * 80)
    print()
    
    # Register implementations
    _register_builtin_implementations()
    
    # Register the specific embedder
    try:
        from src.retrievers.dense.embedder import DensecoderEmbedder
        RetrieverFactory.register_embedder(EMBEDDER_NAME, DensecoderEmbedder)
        logger.info(f"Registered embedder: {EMBEDDER_NAME}")
    except Exception as e:
        logger.error(f"Failed to register embedder: {e}")
        return 1
    
    # Create pipeline
    print("Step 1: Creating retrieval pipeline...")
    print("-" * 80)
    
    try:
        pipeline = RetrieverFactory.create_full_pipeline(
            method=EMBEDDER_NAME,
            db_type="dense_vector",
            pipeline_type="few_shot",
            pipeline_kwargs={
                "top_k": args.top_k,
                "similarity_threshold": args.similarity_threshold
            }
        )
        print("✓ Pipeline created successfully")
        print()
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return 1
    
    # Build or load database
    success = build_or_load_database(
        pipeline,
        DATABASE_SAVE_PATH,
        TRAINING_DATA_PATH,
        max_examples=args.max_examples
    )
    
    if not success:
        logger.error("Failed to build/load database")
        return 1
    print()
    
    # Load benchmark
    print("Step 5: Loading benchmark...")
    print("-" * 80)
    
    benchmark_rust, benchmark_go, benchmark_julia = load_benchmark(BENCHMARK_REPO)
    
    if not (benchmark_rust or benchmark_go or benchmark_julia):
        print("⚠ No benchmark cases loaded. Please check the repository.")
        return 1
    
    print(f"✓ Loaded benchmarks:")
    print(f"  - Rust: {len(benchmark_rust)} cases")
    print(f"  - Go: {len(benchmark_go)} cases")
    print(f"  - Julia: {len(benchmark_julia)} cases")
    print()
    
    # Retrieve prompts for each language
    print("Step 6: Generating prompts for benchmark cases...")
    print("-" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_prompts = 0
    
    for lang, benchmark_cases in [
        ('rust', benchmark_rust),
        ('go', benchmark_go),
        ('julia', benchmark_julia)
    ]:
        if not benchmark_cases:
            print(f"  Skipping {lang}: no cases loaded")
            continue
        
        print(f"\nProcessing {lang.upper()}...")
        count = retrieve_context(pipeline, benchmark_cases, lang, EMBEDDER_NAME, OUTPUT_DIR)
        total_prompts += count
    
    print()
    print("=" * 80)
    print(f"✓ All done! Generated {total_prompts} total prompts")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
