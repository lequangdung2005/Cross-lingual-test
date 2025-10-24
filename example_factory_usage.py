"""
Example usage of the extensible retriever factory.

This script demonstrates a complete workflow:
1. Load training data from JSONL file
2. Build a retrieval database
3. Load benchmark test cases
4. Generate prompts for each benchmark case
5. Save prompts to output directory
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.factory import RetrieverFactory
from src.core.base import CodeExample


# Configuration
TRAINING_DATA_PATH = "data/database/high-resource/method2test/reformat_test.jsonl"
BENCHMARK_PATH = "data/benchmark/example_benchmark.json"
OUTPUT_DIR = "data/constructed_prompt"
DATABASE_SAVE_PATH = "data/database_index.pkl"


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
                    # Adjust field names based on your JSONL structure
                    focal_method = data.get('focal_method') or data.get('method') or data.get('code')
                    unit_test = data.get('unit_test') or data.get('test') or data.get('test_code')
                    
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


def load_benchmark(file_path: str) -> list:
    """
    Load benchmark test cases from JSON file.
    
    Args:
        file_path: Path to benchmark JSON file
        
    Returns:
        List of benchmark dictionaries
    """
    print(f"\nLoading benchmark from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
        
        print(f"✓ Loaded {len(benchmark)} benchmark cases")
        return benchmark
    
    except FileNotFoundError:
        print(f"  Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON: {e}")
        return []


def save_prompt(prompt: str, output_path: str):
    """
    Save a generated prompt to file.
    
    Args:
        prompt: The prompt text
        output_path: Path to save the prompt
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)


def main():
    """
    Main execution: Build database and generate prompts for benchmark.
    """
    print("=" * 80)
    print("FEW-SHOT TEST GENERATION - COMPLETE EXAMPLE")
    print("=" * 80)
    print()
    
    # Step 1: Create pipeline using factory
    print("Step 1: Creating pipeline with factory...")
    print("-" * 80)
    
    pipeline = RetrieverFactory.create_full_pipeline(
        method="unixcoder",
        db_type="dense_vector",
        pipeline_type="few_shot",
        pipeline_kwargs={
            "top_k": 5,
            "similarity_threshold": 0.5
        }
    )
    
    print(f"✓ Pipeline created:")
    print(f"  - Method: unixcoder")
    print(f"  - Database: dense_vector")
    print(f"  - Top-k: {pipeline.top_k}")
    print(f"  - Threshold: {pipeline.similarity_threshold}")
    print()
    
    # Step 2: Load training data
    print("Step 2: Loading training data...")
    print("-" * 80)
    
    # Load a subset for demo (adjust max_examples as needed)
    # Set to None to load all examples (may take time for large files)
    training_examples = load_training_data(
        TRAINING_DATA_PATH,
        max_examples=1000  # Change to None to load all
    )
    
    if not training_examples:
        print("⚠ No training data loaded. Please check the file path and format.")
        return
    
    print()
    
    # Step 3: Build database index
    print("Step 3: Building retrieval database...")
    print("-" * 80)
    
    print(f"Adding {len(training_examples)} examples to database...")
    pipeline.database.add_examples_bulk(training_examples)
    
    print("Building index (this may take a few minutes)...")
    pipeline.database.build_index()
    
    print(f"✓ Database built with {pipeline.database.size} examples")
    print()
    
    # Step 4: Save database index
    print("Step 4: Saving database index...")
    print("-" * 80)
    
    os.makedirs(os.path.dirname(DATABASE_SAVE_PATH), exist_ok=True)
    pipeline.database.save_index(DATABASE_SAVE_PATH)
    
    print(f"✓ Database saved to: {DATABASE_SAVE_PATH}")
    print()
    
    # Step 5: Load benchmark
    print("Step 5: Loading benchmark...")
    print("-" * 80)
    
    benchmark_cases = load_benchmark(BENCHMARK_PATH)
    
    if not benchmark_cases:
        print("⚠ No benchmark cases loaded. Please check the file path.")
        return
    
    print()
    
    # Step 6: Generate prompts for each benchmark case
    print("Step 6: Generating prompts for benchmark cases...")
    print("-" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i, case in enumerate(benchmark_cases, 1):
        case_id = case.get('id', f'case_{i}')
        focal_method = case.get('focal_method')
        
        if not focal_method:
            print(f"  ⚠ Skipping case {case_id}: no focal_method found")
            continue
        
        print(f"  [{i}/{len(benchmark_cases)}] Generating prompt for: {case_id}")
        
        # Generate prompt using pipeline
        prompt = pipeline.run(focal_method)
        
        # Save prompt to file
        output_path = os.path.join(OUTPUT_DIR, f"{case_id}_prompt.txt")
        save_prompt(prompt, output_path)
        
        print(f"      ✓ Saved to: {output_path}")
    
    print()
    print(f"✓ All prompts generated and saved to: {OUTPUT_DIR}/")
    print()
    
    # Step 7: Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training examples: {len(training_examples)}")
    print(f"Database size: {pipeline.database.size}")
    print(f"Database saved: {DATABASE_SAVE_PATH}")
    print(f"Benchmark cases: {len(benchmark_cases)}")
    print(f"Prompts generated: {len(benchmark_cases)}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print()
    print("✓ Complete! You can now use these prompts for test generation.")
    print("=" * 80)


if __name__ == "__main__":
    main()

