#!/usr/bin/env python3
"""
Quick Start Script for Few-Shot Test Generation Pipeline

This script demonstrates a minimal working example.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def quick_demo():
    """Run a quick demonstration of the pipeline."""
    
    print("\n" + "=" * 80)
    print("FEW-SHOT UNIT TEST GENERATION PIPELINE - QUICK START")
    print("=" * 80)
    
    # Check dependencies
    print("\n[1/6] Checking dependencies...")
    if not check_dependencies():
        return False
    print("✓ All dependencies installed")
    
    # Import modules
    print("\n[2/6] Loading pipeline components...")
    try:
        from retriever.unixcoder import create_pipeline
        print("✓ Pipeline modules loaded")
    except Exception as e:
        logger.error(f"Failed to load modules: {e}")
        return False
    
    # Create pipeline
    print("\n[3/6] Initializing UniXcoder embedder (this may take a moment)...")
    try:
        pipeline, database = create_pipeline(
            model_name="microsoft/unixcoder-base",
            top_k=3,
            similarity_threshold=0.4
        )
        print("✓ Pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.info("This requires internet connection to download the model.")
        return False
    
    # Add examples
    print("\n[4/6] Adding example focal-test pairs...")
    examples = [
        (
            """def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b""",
            """def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0""",
            {"category": "arithmetic"}
        ),
        (
            """def multiply(x: int, y: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return x * y""",
            """def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6""",
            {"category": "arithmetic"}
        ),
        (
            """def is_even(n: int) -> bool:
    \"\"\"Check if number is even.\"\"\"
    return n % 2 == 0""",
            """def test_is_even():
    assert is_even(2) == True
    assert is_even(3) == False
    assert is_even(0) == True
    assert is_even(-2) == True""",
            {"category": "validation"}
        ),
    ]
    
    database.add_examples_bulk(examples)
    print(f"✓ Added {len(examples)} examples")
    
    # Build index
    print("\n[5/6] Building embedding index...")
    try:
        database.build_index(batch_size=2)
        print("✓ Index built successfully")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return False
    
    # Generate tests
    print("\n[6/6] Generating few-shot prompt for new code...")
    
    query_code = """def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a.\"\"\"
    return a - b"""
    
    print(f"\nQuery code:\n{query_code}\n")
    
    try:
        result = pipeline.run(query_code)
        
        if result['status'] == 'success':
            print("=" * 80)
            print("SUCCESS! PIPELINE COMPLETED")
            print("=" * 80)
            
            # Show statistics
            validation = result['pipeline_stages']['validation']
            print(f"\n📊 Statistics:")
            print(f"  - Examples retrieved: {result['pipeline_stages']['retrieval']['count']}")
            print(f"  - Average similarity: {validation['avg_similarity']:.4f}")
            print(f"  - Examples used in prompt: {result['pipeline_stages']['prompt_construction']['examples_used']}")
            
            # Show prompt preview
            prompt = result['few_shot_prompt']
            print(f"\n📝 Generated Prompt ({len(prompt)} characters):")
            print("=" * 80)
            
            # Show first 800 characters
            preview_length = 800
            if len(prompt) <= preview_length:
                print(prompt)
            else:
                print(prompt[:preview_length])
                print(f"\n... ({len(prompt) - preview_length} more characters)")
            
            print("=" * 80)
            
            print("\n✅ Next Steps:")
            print("  1. Pass the prompt to your LLM (GPT-4, Claude, etc.)")
            print("  2. The LLM will generate unit tests based on the examples")
            print("  3. Review and validate the generated tests")
            
            print("\n💡 Tips:")
            print("  - Add more diverse examples for better results")
            print("  - Adjust similarity_threshold (currently 0.4)")
            print("  - Try different values of top_k (currently 3)")
            
            return True
            
        else:
            print(f"\n❌ Pipeline failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    success = quick_demo()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 QUICK START COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nFor more examples, run: python example_usage.py")
        print("For documentation, see: README.md")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Quick start encountered errors")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure internet connection (to download model)")
        print("  3. Check that you have enough RAM (~2GB for base model)")
        sys.exit(1)


if __name__ == "__main__":
    main()
