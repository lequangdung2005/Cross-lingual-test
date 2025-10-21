"""
Example usage of the Few-Shot Unit Test Generation Pipeline
"""

from retriever.unixcoder import (
    create_pipeline,
    CodeExample,
    RetrievalResult
)


def example_basic_usage():
    """
    Basic usage example: Create pipeline, add examples, and generate tests.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("=" * 80)
    
    # Step 1: Create the pipeline
    print("\n[Step 1] Creating pipeline...")
    pipeline, database = create_pipeline(
        model_name="microsoft/unixcoder-base",
        top_k=3,
        similarity_threshold=0.3
    )
    print("✓ Pipeline created")
    
    # Step 2: Add example focal-test pairs
    print("\n[Step 2] Adding example focal-test pairs to database...")
    
    examples = [
        # Example 1: String manipulation
        (
            """def reverse_string(s: str) -> str:
    \"\"\"Reverse a string.\"\"\"
    return s[::-1]""",
            """def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("12345") == "54321\"""",
            {"language": "python", "category": "string"}
        ),
        # Example 2: List operations
        (
            """def find_max(numbers: list) -> int:
    \"\"\"Find maximum number in a list.\"\"\"
    if not numbers:
        raise ValueError("Empty list")
    return max(numbers)""",
            """def test_find_max():
    assert find_max([1, 2, 3, 4, 5]) == 5
    assert find_max([-1, -5, -2]) == -1
    assert find_max([42]) == 42
    
    with pytest.raises(ValueError):
        find_max([])""",
            {"language": "python", "category": "list"}
        ),
        # Example 3: Mathematical function
        (
            """def factorial(n: int) -> int:
    \"\"\"Calculate factorial of n.\"\"\"
    if n < 0:
        raise ValueError("Negative input")
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
            """def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(10) == 3628800
    
    with pytest.raises(ValueError):
        factorial(-1)""",
            {"language": "python", "category": "math"}
        ),
        # Example 4: String validation
        (
            """def is_palindrome(s: str) -> bool:
    \"\"\"Check if string is palindrome.\"\"\"
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]""",
            """def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("A man a plan a canal Panama") == True
    assert is_palindrome("") == True
    assert is_palindrome("12321") == True""",
            {"language": "python", "category": "string"}
        ),
        # Example 5: List filtering
        (
            """def filter_even(numbers: list) -> list:
    \"\"\"Filter even numbers from list.\"\"\"
    return [n for n in numbers if n % 2 == 0]""",
            """def test_filter_even():
    assert filter_even([1, 2, 3, 4, 5, 6]) == [2, 4, 6]
    assert filter_even([1, 3, 5]) == []
    assert filter_even([2, 4, 6]) == [2, 4, 6]
    assert filter_even([]) == []
    assert filter_even([0]) == [0]""",
            {"language": "python", "category": "list"}
        ),
    ]
    
    database.add_examples_bulk(examples)
    print(f"✓ Added {len(examples)} examples to database")
    
    # Step 3: Build the search index
    print("\n[Step 3] Building embedding index...")
    database.build_index(batch_size=4)
    print("✓ Index built successfully")
    
    # Step 4: Query with a new focal method
    print("\n[Step 4] Querying with new focal method...")
    
    query_focal = """def sum_list(numbers: list) -> int:
    \"\"\"Calculate sum of all numbers in list.\"\"\"
    total = 0
    for num in numbers:
        total += num
    return total"""
    
    print(f"\nQuery focal method:\n{query_focal}\n")
    
    # Step 5: Run the pipeline
    print("[Step 5] Running RAG pipeline...")
    result = pipeline.run(query_focal)
    
    # Step 6: Display results
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80)
    
    if result['status'] == 'success':
        print("\n✓ Pipeline executed successfully!")
        
        # Show retrieval statistics
        retrieval = result['pipeline_stages']['retrieval']
        print(f"\n📊 Retrieved {retrieval['count']} examples")
        
        validation = result['pipeline_stages']['validation']
        print(f"✓ Validation: {validation['valid_avg_similarity']:.4f} avg similarity")
        print(f"✓ Examples used: {result['pipeline_stages']['prompt_construction']['examples_used']}")
        
        # Show top retrieved examples
        print("\n🔍 Top Retrieved Examples:")
        for res in retrieval['results'][:3]:
            print(f"  - Rank {res.rank}: similarity={res.similarity_score:.4f}")
            print(f"    Category: {res.example.metadata.get('category', 'N/A')}")
        
        # Show the generated prompt (truncated)
        prompt = result['few_shot_prompt']
        print(f"\n📝 Generated Few-Shot Prompt ({len(prompt)} chars)")
        print("\nPrompt preview (first 500 chars):")
        print("-" * 80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 80)
        
        print("\n✅ Ready to pass prompt to LLM for test generation!")
        
    else:
        print(f"\n❌ Pipeline failed: {result.get('message', 'Unknown error')}")
    
    return result


def example_validation_and_correction():
    """
    Example showing validation and self-correction when results are poor.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Validation and Self-Correction")
    print("=" * 80)
    
    # Create pipeline with high similarity threshold
    print("\n[Setup] Creating pipeline with strict threshold (0.8)...")
    pipeline, database = create_pipeline(
        top_k=3,
        similarity_threshold=0.8  # Very strict threshold
    )
    
    # Add only one distantly related example
    database.add_example(
        focal_method="def print_hello(): print('hello')",
        unit_test="def test_print(): pass",
        metadata={"language": "python"}
    )
    database.build_index()
    
    # Query with unrelated focal method
    query = """def complex_calculation(a, b, c):
    return (a ** 2 + b ** 2) / c"""
    
    print("\n[Query] Testing with unrelated focal method...")
    result = pipeline.run(query)
    
    print("\n[Validation] Checking results...")
    if result['status'] == 'failed_validation':
        print("❌ Validation failed - similarity too low")
        print(f"   Message: {result['message']}")
        
        validation = result['pipeline_stages']['validation']
        print(f"   Average similarity: {validation['avg_similarity']:.4f}")
        print(f"   Threshold: {validation['similarity_threshold']:.4f}")
        
        print("\n[Self-Correction] Adjusting threshold and retrying...")
        pipeline.similarity_threshold = 0.3  # Lower threshold
        
        result_corrected = pipeline.run(query)
        if result_corrected['status'] == 'success':
            print("✓ Self-correction successful!")
            print(f"   New threshold: {pipeline.similarity_threshold}")
            prompt_len = len(result_corrected['few_shot_prompt'])
            print(f"   Generated prompt: {prompt_len} characters")
        
        return result_corrected
    else:
        print("✓ Validation passed")
        return result


def example_save_and_load():
    """
    Example showing how to save and load the database.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Save and Load Database")
    print("=" * 80)
    
    # Create and populate database
    print("\n[Step 1] Creating and populating database...")
    pipeline, database = create_pipeline()
    
    database.add_example(
        "def add(a, b): return a + b",
        "def test_add(): assert add(1, 2) == 3",
        {"category": "math"}
    )
    database.build_index()
    print(f"✓ Database created with {len(database.examples)} examples")
    
    # Save database
    save_path = "/tmp/test_database.pkl"
    print(f"\n[Step 2] Saving database to {save_path}...")
    database.save_index(save_path)
    print("✓ Database saved")
    
    # Create new pipeline and load database
    print("\n[Step 3] Creating new pipeline and loading database...")
    pipeline2, database2 = create_pipeline()
    database2.load_index(save_path)
    print(f"✓ Database loaded with {len(database2.examples)} examples")
    
    # Verify it works
    print("\n[Step 4] Verifying loaded database...")
    results = database2.retrieve("def multiply(x, y): return x * y", top_k=1)
    if results:
        print(f"✓ Retrieval works! Found {len(results)} results")
    
    return database2


def example_custom_prompt_instruction():
    """
    Example showing custom prompt instructions.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Prompt Instructions")
    print("=" * 80)
    
    pipeline, database = create_pipeline()
    
    # Add examples
    database.add_example(
        "def greet(name): return f'Hello, {name}'",
        "def test_greet(): assert greet('Alice') == 'Hello, Alice'",
        {}
    )
    database.build_index()
    
    # Run with custom instruction
    custom_instruction = """
Generate comprehensive unit tests with the following requirements:
1. Test normal cases
2. Test edge cases
3. Test error conditions
4. Include docstrings
5. Use pytest fixtures where appropriate
"""
    
    print("\n[Custom Instruction] Running pipeline with custom instruction...")
    result = pipeline.run(
        focal_method="def calculate_discount(price, percent): return price * (1 - percent/100)",
        instruction=custom_instruction
    )
    
    if result['status'] == 'success':
        print("✓ Prompt generated with custom instruction")
        print("\nCustom instruction included in prompt:")
        prompt = result['few_shot_prompt']
        # Extract instruction part
        if custom_instruction.strip() in prompt:
            print("  ✓ Custom instruction found in prompt")
    
    return result


def main():
    """
    Run all examples to demonstrate the pipeline.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Few-Shot Unit Test Generation Pipeline" + " " * 24 + "║")
    print("║" + " " * 25 + "Example Demonstrations" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Run examples
        example_basic_usage()
        example_validation_and_correction()
        example_save_and_load()
        example_custom_prompt_instruction()
        
        print("\n\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\n📚 Next Steps:")
        print("  1. Integrate with your LLM for test generation")
        print("  2. Add more diverse examples to the database")
        print("  3. Fine-tune similarity thresholds for your use case")
        print("  4. Experiment with different prompt templates")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nNote: This requires torch, transformers, and numpy to be installed.")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
