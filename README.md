# Few-Shot Unit Test Generation Pipeline

A **Retrieval-Augmented Generation (RAG)** pipeline for generating unit tests using few-shot learning with UniXcoder embeddings.

## 📋 Overview

This pipeline dynamically retrieves relevant code examples to improve the quality and relevance of generated unit tests. It uses UniXcoder, a unified cross-modal pre-trained model, to generate embeddings for code similarity matching.

### Key Features

- 🔍 **RAG-based retrieval** of relevant focal-test pairs
- 🧠 **UniXcoder embeddings** for semantic code similarity
- 📊 **Automatic validation** of retrieval quality
- 🔄 **Self-correction** mechanisms for poor results
- 💾 **Persistent storage** of indexed examples
- 🎯 **Customizable prompts** and instructions

---

## 🏗️ Pipeline Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEW-SHOT TEST GENERATION PIPELINE              │
└─────────────────────────────────────────────────────────────────┘

  Input: Focal Method (code to test)
    │
    ├──> [Stage 1] Query Processing
    │      • Tokenize and embed focal method
    │      • Generate vector representation
    │      ✓ Validation: Check embedding shape
    │
    ├──> [Stage 2] Retrieval (RAG)
    │      • Compute similarity with database
    │      • Rank and select top-k examples
    │      ✓ Validation: Check retrieval count
    │
    ├──> [Stage 3] Quality Validation
    │      • Apply similarity threshold
    │      • Filter low-quality matches
    │      • Calculate statistics
    │      ✓ Decision: Proceed or self-correct
    │
    ├──> [Stage 4] Prompt Construction
    │      • Format few-shot examples
    │      • Add instructions and query
    │      • Build complete prompt
    │      ✓ Validation: Check prompt completeness
    │
    └──> Output: Few-Shot Prompt
           → Ready for LLM test generation
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch transformers numpy
```

### Basic Usage

```python
from retriever.unixcoder import create_pipeline

# 1. Create pipeline
pipeline, database = create_pipeline(
    model_name="microsoft/unixcoder-base",
    top_k=5,
    similarity_threshold=0.5
)

# 2. Add example focal-test pairs
database.add_example(
    focal_method="""
def add(a: int, b: int) -> int:
    return a + b
    """,
    unit_test="""
def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    """,
    metadata={"language": "python"}
)

# 3. Build index
database.build_index()

# 4. Generate tests for new code
query = """
def multiply(x: int, y: int) -> int:
    return x * y
"""

result = pipeline.run(query)
print(result['few_shot_prompt'])
```

---

## 📖 Detailed Documentation

### Component 1: UniXcoderEmbedder

Generates vector embeddings for code snippets.

**Key Methods:**
- `embed(code)` - Generate embedding for single code snippet
- `embed_batch(codes)` - Batch embedding generation

**Input:** Code string  
**Output:** Numpy array (embedding vector)  
**Validation:** Check embedding shape matches model dimension

```python
from retriever.unixcoder import UniXcoderEmbedder

embedder = UniXcoderEmbedder(model_name="microsoft/unixcoder-base")
embedding = embedder.embed("def hello(): pass")
print(embedding.shape)  # (768,) for base model
```

---

### Component 2: CodeExampleDatabase

Stores and indexes focal-test pairs with their embeddings.

**Key Methods:**
- `add_example(focal, test, metadata)` - Add single example
- `add_examples_bulk(examples)` - Add multiple examples
- `build_index()` - Generate embeddings for all examples
- `retrieve(query, top_k)` - Find similar examples
- `save_index(path)` / `load_index(path)` - Persistence

**Input:** List of (focal_method, unit_test, metadata) tuples  
**Output:** RetrievalResult objects with similarity scores  
**Validation:** Verify index built before retrieval

```python
from retriever.unixcoder import CodeExampleDatabase, UniXcoderEmbedder

embedder = UniXcoderEmbedder()
database = CodeExampleDatabase(embedder)

# Add examples
examples = [
    ("def f(): pass", "def test_f(): pass", {"lang": "python"}),
    # ... more examples
]
database.add_examples_bulk(examples)
database.build_index()

# Retrieve
results = database.retrieve("def new_function(): pass", top_k=3)
for result in results:
    print(f"Similarity: {result.similarity_score:.4f}")
```

---

### Component 3: FewShotTestGenerationPipeline

Complete end-to-end pipeline orchestrating all stages.

**Key Methods:**
- `process_query(focal)` - Stage 1: Process and embed query
- `retrieve_examples(focal)` - Stage 2: RAG retrieval
- `validate_retrieval(results)` - Stage 3: Quality validation
- `construct_few_shot_prompt(focal, results)` - Stage 4: Prompt building
- `run(focal)` - Execute complete pipeline

**Input:** Focal method string  
**Output:** Dictionary with pipeline results and few-shot prompt  
**Validation:** Each stage validates outputs before proceeding

```python
from retriever.unixcoder import FewShotTestGenerationPipeline

pipeline = FewShotTestGenerationPipeline(
    embedder=embedder,
    database=database,
    top_k=5,
    similarity_threshold=0.5
)

# Run pipeline
result = pipeline.run(focal_method="def my_func(): pass")

# Check result
if result['status'] == 'success':
    prompt = result['few_shot_prompt']
    # Pass prompt to LLM...
else:
    print(f"Failed: {result['message']}")
```

---

## 🔄 Pipeline Stage Details

### Stage 1: Query Processing

**Purpose:** Prepare the query focal method for retrieval

**Steps:**
1. Receive focal method as input
2. Tokenize using UniXcoder tokenizer
3. Generate embedding vector
4. Store embedding and metadata

**Validation:**
- ✓ Embedding shape matches model dimension
- ✓ No NaN or Inf values in embedding

**Output:**
```python
{
    'focal_method': str,
    'embedding': np.ndarray,
    'embedding_shape': tuple,
    'status': 'success'
}
```

---

### Stage 2: Retrieval (RAG)

**Purpose:** Find most similar examples from database

**Steps:**
1. Compute similarity between query and all database embeddings
2. Rank examples by similarity score
3. Select top-k most similar examples

**Similarity Metrics:**
- **Cosine Similarity** (default): Measures angle between vectors
- **Euclidean Distance**: Measures L2 distance

**Validation:**
- ✓ Retrieved at least 1 example
- ✓ Similarity scores in valid range

**Output:**
```python
[
    RetrievalResult(
        example=CodeExample(...),
        similarity_score=0.87,
        rank=1
    ),
    # ... more results
]
```

---

### Stage 3: Quality Validation

**Purpose:** Ensure retrieved examples meet quality threshold

**Steps:**
1. Filter results by similarity threshold
2. Calculate statistics (mean, std, etc.)
3. Make go/no-go decision

**Validation Checks:**
- ✓ At least 1 example above threshold
- ✓ Average similarity meets minimum
- ✓ No degenerate results (all zeros)

**Self-Correction:**
If validation fails:
1. Log warning about poor results
2. Optionally lower threshold
3. Retry retrieval or request more examples

**Output:**
```python
{
    'valid_results': List[RetrievalResult],
    'total_retrieved': int,
    'filtered_count': int,
    'avg_similarity': float,
    'passed': bool,
    'similarity_threshold': float
}
```

---

### Stage 4: Prompt Construction

**Purpose:** Build few-shot prompt for LLM

**Steps:**
1. Format instruction header
2. Add each retrieved example (focal + test)
3. Add query focal method
4. Add template for output

**Prompt Template:**
```
# Task: Unit Test Generation

{instruction}

# Examples:

## Example 1 (Similarity: 0.87)

### Focal Method:
```
{focal_method_1}
```

### Unit Test:
```
{unit_test_1}
```

[... more examples ...]

# Your Task:

### Focal Method:
```
{query_focal}
```

### Unit Test:
```
# Generate unit test here
```
```

**Validation:**
- ✓ All examples included
- ✓ Proper formatting
- ✓ No truncation

**Output:** Complete prompt string ready for LLM

---

## 🎯 Advanced Usage

### Custom Similarity Threshold

```python
# Strict threshold for high-quality matches
pipeline = create_pipeline(similarity_threshold=0.8)

# Lenient threshold for broader matches
pipeline = create_pipeline(similarity_threshold=0.3)
```

### Batch Processing

```python
queries = [
    "def func1(): pass",
    "def func2(): pass",
    # ... more queries
]

results = []
for query in queries:
    result = pipeline.run(query)
    if result['status'] == 'success':
        results.append(result['few_shot_prompt'])
```

### Saving and Loading Database

```python
# Save database after indexing
database.build_index()
database.save_index("my_database.pkl")

# Load in new session
database2 = CodeExampleDatabase(embedder)
database2.load_index("my_database.pkl")
```

### Custom Prompt Instructions

```python
custom_instruction = """
Generate unit tests with:
1. Edge case coverage
2. Error handling tests
3. Comprehensive assertions
"""

result = pipeline.run(
    focal_method=code,
    instruction=custom_instruction
)
```

---

## 📊 Output Format

### Pipeline Result Structure

```python
{
    'focal_method': str,              # Input focal method
    'status': str,                     # 'success', 'failed_validation', or 'error'
    'message': str,                    # Status message (if applicable)
    'few_shot_prompt': str,            # Complete prompt for LLM
    'pipeline_stages': {
        'query_processing': {
            'embedding': np.ndarray,
            'embedding_shape': tuple,
            'status': str
        },
        'retrieval': {
            'results': List[RetrievalResult],
            'count': int
        },
        'validation': {
            'valid_results': List[RetrievalResult],
            'total_retrieved': int,
            'avg_similarity': float,
            'passed': bool
        },
        'prompt_construction': {
            'prompt': str,
            'prompt_length': int,
            'examples_used': int
        }
    }
}
```

---

## ✅ Stop Conditions

The pipeline completes when:

1. ✅ **Success Path:**
   - Query processed successfully
   - Examples retrieved and validated
   - Few-shot prompt constructed
   - Status: `'success'`
   - Output: Ready-to-use prompt for LLM

2. ⚠️ **Failed Validation:**
   - Retrieved examples below threshold
   - Status: `'failed_validation'`
   - Recommendation: Lower threshold or add more examples

3. ❌ **Error:**
   - Exception during any stage
   - Status: `'error'`
   - Error details in result

---

## 🧪 Testing

Run the example demonstrations:

```bash
# Run all examples
python example_usage.py

# Run specific example (modify main function)
python -c "from example_usage import example_basic_usage; example_basic_usage()"
```

---

## 🔧 Configuration

### Model Selection

```python
# Use base model (faster, lower accuracy)
embedder = UniXcoderEmbedder("microsoft/unixcoder-base")

# Use larger model (slower, higher accuracy)
embedder = UniXcoderEmbedder("microsoft/unixcoder-base-nine")
```

### Device Selection

```python
# Automatic device selection
embedder = UniXcoderEmbedder(device=None)  # Uses CUDA if available

# Force CPU
embedder = UniXcoderEmbedder(device="cpu")

# Force GPU
embedder = UniXcoderEmbedder(device="cuda")
```

### Pipeline Parameters

```python
pipeline = FewShotTestGenerationPipeline(
    embedder=embedder,
    database=database,
    top_k=5,                      # Number of examples to retrieve
    similarity_threshold=0.5       # Minimum similarity score
)
```

---

## 📝 Example Workflow

```python
# 1. Setup
pipeline, database = create_pipeline()

# 2. Populate database
examples = load_examples_from_file("examples.json")
database.add_examples_bulk(examples)
database.build_index()

# 3. Save for reuse
database.save_index("indexed_examples.pkl")

# 4. Generate tests
focal_code = read_focal_method("src/module.py")
result = pipeline.run(focal_code)

# 5. Validate and use
if result['status'] == 'success':
    validation = result['pipeline_stages']['validation']
    
    if validation['avg_similarity'] < 0.6:
        print("⚠️ Low similarity - consider reviewing examples")
    
    prompt = result['few_shot_prompt']
    
    # 6. Pass to LLM
    generated_tests = llm.generate(prompt)
    
else:
    print(f"❌ Pipeline failed: {result['message']}")
```

---

## 🐛 Troubleshooting

### Issue: "No examples retrieved"
**Solution:** Check if database is indexed: `database.build_index()`

### Issue: "All similarities below threshold"
**Solution:** Lower threshold or add more diverse examples

### Issue: "Model loading failed"
**Solution:** Install transformers: `pip install transformers`

### Issue: "CUDA out of memory"
**Solution:** Use CPU or reduce batch size

---

## 📚 References

- **UniXcoder Paper:** [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://arxiv.org/abs/2203.03850)
- **HuggingFace Model:** [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional embedding models (CodeBERT, GraphCodeBERT)
- More similarity metrics
- Prompt templates
- Example datasets

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Next Steps

After generating the few-shot prompt:

1. **Integrate with LLM:** Pass prompt to GPT-4, Claude, or CodeLlama
2. **Post-process:** Parse and validate generated tests
3. **Iterative refinement:** Use feedback to improve examples
4. **Evaluation:** Measure test quality and coverage

**Happy Test Generation! 🚀**
