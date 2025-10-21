# Module Structure

The codebase has been refactored into a modular structure for better readability and maintainability.

## Directory Structure

```
Few_shot/
├── retriever/                          # Main package
│   ├── __init__.py                     # Package initialization & exports
│   ├── models.py                       # Data models
│   ├── embedder.py                     # UniXcoder embedding model
│   ├── database.py                     # Code example database
│   ├── pipeline.py                     # Complete pipeline
│   ├── utils.py                        # Utility functions
│   └── unixcoder.py                    # Backward compatibility (deprecated)
│
├── example_usage.py                    # Usage examples
├── quickstart.py                       # Quick start script
├── test_pipeline.py                    # Unit tests
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation
```

## Module Responsibilities

### 📦 `retriever/__init__.py`
**Purpose:** Package initialization and public API exports

**Exports:**
- `UniXcoderEmbedder`
- `CodeExampleDatabase`
- `FewShotTestGenerationPipeline`
- `CodeExample`
- `RetrievalResult`
- `create_pipeline`

**Usage:**
```python
from retriever import create_pipeline, CodeExample

# All components accessible from package level
pipeline, database = create_pipeline()
```

---

### 📊 `retriever/models.py`
**Purpose:** Data models and structures

**Classes:**
- `CodeExample`: Represents focal method + unit test pair
- `RetrievalResult`: Represents a retrieval result with similarity score

**Dependencies:** numpy (for embeddings)

**Usage:**
```python
from retriever.models import CodeExample, RetrievalResult

example = CodeExample(
    focal_method="def add(a, b): return a + b",
    unit_test="def test_add(): assert add(1, 2) == 3",
    metadata={"language": "python"}
)
```

---

### 🧠 `retriever/embedder.py`
**Purpose:** UniXcoder model wrapper for code embeddings

**Classes:**
- `UniXcoderEmbedder`: Embedding generation

**Key Methods:**
- `embed(code)` - Generate embedding for single code snippet
- `embed_batch(codes)` - Batch embedding generation

**Dependencies:**
- torch
- transformers
- numpy

**Usage:**
```python
from retriever.embedder import UniXcoderEmbedder

embedder = UniXcoderEmbedder(model_name="microsoft/unixcoder-base")
embedding = embedder.embed("def hello(): pass")
```

---

### 💾 `retriever/database.py`
**Purpose:** Code example database with embedding-based retrieval

**Classes:**
- `CodeExampleDatabase`: Store and retrieve code examples

**Key Methods:**
- `add_example()` - Add focal-test pair
- `add_examples_bulk()` - Add multiple pairs
- `build_index()` - Generate embeddings for all examples
- `retrieve()` - Find similar examples
- `save_index() / load_index()` - Persistence

**Dependencies:**
- numpy
- .models (CodeExample, RetrievalResult)
- .embedder (UniXcoderEmbedder)

**Usage:**
```python
from retriever.database import CodeExampleDatabase
from retriever.embedder import UniXcoderEmbedder

embedder = UniXcoderEmbedder()
database = CodeExampleDatabase(embedder)

database.add_example(focal, test, metadata)
database.build_index()
results = database.retrieve(query, top_k=5)
```

---

### 🔄 `retriever/pipeline.py`
**Purpose:** Complete end-to-end RAG pipeline

**Classes:**
- `FewShotTestGenerationPipeline`: Orchestrates all stages

**Pipeline Stages:**
1. `process_query()` - Embed input focal method
2. `retrieve_examples()` - RAG retrieval
3. `validate_retrieval()` - Quality checks
4. `construct_few_shot_prompt()` - Build prompt
5. `run()` - Execute complete pipeline

**Dependencies:**
- numpy
- .models (RetrievalResult)
- .embedder (UniXcoderEmbedder)
- .database (CodeExampleDatabase)

**Usage:**
```python
from retriever.pipeline import FewShotTestGenerationPipeline

pipeline = FewShotTestGenerationPipeline(
    embedder=embedder,
    database=database,
    top_k=5,
    similarity_threshold=0.5
)

result = pipeline.run(focal_method)
prompt = result['few_shot_prompt']
```

---

### 🛠️ `retriever/utils.py`
**Purpose:** Utility functions for quick setup

**Functions:**
- `create_pipeline()` - One-line pipeline creation

**Usage:**
```python
from retriever.utils import create_pipeline

# Creates embedder, database, and pipeline in one call
pipeline, database = create_pipeline(
    model_name="microsoft/unixcoder-base",
    top_k=5,
    similarity_threshold=0.5
)
```

---

### ⚠️ `retriever/unixcoder.py` (Deprecated)
**Purpose:** Backward compatibility wrapper

**Note:** This file re-exports everything from the new modular structure.
Legacy code using `from retriever.unixcoder import ...` will still work.

**Recommendation:** Update imports to use the new structure:
```python
# Old (still works, but deprecated)
from retriever.unixcoder import create_pipeline

# New (recommended)
from retriever import create_pipeline
```

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                    retriever Package                     │
└─────────────────────────────────────────────────────────┘
                             │
                             ├─── __init__.py (exports all)
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌─────▼─────┐   ┌─────▼──────┐
     │  models.py  │  │ embedder  │   │  utils.py  │
     │             │  │    .py    │   │            │
     │ • CodeEx... │  │ • UniX... │   │ • create_  │
     │ • Retriev...│  │   coder   │   │   pipeline │
     └─────────────┘  │   Embedder│   └──────┬─────┘
            ▲         └─────┬─────┘          │
            │               │                │
            │               ▼                │
            │        ┌─────────────┐         │
            └────────┤ database.py │◄────────┘
                     │             │
                     │ • CodeEx... │
                     │   Database  │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ pipeline.py │
                     │             │
                     │ • FewShot.. │
                     │   Pipeline  │
                     └─────────────┘
```

## Import Relationships

```python
# models.py
└── numpy

# embedder.py
├── torch
├── transformers
└── numpy

# database.py
├── numpy
├── models (CodeExample, RetrievalResult)
└── embedder (UniXcoderEmbedder)

# pipeline.py
├── numpy
├── models (RetrievalResult)
├── embedder (UniXcoderEmbedder)
└── database (CodeExampleDatabase)

# utils.py
├── embedder (UniXcoderEmbedder)
├── database (CodeExampleDatabase)
└── pipeline (FewShotTestGenerationPipeline)

# __init__.py
├── models
├── embedder
├── database
├── pipeline
└── utils
```

## Benefits of Modular Structure

### ✅ Improved Readability
- Each file has a single, clear responsibility
- Easier to navigate and understand
- Smaller, focused modules (~100-200 lines each)

### ✅ Better Maintainability
- Changes isolated to specific modules
- Easier to locate and fix bugs
- Reduced risk of unintended side effects

### ✅ Enhanced Testability
- Can test each module independently
- Mock dependencies easily
- Better unit test coverage

### ✅ Reusability
- Import only what you need
- Use components independently
- Extend functionality without modifying core

### ✅ Collaboration-Friendly
- Multiple developers can work on different modules
- Clearer code ownership
- Easier code reviews

## Migration Guide

### From Old Structure
```python
# Old: Everything in one file
from retriever.unixcoder import (
    UniXcoderEmbedder,
    CodeExampleDatabase,
    FewShotTestGenerationPipeline,
    create_pipeline
)
```

### To New Structure
```python
# New: Clean package imports (recommended)
from retriever import (
    UniXcoderEmbedder,
    CodeExampleDatabase,
    FewShotTestGenerationPipeline,
    create_pipeline
)

# Or import specific modules
from retriever.embedder import UniXcoderEmbedder
from retriever.database import CodeExampleDatabase
from retriever.pipeline import FewShotTestGenerationPipeline
from retriever.models import CodeExample, RetrievalResult
```

### Backward Compatibility
The old `retriever.unixcoder` import path still works but is deprecated:
```python
# Still works (backward compatible)
from retriever.unixcoder import create_pipeline

# But shows deprecation notice in docstring
```

## File Sizes (Approximate)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 22 | Package exports |
| `models.py` | 24 | Data structures |
| `embedder.py` | 118 | Embedding model |
| `database.py` | 181 | Database & retrieval |
| `pipeline.py` | 258 | Pipeline orchestration |
| `utils.py` | 44 | Helper functions |
| `unixcoder.py` | 61 | Compatibility wrapper |
| **Total** | **~708** | (vs 650 in monolithic) |

## When to Use Each Module

### Use `models.py` when:
- Creating custom data structures
- Extending CodeExample with new fields
- Custom serialization logic

### Use `embedder.py` when:
- Need embeddings without full pipeline
- Experimenting with different models
- Building custom retrieval systems

### Use `database.py` when:
- Managing large example collections
- Custom similarity metrics
- Persistence requirements

### Use `pipeline.py` when:
- End-to-end test generation
- Need validation logic
- Prompt construction

### Use `utils.py` when:
- Quick prototyping
- Standard use cases
- Getting started quickly

---

**Last Updated:** October 21, 2025
