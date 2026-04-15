# Configuration System Refactoring - Summary

## What Was Changed

The entire RAG pipeline has been refactored to use a centralized YAML configuration file instead of hardcoded parameters scattered throughout the codebase.

## Files Created

### 1. **config.yaml** (NEW)
- Central configuration file with all pipeline parameters
- Organized into 7 logical sections
- Single source of truth for all settings

### 2. **src/config.py** (NEW)
- Configuration loader class using singleton pattern
- Methods: `get()`, `get_section()`, `get_all()`, `reload()`
- Convenience function: `get_config(key, default=None)`

### 3. **CONFIG_GUIDE.md** (NEW)
- Comprehensive guide to the configuration system
- Explains all parameters and their purpose
- Shows how to modify configuration
- Provides best practices

### 4. **CONFIG_QUICK_REF.md** (NEW)
- Quick reference for common configuration needs
- Lists all available configuration keys
- Shows customization scenarios
- Includes validation script

## Files Modified

### 1. **src/rag/doc_proc/processor.py**
**Changes:**
- Added `from src.config import get_config` import
- Modified `__init__` to load chunk_size, chunk_overlap, min_chunk_size from config if not provided
- Updated `load_documents()` to use `self.supported_formats` from config

**Before:**
```python
def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100, min_chunk_size: int = 50):
    ...
```

**After:**
```python
def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, min_chunk_size: Optional[int] = None):
    if chunk_size is None:
        chunk_size = get_config("document_processing.chunk_size")
    ...
```

### 2. **src/rag/generation/langchain_setup.py**
**Changes:**
- Removed hardcoded `EMBEDDING_DIM = 384`
- Modified `_build_local_embedding_fn()` to use embedding_dim from config
- Modified `setup_embedding_fn()` to load model_name, device, show_progress from config
- Modified `setup_llm()` to load model_id, max_new_tokens, do_sample, temperature, top_p, device from config

**Key changes:**
- Old: `embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", ...)`
- New: `embeddings = HuggingFaceEmbeddings(model_name=get_config("embeddings.model_name"), ...)`

### 3. **src/rag/retrieval/retriever.py**
**Changes:**
- Added `from src.config import get_config` import
- Modified `__init__` to load dense_weight, sparse_weight from config if not provided
- Updated `retrieve()` method to load candidate_k_multiplier from config

**Before:**
```python
def __init__(self, vector_store, embedding_fn=None, dense_weight: float = 0.7, sparse_weight: float = 0.3):
    ...
    candidate_k = max(top_k * 2, top_k)
```

**After:**
```python
def __init__(self, vector_store, embedding_fn=None, dense_weight: Optional[float] = None, sparse_weight: Optional[float] = None):
    if dense_weight is None:
        dense_weight = get_config("retriever.dense_weight")
    ...
    candidate_k_multiplier = get_config("retriever.candidate_k_multiplier")
    candidate_k = max(top_k * candidate_k_multiplier, top_k)
```

### 4. **src/rag/generation/generator.py**
**Changes:**
- Added `from src.config import get_config` import
- Modified `__init__` to load min_context_score, top_k from config if not provided

**Before:**
```python
def __init__(self, retriever, llm_fn, min_context_score: float = 0.3, top_k: int = 5):
    ...
```

**After:**
```python
def __init__(self, retriever, llm_fn, min_context_score: Optional[float] = None, top_k: Optional[int] = None):
    if min_context_score is None:
        min_context_score = get_config("rag_generator.min_context_score")
    if top_k is None:
        top_k = get_config("rag_generator.top_k")
    ...
```

### 5. **main.py**
**Changes:**
- Added `from src.config import get_config, Config` import
- Removed all hardcoded parameter values
- Modified pipeline initialization to use `get_config()` for all parameter-dependent logic
- Removed hardcoded values like `chunk_size=400`, `dense_weight=0.7`, `top_k=3`, etc.

**Before:**
```python
processor = DocumentProcessor(chunk_size=400, chunk_overlap=100)
vector_store = VectorStoreFactory.create("faiss", embedding_dim=384)
generator = RAGGenerator(retriever, llm_fn, min_context_score=0.3, top_k=3)
```

**After:**
```python
config = Config()
processor = DocumentProcessor()  # Parameters loaded from config
embedding_dim = get_config("embeddings.embedding_dim")
vector_store = VectorStoreFactory.create("faiss", embedding_dim=embedding_dim)
generator = RAGGenerator(retriever, llm_fn)  # Parameters loaded from config
```

## Configuration Parameters

All configuration is now centralized in `config.yaml`:

```
document_processing:
  - chunk_size: 400
  - chunk_overlap: 100
  - min_chunk_size: 50
  - supported_formats: [.txt, .md, .json, .csv]

embeddings:
  - model_name: sentence-transformers/all-MiniLM-L6-v2
  - embedding_dim: 384
  - device: cpu
  - show_progress: false

vector_store:
  - type: faiss
  - embedding_dim: 384

llm:
  - model_id: distilgpt2
  - max_new_tokens: 160
  - do_sample: true
  - temperature: 0.7
  - top_p: 0.9
  - device: auto

retriever:
  - dense_weight: 0.7
  - sparse_weight: 0.3
  - candidate_k_multiplier: 2
  - use_dense: true
  - use_sparse: true

rag_generator:
  - min_context_score: 0.3
  - top_k: 5
  - use_verification: false

logging:
  - verbose: true
  - show_retrieval_reasoning: true
```

## Benefits of This Refactoring

1. **Centralized Configuration** - All parameters in one place
2. **Easy Tuning** - Change parameters without modifying code
3. **No Code Changes** - Modify behavior by editing YAML
4. **Better Maintainability** - Parameter values are documented
5. **Flexibility** - Can override config values when instantiating classes
6. **Singleton Pattern** - Configuration loaded once and reused
7. **Type-Safe Defaults** - Classes handle missing config values gracefully

## Testing

The complete pipeline was tested and runs successfully:
```bash
✓ Configuration loaded from /home/arko/AgenticRAG/config.yaml
✓ Loaded 1 documents
✓ Created 19 chunks
✓ Created FAISS vector store (384-dim embeddings)
✓ HuggingFace embeddings initialized
✓ Vector store ready
✓ Hybrid retriever initialized (70% dense, 30% sparse)
✓ HuggingFace LLM initialized
✓ RAG generator initialized
✓ Tested pipeline with 2 sample queries
✓ RAG Pipeline Test Complete
```

## How to Use

### Basic Usage
```python
from src.config import get_config

# Get any parameter using dot notation
value = get_config("section.parameter")
```

### Modify Configuration
```yaml
# Edit config.yaml and save
# Changes are loaded automatically on next run
```

### Override Configuration
```python
# When instantiating classes, parameters override config
retriever = HybridRetriever(
    vector_store=store,
    embedding_fn=embedding_fn,
    dense_weight=0.8  # Overrides config value of 0.7
)
```

## Dependencies

- PyYAML (already installed)
- All existing RAG pipeline dependencies

## Migration Guide (for future development)

When adding new parameters:

1. Add to `config.yaml` under appropriate section
2. Update relevant class `__init__` to load parameter if not provided
3. Update documentation (CONFIG_GUIDE.md, CONFIG_QUICK_REF.md)

Example:
```yaml
# 1. In config.yaml
document_processing:
  new_parameter: 100
```

```python
# 2. In processor.py
def __init__(self, new_parameter: Optional[int] = None):
    if new_parameter is None:
        new_parameter = get_config("document_processing.new_parameter")
    self.new_parameter = new_parameter
```

## Files Status

- ✅ config.yaml - Created and validated
- ✅ src/config.py - Created and tested
- ✅ src/rag/doc_proc/processor.py - Updated
- ✅ src/rag/generation/langchain_setup.py - Updated
- ✅ src/rag/retrieval/retriever.py - Updated
- ✅ src/rag/generation/generator.py - Updated
- ✅ main.py - Updated
- ✅ CONFIG_GUIDE.md - Created
- ✅ CONFIG_QUICK_REF.md - Created
- ✅ Pipeline tested end-to-end

## Next Steps

Now that the configuration system is in place, you can:

1. Modify `config.yaml` to tune pipeline parameters
2. Add new configuration sections as needed
3. Use `get_config()` in any new code
4. Maintain backward compatibility with optional parameters in classes
