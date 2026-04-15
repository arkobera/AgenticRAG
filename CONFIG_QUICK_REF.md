# Quick Configuration Reference

## How to Access Configuration Parameters

### In main.py or any script:
```python
from src.config import get_config, Config

# Load entire config (happens automatically)
config = Config()

# Get individual parameters
chunk_size = get_config("document_processing.chunk_size")           # 400
embedding_model = get_config("embeddings.model_name")              # sentence-transformers/all-MiniLM-L6-v2
llm_model = get_config("llm.model_id")                            # distilgpt2
top_k = get_config("rag_generator.top_k")                         # 5
dense_weight = get_config("retriever.dense_weight")               # 0.7
max_tokens = get_config("llm.max_new_tokens")                     # 160
```

## All Available Configuration Keys

### Document Processing
- `document_processing.chunk_size` - Document chunk size (400)
- `document_processing.chunk_overlap` - Overlap between chunks (100)
- `document_processing.min_chunk_size` - Minimum chunk size (50)
- `document_processing.supported_formats` - File formats to load

### Embeddings
- `embeddings.model_name` - Embedding model
- `embeddings.embedding_dim` - Embedding dimension (384)
- `embeddings.device` - Device for embeddings (cpu/cuda)
- `embeddings.show_progress` - Show download progress

### Vector Store
- `vector_store.type` - Vector store backend (faiss)
- `vector_store.embedding_dim` - Dimension (must match embeddings.embedding_dim)

### LLM
- `llm.model_id` - Model ID on HuggingFace (distilgpt2)
- `llm.max_new_tokens` - Max generation length (160)
- `llm.do_sample` - Use sampling (true)
- `llm.temperature` - Sampling temperature (0.7)
- `llm.top_p` - Nucleus sampling (0.9)
- `llm.device` - Device (auto/cpu/cuda)

### Retriever
- `retriever.dense_weight` - Vector search weight (0.7)
- `retriever.sparse_weight` - Keyword search weight (0.3)
- `retriever.candidate_k_multiplier` - Candidate set multiplier (2)
- `retriever.use_dense` - Enable vector search (true)
- `retriever.use_sparse` - Enable keyword search (true)

### RAG Generator
- `rag_generator.min_context_score` - Relevance threshold (0.3)
- `rag_generator.top_k` - Chunks to retrieve (5)
- `rag_generator.use_verification` - Verify grounding (false)

### Logging
- `logging.verbose` - Detailed logs (true)
- `logging.show_retrieval_reasoning` - Explain retrieval (true)

## Common Customization Scenarios

### Scenario 1: Increase Number of Retrieved Chunks
```yaml
# In config.yaml:
rag_generator:
  top_k: 10  # changed from 5
```

### Scenario 2: Better Quality with Slower Speed
```yaml
# In config.yaml:
llm:
  model_id: "mistralai/Mistral-7B-Instruct-v0.1"  # Switch to larger model
  max_new_tokens: 512                             # More tokens
  temperature: 0.5                                # Less random
```

### Scenario 3: Use GPU for Faster Processing
```yaml
# In config.yaml:
embeddings:
  device: "cuda"
llm:
  device: "cuda"
```

### Scenario 4: Larger Embedding Model for Better Semantics
```yaml
# In config.yaml:
embeddings:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  embedding_dim: 768  # Must change this too!
vector_store:
  embedding_dim: 768  # Must match embeddings.embedding_dim
```

### Scenario 5: More Aggressive Filtering
```yaml
# In config.yaml:
rag_generator:
  min_context_score: 0.7  # Only use very relevant chunks
```

## How Classes Use Configuration

### DocumentProcessor
```python
from src.rag.doc_proc.processor import DocumentProcessor

# Loads chunk_size, chunk_overlap, min_chunk_size from config
processor = DocumentProcessor()

# Or override:
processor = DocumentProcessor(chunk_size=500, chunk_overlap=200)
```

### HybridRetriever
```python
from src.rag.retrieval.retriever import HybridRetriever

# Loads dense_weight, sparse_weight from config
retriever = HybridRetriever(
    vector_store=store,
    embedding_fn=embed_fn
)

# Or override:
retriever = HybridRetriever(
    vector_store=store,
    embedding_fn=embed_fn,
    dense_weight=0.8,
    sparse_weight=0.2
)
```

### RAGGenerator
```python
from src.rag.generation.generator import RAGGenerator

# Loads min_context_score, top_k from config
generator = RAGGenerator(
    retriever=retriever,
    llm_fn=llm_fn
)

# Or override:
generator = RAGGenerator(
    retriever=retriever,
    llm_fn=llm_fn,
    min_context_score=0.5,
    top_k=10
)
```

### setup_embedding_fn & setup_llm
```python
from src.rag.generation.langchain_setup import setup_embedding_fn, setup_llm

# Both functions automatically load all parameters from config
embedding_fn = setup_embedding_fn()  # Loads embeddings.* config
llm_fn = setup_llm()                 # Loads llm.* config
```

## Configuration Validation Script
```bash
python3 -c "
from src.config import get_config, Config

print('=== Configuration Summary ===')
print(f'Embedding: {get_config(\"embeddings.model_name\")}')
print(f'LLM: {get_config(\"llm.model_id\")}')
print(f'Retrieval: {get_config(\"retriever.dense_weight\"):.0%} dense, {get_config(\"retriever.sparse_weight\"):.0%} sparse')
print(f'Top-K: {get_config(\"rag_generator.top_k\")}')
print(f'Chunk Size: {get_config(\"document_processing.chunk_size\")}')
"
```

## File locations
- **Configuration file**: `config.yaml` (project root)
- **Config loader**: `src/config.py`
- **This guide**: `CONFIG_QUICK_REF.md`
- **Full guide**: `CONFIG_GUIDE.md`
