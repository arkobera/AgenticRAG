# RAG Pipeline Configuration Guide

## Overview

The entire RAG pipeline now uses a centralized configuration system stored in `config.yaml`. This replaces hardcoded parameters scattered throughout the codebase, making it easy to tune the pipeline without modifying code.

## Configuration File Structure

The `config.yaml` file is organized into logical sections:

### 1. **document_processing** - Document Loading & Chunking
```yaml
document_processing:
  chunk_size: 400              # Tokens per chunk
  chunk_overlap: 100           # Overlap between chunks
  min_chunk_size: 50           # Minimum chunk size
  supported_formats:           # File types to load
    - .txt
    - .md
    - .json
    - .csv
```

### 2. **embeddings** - Text Embedding Configuration
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384           # Vector dimensionality
  device: "cpu"                # Device (cpu/cuda)
  show_progress: false         # Download progress
```

### 3. **vector_store** - Vector Database Configuration
```yaml
vector_store:
  type: "faiss"                # Backend type
  embedding_dim: 384           # Must match embeddings.embedding_dim
```

### 4. **llm** - Large Language Model Configuration
```yaml
llm:
  model_id: "distilgpt2"       # HuggingFace model
  max_new_tokens: 160          # Max generation length
  do_sample: true              # Use sampling
  temperature: 0.7             # Sampling temperature
  top_p: 0.9                   # Nucleus sampling
  device: "auto"               # Device (auto/cpu/cuda)
```

### 5. **retriever** - Hybrid Retrieval Configuration
```yaml
retriever:
  dense_weight: 0.7            # Vector search weight
  sparse_weight: 0.3           # Keyword search weight
  candidate_k_multiplier: 2    # Candidate set multiplier
  use_dense: true              # Enable vector search
  use_sparse: true             # Enable keyword search
```

### 6. **rag_generator** - Generation & Retrieval Configuration
```yaml
rag_generator:
  min_context_score: 0.3       # Relevance threshold
  top_k: 5                     # Chunks to retrieve
  use_verification: false      # Verify grounding
```

### 7. **logging** - Logging Configuration
```yaml
logging:
  verbose: true                # Detailed logs
  show_retrieval_reasoning: true  # Explain retrieval
```

## How to Use

### Loading Configuration in Code

**In Python files:**
```python
from src.config import get_config, Config

# Get individual values using dot notation
chunk_size = get_config("document_processing.chunk_size")
embedding_model = get_config("embeddings.model_name")

# Get entire sections
retriever_config = Config.get_section("retriever")

# Get all configuration
all_config = Config.get_all()
```

### Configuration Override

**When instantiating classes:**

The pipeline classes support optional parameters that override config values:

```python
# DocumentProcessor
processor = DocumentProcessor(
    chunk_size=500,              # Overrides config
    chunk_overlap=150
)

# HybridRetriever
retriever = HybridRetriever(
    vector_store=store,
    embedding_fn=embed_fn,
    dense_weight=0.8,            # Overrides config
    sparse_weight=0.2
)

# RAGGenerator
generator = RAGGenerator(
    retriever=retriever,
    llm_fn=llm_fn,
    min_context_score=0.5,       # Overrides config
    top_k=10
)
```

If parameters are not provided, the class automatically loads them from `config.yaml`.

## Files Updated

All pipeline components now use the configuration system:

- **src/config.py** - New Config loader class
- **src/rag/doc_proc/processor.py** - DocumentProcessor uses config
- **src/rag/generation/langchain_setup.py** - Embedding/LLM setup use config
- **src/rag/retrieval/retriever.py** - HybridRetriever uses config
- **src/rag/generation/generator.py** - RAGGenerator uses config
- **main.py** - Orchestration script uses config

## Modifying Configuration

To change any parameter:

1. Edit `config.yaml` in the project root
2. Adjust the relevant value
3. Run the pipeline - changes are loaded automatically

**Example: Increase retrieved chunks from 5 to 10**
```yaml
# In config.yaml
rag_generator:
  top_k: 10  # Changed from 5
```

**Example: Use a larger embedding model**
```yaml
# In config.yaml
embeddings:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Changed
  embedding_dim: 768  # Changed from 384
```

## Configuration Validation

Run this to validate your config.yaml:
```bash
python3 -c "
from src.config import get_config
print('Configuration loaded successfully!')
print(f'Chunk size: {get_config(\"document_processing.chunk_size\")}')
print(f'Embedding model: {get_config(\"embeddings.model_name\")}')
"
```

## Adding New Configuration

To add new parameters:

1. Add the parameter to `config.yaml` under the appropriate section
2. Load it in your code using `get_config("section.parameter")`

Example:
```yaml
# In config.yaml
document_processing:
  max_documents: 1000  # New parameter
```

```python
# In your code
from src.config import get_config

max_documents = get_config("document_processing.max_documents")
```

## Configuration Singleton Pattern

The `Config` class uses the singleton pattern, ensuring only one configuration instance is loaded and shared across the entire application. Configuration is loaded once on first access and reused thereafter.

To reload configuration:
```python
from src.config import Config
Config.reload()  # Reloads from config.yaml
```

## Best Practices

1. **Use descriptive parameter names** - Makes configs self-documenting
2. **Add comments** - Explain what each parameter does
3. **Group related parameters** - Organize by component
4. **Document ranges** - Specify valid value ranges in comments
5. **Test changes** - Always test after config modifications
6. **Version configs** - Keep configs in version control
