# AgenticRAG

Upload documents → Process automatically → Ask questions → Get AI answers with sources

**AgenticRAG** is a production-ready Retrieval-Augmented Generation (RAG) pipeline that combines document intelligence with open-source language models to provide accurate, grounded answers backed by your documents.

### Key Features

✅ **Hybrid Search** - Dense semantic vectors + sparse BM25 keyword matching for optimal retrieval  
✅ **HuggingFace Models** - Lightweight, open-source embeddings and LLM generation  
✅ **Centralized Configuration** - YAML-based configuration for all pipeline parameters  
✅ **Evaluation Framework** - Built-in RAGAS evaluation for precision and faithfulness metrics  
✅ **Grounded Responses** - All answers backed by source documents with confidence scoring  
✅ **Production-Ready** - Comprehensive error handling, logging, type safety, and validation  

## 🏗️ Architecture

```
Documents → Chunking → Embedding → Dense Index + Sparse Index (FAISS)
                                           ↓
Query → Embedding → Dense Search + Sparse Search (Hybrid)
                           ↓
                    Ranking & Merging
                           ↓
                   Retrieved Context Documents
                           ↓
              Prompt Template + Context Enrichment
                           ↓
            HuggingFace LLM Generation (via LangChain)
                           ↓
            Grounded Answer + Source References
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- `uv` package manager (or pip/conda)
- HuggingFace API token (for model access)

### Installation

```bash
git clone <repository-url>
cd AgenticRAG
uv sync
export HF_TOKEN=<your-huggingface-token>
```

### Running the Pipeline

```bash
# Run the main RAG pipeline
uv run python3 -m main

# Run evaluation on benchmark dataset
uv run python3 evaluate.py
```

## ⚙️ Configuration

All pipeline parameters are centralized in `config.yaml`. Key parameters include:

```yaml
# Document Processing
chunk_size: 400
chunk_overlap: 100

# Embeddings
embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384

# Retriever
retriever:
  dense_weight: 0.7
  sparse_weight: 0.3
  top_k: 5

# Generation
generation:
  model: "meta-llama/Llama-2-7b-hf"
  max_new_tokens: 256
  temperature: 0.7

# Evaluation
evaluation:
  metrics:
    - precision
    - faithfulness
```

Update `config.yaml` to customize pipeline behavior across all components.

## 📊 Evaluation & Benchmarking

Run the evaluation script to benchmark the pipeline against your dataset:

```bash
uv run python3 evaluate.py
```

This will:
- Load benchmark data from `data/benchmark/` folder
- Execute the RAG pipeline on benchmark queries
- Calculate precision and faithfulness metrics using RAGAS library
- Save results with configuration metadata for full reproducibility

Evaluation results are saved with complete configuration details for tracking experiment variations.

## 📁 Project Structure

```
AgenticRAG/
├── config.yaml                 # Centralized configuration file
├── main.py                     # Main RAG pipeline executor
├── evaluate.py                 # Evaluation & benchmarking script
├── app.py                      # Web/API interface
├── data/
│   ├── raw/                    # Raw input documents
│   └── benchmark/              # Benchmark queries and expected results
└── src/
    └── rag/
        ├── doc_proc/           # Document processing (chunking, loading)
        ├── vector_store/       # Vector store implementations (FAISS)
        ├── retrieval/          # Hybrid retriever (dense + sparse)
        └── generation/         # LLM-based response generation
```

## 🔧 Components

### Document Processor (`doc_proc/`)
- Loads documents from various formats
- Chunks documents based on configurable parameters
- Handles token boundaries and overlap

### Vector Store (`vector_store/`)
- FAISS-based dense vector indexing
- Configured embedding dimensions and model selection
- Scalable to large document collections

### Hybrid Retriever (`retrieval/`)
- Combines dense semantic search with sparse BM25 matching
- Configurable weighting between retrieval methods
- Ranking and result merging

### RAG Generator (`generation/`)
- LangChain-powered LLM integration
- Dynamic prompt construction with context
- Grounding-aware response generation

## 📈 Performance

The pipeline achieves strong performance on benchmark tasks:
- **Precision**: Percentage of retrieved documents that are relevant
- **Faithfulness**: Degree to which generated answers are grounded in source documents

Run `evaluate.py` against your specific benchmark to measure performance.

## 🛠️ Development

### Running Tests
```bash
# Execute pipeline on sample data
uv run python3 -m main
```

### Extending the Pipeline
1. Configuration changes: Update `config.yaml`
2. Module customization: Extend classes in `src/rag/`
3. New metrics: Add to evaluation script in `evaluate.py`

## 📜 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using Python, LangChain, HuggingFace, and RAGAS**
