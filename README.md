# AgenticRAG

Upload documents → Process automatically → Ask questions → Get AI answers with sources

**AgenticRAG** is a production-ready Retrieval-Augmented Generation (RAG) pipeline that combines document intelligence with open-source language models to provide accurate, grounded answers backed by your documents.

### Key Features

✅ **Hybrid Search** - Dense semantic vectors + sparse BM25 keyword matching for optimal retrieval  
✅ **HuggingFace Models** - Lightweight, open-source embeddings and LLM generation  
✅ **Centralized Configuration** - YAML-based configuration for all pipeline parameters  
✅ **Evaluation Framework** - Built-in RAGAS evaluation + LLM Judge (Google Generative AI) for multiple dimensions  
✅ **Comprehensive Logging** - Structured logging with rotating file handlers and console output  
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
```

### Environment Setup

Create a `.env` file with required API keys:

```env
# HuggingFace token for model access
HF_TOKEN=<your-huggingface-token>

# Google Generative AI key (for LLM Judge evaluation)
GOOGLE_API_KEY=<your-google-api-key>
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
document_processing:
  chunk_size: 400
  chunk_overlap: 100
  supported_formats:
    - .txt
    - .pdf
    - .md

# Embeddings
embeddings:
  model: "all-MiniLM-L6-v2"
  embedding_dim: 384
  device: "cpu"

# Retriever
retriever:
  dense_weight: 0.7
  sparse_weight: 0.3
  top_k: 5
  min_context_score: 0.3

# Generation
rag_generator:
  max_new_tokens: 256
  temperature: 0.7
  top_k: 5

# Evaluation
evaluation:
  api_key: "${GOOGLE_API_KEY}"  # Set via environment variable
  model: "gemini-pro"
  metrics:
    - relevance
    - correctness
    - completeness
    - grounding
```

Update `config.yaml` to customize pipeline behavior across all components.

## � Logging

The pipeline includes comprehensive logging with rotating file handlers:

- **File Logs**: Detailed DEBUG-level logs saved to `log/` directory with timestamped filenames
- **Console Output**: INFO-level messages printed to console for real-time monitoring
- **Rotating Handlers**: Log files auto-rotate at 5MB, keeping last 3 backups
- **Structured Format**: All logs include timestamp, module name, log level, and message

Logs are automatically initialized when running `main.py` or `evaluate.py`. Access logs in:
```bash
ls -la log/  # View all log files
tail -f log/*.log  # Monitor live logging
```

## �📊 Evaluation & Benchmarking

Run the evaluation script to benchmark the pipeline against your dataset:

```bash
uv run python3 evaluate.py
```

This will:
- Load benchmark data from `benchmark/` folder or train.csv
- Execute the RAG pipeline on benchmark queries
- Evaluate answers using multiple methods:
  - **RAGAS Metrics**: Precision and faithfulness calculations
  - **LLM Judge**: Google Generative AI evaluates answers on 4 dimensions (relevance, correctness, completeness, grounding)
- Calculate comprehensive metrics with statistical summaries
- Save results with complete configuration metadata for reproducibility

Evaluation results are saved to `results/` with detailed metrics, generated answers, and evaluation reports for full experiment tracking.

## 📁 Project Structure

```
AgenticRAG/
├── config.yaml                 # Centralized configuration file
├── main.py                     # Main RAG pipeline executor
├── evaluate.py                 # Evaluation & benchmarking script
├── data/
│   ├── raw/                    # Raw input documents
│   └── benchmark/              # Benchmark queries and expected results
├── log/                        # Timestamped log files (auto-created)
└── src/
    ├── logger.py               # Centralized logging setup
    ├── config.py               # Configuration management
    └── rag/
        ├── doc_proc/           # Document processing (chunking, loading)
        ├── vector_store/       # Vector store implementations (FAISS)
        ├── retrieval/          # Hybrid retriever (dense + sparse)
        ├── generation/         # LLM-based response generation
        └── evaluation/         # Evaluation & benchmarking (RAGAS + LLM Judge)
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

### Evaluation Module (`evaluation/`)
- RAGAS-based metric calculations (precision, faithfulness)
- LLM Judge integration with Google Generative AI
- Multi-dimensional answer evaluation (relevance, correctness, completeness, grounding)
- Batch processing with rate-limit handling and comprehensive reporting

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
