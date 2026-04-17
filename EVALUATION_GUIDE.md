# RAG Pipeline Evaluation Guide

## Overview

The evaluation script (`evaluate.py`) measures the performance of your RAG pipeline using standardized metrics. It evaluates generated answers against benchmark queries and reference answers.

## Quick Start

### 1. Create Sample Benchmark Data
```bash
python3 create_sample_benchmark.py
```

This creates sample benchmark data in `data/benchmark/`:
- `queries.json` - List of test queries
- `ansers.json` - Reference answers (note: typo in filename is intentional)
- `corpus.json` - Document corpus

### 2. Run Evaluation
```bash
python3 evaluate.py
```

### 3. View Results
Results are saved to `results/eval_TIMESTAMP/` with configuration snapshots.

## Benchmark Data Format

### queries.json
```json
[
  "Query 1?",
  "Query 2?",
  "Query 3?"
]
```
A JSON array of query strings to test the pipeline against.

### ansers.json
```json
[
  "Reference answer 1",
  "Reference answer 2",
  "Reference answer 3"
]
```
A JSON array of expected/reference answers (must match query count).

### corpus.json
```json
{
  "doc_0": "Document content...",
  "doc_1": "Document content...",
  "doc_2": "Document content..."
}
```
A JSON object mapping document IDs to document content.

## Evaluation Metrics

### Custom Metrics (Always Computed)

- **total_queries**: Number of benchmark queries
- **successful_answers**: Queries answered successfully
- **failed_answers**: Queries that failed to generate answers
- **average_confidence**: Mean retrieval confidence score
- **average_context_chunks**: Mean number of context chunks used
- **queries_with_context**: Queries with retrieved context
- **context_coverage**: Percentage of queries with context (0-1)

### Faithfulness Proxy

Proxy metric measuring how well-grounded the answers are:
- **Calculation**: 50% from confidence score + 50% from context usage
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - 0.9-1.0: Excellent - Answers are well-grounded
  - 0.7-0.9: Good - Answers have good grounding
  - 0.5-0.7: Moderate - Some grounding with room for improvement
  - <0.5: Weak - Answers lack sufficient grounding

### Relevancy Proxy

Local relevancy metric based on ROUGE-L F1 score:
- **Calculation**: ROUGE-L F1 score against reference answers
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 0.8-1.0: Excellent - Very similar to reference
  - 0.6-0.8: Good - Similar to reference
  - 0.4-0.6: Moderate - Some overlap with reference
  - <0.4: Weak - Limited overlap with reference

### RAGAS Metrics (Optional)

If `OPENAI_API_KEY` is set, the evaluation also computes:

- **Faithfulness**: Measures whether the answer is faithful to the retrieved context
- **Answer Relevancy**: Measures how relevant the answer is to the query

## Results Directory Structure

```
results/
├── eval_20260415_135302/
│   ├── config.yaml                 # Configuration snapshot
│   ├── evaluation_report.txt        # Human-readable report
│   ├── metrics.json                 # Structured metrics
│   └── generated_answers.json       # Detailed answers with scoring
├── eval_20260415_135145/
│   └── ...
└── eval_20260415_134950/
    └── ...
```

### config.yaml
Complete configuration used for evaluation (from config.yaml).

### evaluation_report.txt
Human-readable report with:
- Pipeline configuration
- All computed metrics
- Ranges and standard deviations

### metrics.json
Structured results with:
```json
{
  "evaluation_timestamp": "20260415_135302",
  "custom_metrics": { ... },
  "ragas_metrics": { ... } or "evaluation_config": { ... },
  "evaluation_config": {
    "embedding_model": "...",
    "llm_model": "...",
    "top_k": 5,
    "min_context_score": 0.3,
    ...
  }
}
```

### generated_answers.json
Detailed answers with metadata:
```json
[
  {
    "query_id": 0,
    "query": "What is the main topic?",
    "generated_answer": "The main topic is...",
    "source_documents": ["doc_0"],
    "confidence": 0.85,
    "num_context_chunks": 3,
    "context_snippets": [
      {
        "chunk_id": "chunk_0",
        "score": 0.92,
        "content": "..."
      }
    ]
  },
  ...
]
```

## Using Your Own Benchmark Data

Place your benchmark files in `data/benchmark/`:
1. Create `queries.json` with your test queries
2. Create `ansers.json` with reference answers
3. Create `corpus.json` with document content
4. Run `python3 evaluate.py`

## Configuration Used in Evaluation

The evaluation uses the same configuration as the pipeline. To modify evaluation behavior, edit `config.yaml`:

```yaml
rag_generator:
  top_k: 5                    # Number of chunks to retrieve
  min_context_score: 0.3      # Minimum relevance threshold

embeddings:
  model_name: "..."           # Embedding model

llm:
  model_id: "..."             # LLM model

retriever:
  dense_weight: 0.7           # Semantic search weight
  sparse_weight: 0.3          # Keyword search weight
```

Then rerun the evaluation to see how config changes affect metrics.

## Interpreting Results

### Good Pipeline Indicators
- ✓ 100% context coverage (all queries retrieve context)
- ✓ Faithfulness > 0.7
- ✓ Relevancy > 0.5
- ✓ 0% failed answers

### Areas for Improvement
- Low faithfulness: Increase `min_context_score`, review retrieval
- Low relevancy: Consider better embedding models or larger chunks
- High failed answers: Debug pipeline components
- Low context coverage: Adjust `top_k` or embedding model

## Advanced: Using RAGAS with OpenAI

For production-grade evaluation with RAGAS metrics:

1. Install OpenAI library:
   ```bash
   pip install openai
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run evaluation:
   ```bash
   python3 evaluate.py
   ```

RAGAS will compute more sophisticated metrics using language models.

## Batch Evaluation

To evaluate multiple configurations:

```bash
# Create a loop script
for config in configs/*.yaml; do
  cp "$config" config.yaml
  python3 evaluate.py
done
```

Results will be saved with different timestamps for comparison.

## Troubleshooting

### "Benchmark data files not found"
- Run `python3 create_sample_benchmark.py` first
- Or place your files in `data/benchmark/`

### "RAGAS evaluation failed"
- Make sure OPENAI_API_KEY is set for RAGAS evaluation
- Or use fallback metrics (no API key needed)

### "Low relevancy scores"
- Generate is using an extractive fallback LLM
- Replace `distilgpt2` with a better model in config.yaml
- Example: "mistralai/Mistral-7B-Instruct-v0.1"

### Very long evaluation time
- Reduce queries in benchmark data
- Use smaller embedding/LLM models
- Reduce `top_k` in configuration

## Files Reference

| File | Purpose |
|------|---------|
| `evaluate.py` | Main evaluation script |
| `create_sample_benchmark.py` | Generate sample benchmark data |
| `data/benchmark/queries.json` | Test queries |
| `data/benchmark/ansers.json` | Reference answers |
| `data/benchmark/corpus.json` | Document corpus |
| `config.yaml` | Pipeline configuration |
| `results/` | Evaluation results (timestamped) |

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for RAGAS evaluation (optional)
- `HF_TOKEN` - HuggingFace token (used by pipeline)

## Tips for Better Evaluation

1. **Use diverse queries**: Cover different aspects of your documents
2. **Provide accurate references**: Ensure reference answers are correct
3. **Match corpus**: Include same documents used during evaluation
4. **Review failures**: Check `generated_answers.json` for error cases
5. **Iterate**: Run multiple evaluations with different configs
6. **Track metrics**: Save results folder with meaningful names

## Next Steps

After evaluation:
1. Review metrics in `evaluation_report.txt`
2. Check detailed answers in `generated_answers.json`
3. Identify bottlenecks (retrieval, generation, etc.)
4. Adjust `config.yaml` parameters
5. Re-run evaluation to measure improvements
6. Compare across runs by examining timestamped results folders
