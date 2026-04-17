# RAG Pipeline Evaluation System

## Overview

The evaluation system provides comprehensive metrics for assessing RAG pipeline performance. It evaluates generated answers against benchmark datasets using metrics like faithfulness and relevancy.

## Quick Start (30 seconds)

```bash
# 1. Create sample benchmark data
python3 create_sample_benchmark.py

# 2. Run evaluation
python3 evaluate.py

# 3. View results
python3 view_results.py
```

## System Components

### 1. `evaluate.py` - Main Evaluation Script
Comprehensive evaluation pipeline that:
- Loads benchmark queries and reference answers
- Sets up and configures the RAG pipeline
- Generates answers for all queries
- Computes evaluation metrics (local + optional RAGAS)
- Saves results with configuration snapshots

**Usage:**
```bash
python3 evaluate.py              # Single evaluation run
```

**Output:**
- Timestamped results directory in `results/eval_TIMESTAMP/`
- Configuration snapshot (config.yaml)
- Evaluation report (evaluation_report.txt)
- Structured metrics (metrics.json)
- Detailed answers (generated_answers.json)

### 2. `create_sample_benchmark.py` - Benchmark Generator
Creates sample benchmark data for testing:
- 10 sample queries
- 10 reference answers
- 10 document corpus items

**Usage:**
```bash
python3 create_sample_benchmark.py
```

**Creates:**
- `data/benchmark/queries.json`
- `data/benchmark/ansers.json`
- `data/benchmark/corpus.json`

### 3. `view_results.py` - Results Viewer
View and compare evaluation results:

**Usage:**
```bash
python3 view_results.py                    # Show latest results
python3 view_results.py list               # List all evaluations
python3 view_results.py compare T1 T2      # Compare two runs
```

### 4. `EVALUATION_GUIDE.md` - Detailed Guide
Comprehensive documentation covering:
- Benchmark data format
- Metrics explanation
- Results interpretation
- Configuration tuning
- RAGAS integration
- Troubleshooting

## Evaluation Metrics

### Metrics Computed

| Metric | Type | Range | Interpretation |
|--------|------|-------|-----------------|
| **Total Queries** | Custom | N/A | Number of benchmark queries |
| **Success Rate** | Custom | 0-100% | % of queries answered successfully |
| **Avg Confidence** | Custom | 0.0-1.0 | Mean retrieval confidence |
| **Avg Context Chunks** | Custom | 0-N | Mean chunks retrieved per query |
| **Context Coverage** | Custom | 0-1.0 | % queries with retrieved context |
| **Faithfulness Proxy** | Local | 0.0-1.0 | How grounded answers are |
| **Relevancy Proxy** | Local | 0.0-1.0 | ROUGE-L similarity to reference |

### Optional RAGAS Metrics

If `OPENAI_API_KEY` is set:
- **Faithfulness**: LLM-based grounding assessment
- **Answer Relevancy**: LLM-based query relevance assessment

## Results Structure

```
results/
├── eval_20260415_135302/          # Most recent
│   ├── config.yaml                # Pipeline config used
│   ├── evaluation_report.txt       # Human-readable report
│   ├── metrics.json               # Structured metrics
│   └── generated_answers.json      # Detailed answers
├── eval_20260415_135145/
│   └── ...
└── eval_20260415_134950/
    └── ...
```

## Workflow Examples

### Example 1: Basic Evaluation

```bash
# Create sample data
python3 create_sample_benchmark.py

# Run evaluation
python3 evaluate.py

# View results
python3 view_results.py
```

### Example 2: Benchmark Your Own Data

```bash
# Place your data in data/benchmark/
# - queries.json (list of queries)
# - ansers.json (list of reference answers)
# - corpus.json (dict of documents)

# Run evaluation with your data
python3 evaluate.py

# View results
python3 view_results.py
```

### Example 3: Tune Configuration and Compare

```bash
# Run baseline evaluation
python3 evaluate.py
# Note the timestamp (e.g., 20260415_135302)

# Edit config.yaml to change parameters
# e.g., increase top_k from 5 to 10

# Run evaluation again
python3 evaluate.py
# Note new timestamp (e.g., 20260415_135450)

# Compare results
python3 view_results.py compare 20260415_135302 20260415_135450
```

### Example 4: List All Evaluations

```bash
# See all evaluations performed
python3 view_results.py list

# Shows:
# Timestamp              Success Rate    Faithfulness   Relevancy
# 20260415_135302        100.0%          0.725          0.036
# 20260415_135145        100.0%          0.723          0.035
# 20260415_134950        100.0%          0.722          0.034
```

## Configuration Integration

The evaluation system **uses and saves your pipeline configuration**:

1. **During Evaluation**: Uses parameters from `config.yaml` to run the pipeline
2. **After Evaluation**: Saves config snapshot in `results/eval_TIMESTAMP/config.yaml`

This means:
- Different configs = Different evaluation results
- Each result includes the config that produced it
- You can compare configs by comparing results

**Configuration parameters affecting evaluation:**

```yaml
document_processing:
  chunk_size: 400              # Affects retrieval granularity
  chunk_overlap: 100           # Affects context continuity

embeddings:
  model_name: "..."            # Affects semantic matching
  embedding_dim: 384           # Affects vector space

llm:
  model_id: "..."              # Affects answer quality
  max_new_tokens: 160          # Affects answer length

rag_generator:
  top_k: 5                     # Affects # of context chunks
  min_context_score: 0.3       # Affects context filtering

retriever:
  dense_weight: 0.7            # Affects retrieval balance
  sparse_weight: 0.3
```

## Expected Outputs

After running evaluation, you'll see:

**Console Output:**
```
[1] LOADING BENCHMARK DATA
    ✓ Loaded 10 queries
    ✓ Loaded 10 ground truth answers
    ✓ Loaded 10 corpus documents

[2] SETTING UP RAG PIPELINE
    ✓ Configuration loaded
    ✓ Loaded 1 documents
    ✓ Created 37 chunks
    ...

[3] GENERATING ANSWERS
    ✓ Generated 10 answers

[4] EVALUATION METRICS
    ✓ Local Metrics Computed
    Faithfulness Proxy: 0.725 ± 0.025
    Relevancy Proxy: 0.036 ± 0.008

[5] CUSTOM METRICS
    ✓ Custom Metrics Computed
    Success Rate: 10/10
    Avg Confidence: 0.784

[6] SAVING RESULTS
    ✓ Configuration saved
    ✓ Generated answers saved
    ✓ Metrics saved
    ✓ Report saved
    ✓ All results saved to: results/eval_20260415_135302
```

**Result Files:**
- `evaluation_report.txt` - Summary of all metrics
- `metrics.json` - Structured metric data
- `generated_answers.json` - Detailed per-query results
- `config.yaml` - Configuration used

## Key Features

✅ **No External APIs by Default** - Works with local metrics
✅ **Optional RAGAS Integration** - Set OPENAI_API_KEY for advanced metrics
✅ **Configuration Snapshots** - Each result includes the config that produced it
✅ **Comprehensive Metrics** - Faithfulness, relevancy, success rates
✅ **Result Comparison** - View and compare multiple evaluation runs
✅ **Detailed Reports** - Human-readable + machine-readable outputs
✅ **Easy to Extend** - Add custom metrics to evaluate.py

## Troubleshooting

**Q: "Benchmark data files not found"**
A: Run `python3 create_sample_benchmark.py` to create sample data

**Q: "Results are very low"**
A: This is normal with small models like distilgpt2. Try:
   - Using a better embedding model
   - Using a better LLM model
   - Adjusting config.yaml parameters
   - Creating more relevant benchmark data

**Q: "Evaluation takes too long"**
A: Try:
   - Reducing the number of queries
   - Using smaller models
   - Setting shorter timeout

**Q: "Can I use RAGAS metrics?"**
A: Yes! Install OpenAI and set:
   ```bash
   export OPENAI_API_KEY="sk-..."
   python3 evaluate.py
   ```

## Performance Notes

- **First Run**: May download embedding/LLM models (one-time)
- **Typical Time**: 1-5 minutes for 10 queries depending on models
- **Results**: Always timestamped, never overwritten

## Architecture

```
evaluate.py
├── Load Configuration (config.yaml)
├── Load Benchmark Data (queries.json, answers.json, corpus.json)
├── Setup RAG Pipeline
│   ├── DocumentProcessor (chunk documents)
│   ├── VectorStoreFactory (create FAISS store)
│   ├── HybridRetriever (setup retrieval)
│   └── RAGGenerator (setup generation)
├── Generate Answers (for each query)
├── Compute Metrics
│   ├── Custom Metrics (always)
│   └── RAGAS Metrics (if OPENAI_API_KEY set)
└── Save Results
    ├── config.yaml (snapshot)
    ├── evaluation_report.txt
    ├── metrics.json
    └── generated_answers.json
```

## Files

| File | Purpose |
|------|---------|
| `evaluate.py` | Main evaluation script |
| `create_sample_benchmark.py` | Generate sample benchmark data |
| `view_results.py` | View and compare results |
| `EVALUATION_GUIDE.md` | Detailed evaluation guide |
| `data/benchmark/` | Benchmark data directory |
| `results/` | Evaluation results (timestamped) |

## Next Steps

1. **Run Baseline**: `python3 evaluate.py`
2. **Review Results**: `python3 view_results.py`
3. **Analyze Metrics**: Check `evaluation_report.txt`
4. **Identify Issues**: Look at detailed answers
5. **Tune Config**: Edit `config.yaml`
6. **Re-evaluate**: `python3 evaluate.py`
7. **Compare**: `python3 view_results.py list`

## Integration with Main Pipeline

The evaluation system is fully integrated with your RAG pipeline:

- Uses same configuration from `config.yaml`
- Uses same components (DocumentProcessor, HybridRetriever, etc.)
- Uses same embedding and LLM setup
- Saves configuration with results for reproducibility

This ensures your evaluation precisely reflects pipeline behavior.
