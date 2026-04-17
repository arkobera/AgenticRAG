# LLM Judge Evaluation Pipeline

## Overview

The updated RAG pipeline evaluation system includes a new **LLM Judge** component that uses Google's Generative AI (Gemini) to evaluate generated answers intelligently.

## Features

### Evaluation Methods (Priority Order)

1. **Google Generative AI Judge** (Primary)
   - Uses Gemini model for intelligent evaluation
   - Scores on 0-10 scale
   - Evaluates multiple dimensions:
     - **Answer Relevance**: How well does the answer address the query?
     - **Answer Faithfulness**: Is the answer grounded in provided context?
     - **Answer Completeness**: Does it cover all important aspects?
     - **Context Utilization**: How effectively is the context used?
   - Provides detailed reasoning and strengths/weaknesses

2. **RAGAS Metrics** (Fallback)
   - Faithfulness and AnswerRelevancy metrics
   - Uses OpenAI API (optional)
   
3. **Local Fallback Metrics** (Final Fallback)
   - ROUGE scores
   - Confidence-based metrics
   - No external APIs required

## Installation

### 1. Install Google Generative AI

```bash
pip install google-generativeai
```

Update your `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "google-generativeai>=0.3.0",
]
```

### 2. Get Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or add to `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Method 1: Evaluate with Benchmark Data

```bash
# Using default benchmark data
python3 evaluate.py

# Or explicitly
python3 evaluate.py --benchmark
```

### Method 2: Evaluate with train.csv (Recommended)

The `train.csv` file in the `raw/` folder contains real evaluation data.

```bash
# Evaluate all samples from train.csv
python3 evaluate.py --train-csv

# Evaluate first 10 samples
python3 evaluate.py --train-csv 10

# Evaluate first 100 samples
python3 evaluate.py --train-csv 100

# Evaluate with specific limit
python3 evaluate.py --train-csv 50
```

### Method 3: Force Specific Evaluation Method

```bash
# Force using Google Judge (if available)
python3 evaluate.py --google-judge --train-csv 10

# Force fallback evaluation (skip Google Judge)
python3 evaluate.py --no-google-judge --train-csv 10
```

### Method 4: Programmatic Usage

```python
from evaluate import BenchmarkEvaluator

# Initialize evaluator
evaluator = BenchmarkEvaluator(
    benchmark_dir="data/benchmark",
    results_dir="results"
)

# Evaluate with train.csv
success = evaluator.run_evaluation_with_train_csv(
    csv_path="raw/train.csv",
    generate_answers=True,
    limit=10  # Evaluate first 10 samples
)

# Or evaluate with benchmark data
success = evaluator.run_evaluation()
```

## Train.csv Format

Expected columns:
- `query`: The question/query
- `answer`: The reference/correct answer
- `context`: The context or relevant information
- Additional metadata columns (optional)

Example:
```csv
query,answer,context,sample_number,tokens,category
"What is the total amount?","$22,500.00","Services Vendor Inc...","1","150","invoice"
```

## Output

Evaluation results are saved to `results/eval_TIMESTAMP/` with:

- **metrics.json**: Numerical scores and statistics
- **generated_answers.json**: Detailed answer data with judge scores
- **evaluation_report.txt**: Human-readable report
- **config.yaml**: Configuration snapshot

### Metrics.json Structure

```json
{
  "evaluation_timestamp": "20260415_135302",
  "google_judge_metrics": {
    "method": "Google Generative AI Judge",
    "answer_relevance": {
      "mean": 8.2,
      "std": 1.1,
      "min": 5.0,
      "max": 9.8
    },
    "answer_faithfulness": {
      "mean": 7.9,
      "std": 1.3,
      "min": 4.5,
      "max": 9.9
    },
    "answer_completeness": {
      "mean": 7.5,
      "std": 1.5,
      "min": 4.0,
      "max": 9.7
    },
    "context_utilization": {
      "mean": 7.3,
      "std": 1.6,
      "min": 3.5,
      "max": 9.8
    },
    "overall": {
      "mean": 7.7,
      "std": 1.3,
      "min": 4.5,
      "max": 9.8
    },
    "judge_scores": {
      "detailed_judgments": [...]
    }
  },
  "custom_metrics": {
    "total_queries": 10,
    "successful_answers": 10,
    "context_coverage": 1.0
  }
}
```

## Google Judge Details

### Scoring Scale (0-10)

| Score | Interpretation |
|-------|-----------------|
| 9-10  | Excellent - Accurate, complete, well-grounded |
| 7-8   | Good - Mostly accurate and complete |
| 5-6   | Fair - Some accuracy, partially complete |
| 3-4   | Poor - Inaccurate or incomplete |
| 0-2   | Very Poor - Incorrect or irrelevant |

### Evaluation Prompt

The judge evaluates using this structure:

```
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Evaluate the generated answer based on:
1. Answer Relevance (0-10)
2. Answer Faithfulness (0-10)
3. Answer Completeness (0-10)
4. Context Utilization (0-10)
5. Overall Score (0-10)

Response format: JSON with scores, reasoning, strengths, weaknesses
```

## Rate Limiting

The Google Judge respects API rate limits:
- Default batch size: 5 evaluations
- Default delay: 1.0 second between batches
- Adjustable via `batch_size` and `delay_seconds` parameters

```python
judge.evaluate_batch(
    evaluations,
    batch_size=5,      # Evaluations per batch
    delay_seconds=1.0  # Delay between batches
)
```

## Troubleshooting

### Error: "google-generativeai is not installed"
```bash
pip install google-generativeai
```

### Error: "Google API key not found"
```bash
export GOOGLE_API_KEY="your-key-here"
# Verify it's set
echo $GOOGLE_API_KEY
```

### Error: "No valid evaluations to process"
Check that your generated answers have:
- Non-empty `query` field
- Non-empty `generated_answer` field
- Matching reference answer in `self.answers`

### Timeout/Rate Limit Issues
- Reduce batch size
- Increase delay between batches
- Use `--train-csv 10` to test with fewer samples first

## Advanced Configuration

### Custom Model

```python
from src.rag.evaluation.google_judge import GoogleGenerativeAIJudge

judge = GoogleGenerativeAIJudge(
    model_name="gemini-pro",  # Different model
    api_key="your-key"
)
```

### Batch Evaluation with Custom Parameters

```python
evaluations = [
    {
        "query": "What is X?",
        "generated_answer": "X is...",
        "reference_answer": "X is actually...",
        "context": "Context about X..."
    },
    # ... more items
]

judge_scores = judge.evaluate_batch(
    evaluations,
    batch_size=10,
    delay_seconds=2.0
)

# Access individual scores
for score in judge_scores:
    print(f"Overall: {score.overall_score}")
    print(f"Reasoning: {score.reasoning}")
    print(f"Strengths: {score.strengths}")
    print(f"Weaknesses: {score.weaknesses}")
```

## Performance Tips

1. **Use Smaller Batches Initially**: Start with `--train-csv 5` to verify setup
2. **Monitor API Usage**: Check Google API dashboard for remaining quota
3. **Cache Results**: Results are saved, avoid re-evaluating same data
4. **Use Appropriate Limits**: Balance between thoroughness and cost

## API Costs

Google Generative AI offers a free tier. Check current pricing at [Google AI Studio](https://aistudio.google.com/app/apikey).

Typical costs:
- Gemini 1.5 Flash: More affordable, sufficient for evaluation
- Gemini 1.5 Pro: Better quality, higher cost
- Free tier: Adequate for development/testing

## Example Results

```
✓ Google Judge Evaluation Complete
  Answer Relevance: 8.23 ± 1.12
  Answer Faithfulness: 7.89 ± 1.34
  Answer Completeness: 7.54 ± 1.48
  Context Utilization: 7.32 ± 1.58
  Overall Score: 7.74 ± 1.28
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Evaluate RAG Pipeline

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run evaluation
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: python3 evaluate.py --train-csv 20
```

## Additional Resources

- [Google Generative AI Documentation](https://ai.google.dev/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [RAG Evaluation Best Practices](https://python.langchain.com/docs/guides/evaluation)
