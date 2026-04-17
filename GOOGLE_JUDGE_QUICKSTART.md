# LLM Judge Evaluation - Quick Start Guide

## What Changed? 🎯

Your RAG pipeline now has an intelligent LLM-based evaluation system using Google's Generative AI (Gemini) as a judge.

**New Components:**
- `src/rag/evaluation/google_judge.py` - Google Generative AI Judge implementation
- `src/rag/evaluation/metrics.py` - Metrics computation utilities
- Updated `evaluate.py` - Enhanced with Google Judge integration
- `LLM_JUDGE_EVALUATION.md` - Comprehensive documentation
- `example_google_judge.py` - Example usage script

## 3-Minute Setup ⚡

### Step 1: Install Dependencies (1 minute)

```bash
# Install google-generativeai
pip install google-generativeai

# Or update your environment
pip install -e .
```

### Step 2: Get Google API Key (1 minute)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key
4. Set environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"

# Verify it's set
echo $GOOGLE_API_KEY
```

### Step 3: Run Evaluation (1 minute)

```bash
# Quick test with first 5 samples
python3 evaluate.py --train-csv 5

# Full evaluation with train.csv
python3 evaluate.py --train-csv

# Or test with example
python3 example_google_judge.py
```

## What You Can Do Now 🚀

### 1. Evaluate with your train.csv data

```bash
# Evaluate all samples
python3 evaluate.py --train-csv

# Evaluate first N samples (faster for testing)
python3 evaluate.py --train-csv 10
python3 evaluate.py --train-csv 100
```

### 2. View detailed evaluation results

Results are saved in `results/eval_TIMESTAMP/`:
- `metrics.json` - Scores and statistics
- `generated_answers.json` - Detailed answer data
- `evaluation_report.txt` - Human-readable report

### 3. Understand the Scores (0-10 scale)

The judge evaluates on 4 dimensions + overall:

| Metric | What it measures |
|--------|-----------------|
| **Answer Relevance** | How well the answer addresses the query |
| **Answer Faithfulness** | Is the answer grounded in the context? |
| **Answer Completeness** | Does it cover important aspects? |
| **Context Utilization** | How well is the provided context used? |
| **Overall Score** | Combined quality rating |

### 4. Use in Python Code

```python
from src.rag.evaluation.google_judge import GoogleGenerativeAIJudge

# Initialize judge
judge = GoogleGenerativeAIJudge()

# Evaluate a single answer
score = judge.evaluate(
    query="What is X?",
    generated_answer="X is...",
    reference_answer="X is actually...",
    context="Context about X"
)

print(f"Score: {score.overall_score}/10")
print(f"Reasoning: {score.reasoning}")
print(f"Strengths: {score.strengths}")
print(f"Weaknesses: {score.weaknesses}")
```

## File Overview 📁

### New Files Created

```
src/rag/evaluation/
├── __init__.py                  # Package exports
├── google_judge.py              # Google Generative AI Judge
└── metrics.py                   # Metrics utilities

Root:
├── LLM_JUDGE_EVALUATION.md      # Full documentation
├── example_google_judge.py      # Example usage
└── evaluate.py                  # Updated evaluation script
```

### Changes to Existing Files

- **evaluate.py**
  - Added Google Judge integration
  - Added train.csv support
  - New CLI options
  - Enhanced documentation

- **pyproject.toml**
  - Added `google-generativeai>=0.3.0` dependency

## Training Data Format 📊

Your `raw/train.csv` should have these columns:

| Column | Required | Example |
|--------|----------|---------|
| query | Yes | "What is the total amount?" |
| answer | Yes | "$22,500.00" |
| context | Yes | "Services Vendor Inc..." |
| sample_number | Optional | "1" |
| tokens | Optional | "150" |
| category | Optional | "invoice" |

## Example Output 📈

```
✓ Google Judge Evaluation Complete
  Answer Relevance: 8.23 ± 1.12
  Answer Faithfulness: 7.89 ± 1.34
  Answer Completeness: 7.54 ± 1.48
  Context Utilization: 7.32 ± 1.58
  Overall Score: 7.74 ± 1.28
```

## Common Use Cases 💡

### Scenario 1: Quick Test
```bash
# Test with 5 samples before full evaluation
python3 evaluate.py --train-csv 5
```

### Scenario 2: Full Evaluation
```bash
# Evaluate all data
python3 evaluate.py --train-csv
```

### Scenario 3: Benchmark Comparison
```bash
# Compare different configurations
export CONFIG_VERSION=v1
python3 evaluate.py --train-csv 20
# ... modify config ...
python3 evaluate.py --train-csv 20
```

### Scenario 4: CI/CD Integration
```bash
# Add to GitHub Actions workflow
python3 evaluate.py --train-csv 100
```

## Troubleshooting 🔧

### "Google API key not found"
```bash
# Set the environment variable
export GOOGLE_API_KEY="your-key"

# Verify it's set
echo $GOOGLE_API_KEY
```

### "google-generativeai is not installed"
```bash
pip install google-generativeai
```

### Evaluation is slow
- Use smaller limit: `--train-csv 5` for testing
- Default batch size is 5 evaluations
- Adjust in code if needed (see full documentation)

### Results don't show Google scores
- Check GOOGLE_API_KEY is set
- Verify google-generativeai is installed
- System falls back to RAGAS if Google Judge unavailable

## Key Differences from Old System 🔄

| Aspect | Old | New |
|--------|-----|-----|
| Primary Judge | RAGAS + Local metrics | Google Generative AI |
| Data Source | Only benchmark JSON | train.csv supported |
| Score Scale | Various (%)  | Unified 0-10 scale |
| Dimensions | 2 metrics | 4 dimensions |
| Reasoning | None | Full reasoning + weaknesses |
| CLI Options | Limited | Rich options |

## Next Steps 🎓

1. **Run quick test**: `python3 evaluate.py --train-csv 5`
2. **Check results**: Look in `results/eval_TIMESTAMP/`
3. **Read full docs**: See `LLM_JUDGE_EVALUATION.md`
4. **Customize**: Modify queries and contexts as needed
5. **Monitor**: Check Google API usage in AI Studio

## Important Notes ⚠️

- **Rate Limiting**: Default 1 second between batches
- **API Quota**: Check your usage in Google AI Studio
- **Free Tier**: Usually sufficient for evaluation
- **Costs**: Gemini 1.5 Flash is cost-efficient for evaluation

## Support & Resources 📚

- [Google Generative AI Docs](https://ai.google.dev/)
- [Full Evaluation Documentation](./LLM_JUDGE_EVALUATION.md)
- [Example Script](./example_google_judge.py)
- [RAG System README](./README.md)

---

**Ready to evaluate?** Start with:
```bash
python3 evaluate.py --train-csv 10
```

Questions? Check the full documentation in `LLM_JUDGE_EVALUATION.md`
