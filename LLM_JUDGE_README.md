# LLM Judge Evaluation Pipeline - Complete Reference

## 🎯 What Was Built

A complete **LLM-based evaluation pipeline** that uses Google's Generative AI (Gemini) to intelligently judge RAG-generated answers. The system evaluates on 4 dimensions and provides detailed reasoning for each evaluation.

## 📦 What You Have Now

### New Core Modules
```
✓ src/rag/evaluation/
  ├── __init__.py          - Package initialization
  ├── google_judge.py      - Google Generative AI Judge
  └── metrics.py           - Metrics utilities
```

### Enhanced Scripts
```
✓ evaluate.py             - Updated with Google Judge integration
✓ example_google_judge.py - Executable example
✓ pyproject.toml          - Added google-generativeai dependency
```

### Documentation
```
✓ LLM_JUDGE_EVALUATION.md               - Full documentation
✓ GOOGLE_JUDGE_QUICKSTART.md           - Quick start guide
✓ EVALUATION_IMPLEMENTATION_SUMMARY.md - Implementation details
✓ This file                            - Complete reference
```

## 🚀 Quick Start (3 Steps)

### Step 1: Install
```bash
pip install google-generativeai
# OR
pip install -e .  # Install with all dependencies
```

### Step 2: Configure API Key
```bash
export GOOGLE_API_KEY="your-api-key-from-aistudio.google.com"
```

### Step 3: Run Evaluation
```bash
# Quick test with 5 samples
python3 evaluate.py --train-csv 5

# Full evaluation
python3 evaluate.py --train-csv
```

## 📊 Evaluation Dimensions Explained

The Google Judge evaluates answers on **4 dimensions**, each scored 0-10:

### 1. Answer Relevance
**What**: How well does the answer address the query?
- **9-10**: Perfectly addresses query with all details
- **7-8**: Good response, addresses main points
- **5-6**: Partially relevant, missing some details
- **3-4**: Tangentially related
- **0-2**: Off-topic or irrelevant

### 2. Answer Faithfulness
**What**: Is the answer grounded in provided context?
- **9-10**: Completely grounded, no hallucinations
- **7-8**: Mostly grounded with minimal unsupported claims
- **5-6**: Partially grounded, some unsupported info
- **3-4**: Mostly unsupported claims
- **0-2**: Contradicts or ignores context

### 3. Answer Completeness
**What**: Does it cover all important aspects?
- **9-10**: Comprehensive, covers all aspects
- **7-8**: Good coverage of main aspects
- **5-6**: Covers half of important aspects
- **3-4**: Limited coverage
- **0-2**: Missing most important aspects

### 4. Context Utilization
**What**: How effectively is the provided context used?
- **9-10**: Excellent use of context facts and details
- **7-8**: Good integration of context
- **5-6**: Some context use
- **3-4**: Minimal context use
- **0-2**: Ignores provided context

### 5. Overall Score
**What**: Composite score combining all dimensions
- Average or weighted combination of above

## 💻 Usage Examples

### Example 1: Basic Evaluation
```bash
python3 evaluate.py --train-csv 10
```
Evaluates first 10 samples from train.csv with RAG generation.

### Example 2: Full Dataset
```bash
python3 evaluate.py --train-csv
```
Evaluates all samples from train.csv.

### Example 3: With Specific Limit
```bash
python3 evaluate.py --train-csv 50
```
Evaluates first 50 samples.

### Example 4: Benchmark Data
```bash
python3 evaluate.py
```
Uses original benchmark data from data/benchmark/.

### Example 5: Run Example Script
```bash
python3 example_google_judge.py
```
Runs a standalone example with sample data.

### Example 6: Programmatic Usage
```python
from evaluate import BenchmarkEvaluator

# Method 1: With train.csv
evaluator = BenchmarkEvaluator()
evaluator.run_evaluation_with_train_csv(
    csv_path="raw/train.csv",
    generate_answers=True,
    limit=10
)

# Method 2: Direct judge usage
from src.rag.evaluation.google_judge import GoogleGenerativeAIJudge

judge = GoogleGenerativeAIJudge()
score = judge.evaluate(
    query="What is X?",
    generated_answer="X is...",
    reference_answer="X is...",
    context="..."
)
print(f"Overall: {score.overall_score:.1f}/10")
print(f"Relevance: {score.answer_relevance_score:.1f}/10")
print(f"Faithfulness: {score.answer_faithfulness_score:.1f}/10")
print(f"Completeness: {score.answer_completeness_score:.1f}/10")
print(f"Context: {score.context_utilization_score:.1f}/10")
```

## 📈 Understanding Your Results

### Result Directory Structure
```
results/eval_20260415_135302/
├── config.yaml                 # Configuration used
├── generated_answers.json       # All answers + judge scores
├── metrics.json               # Aggregated statistics
└── evaluation_report.txt      # Human-readable report
```

### Sample metrics.json Output
```json
{
  "google_judge_metrics": {
    "answer_relevance": {
      "mean": 8.23,
      "std": 1.12,
      "min": 5.0,
      "max": 9.8
    },
    "answer_faithfulness": {
      "mean": 7.89,
      "std": 1.34
    },
    "answer_completeness": {
      "mean": 7.54,
      "std": 1.48
    },
    "context_utilization": {
      "mean": 7.32,
      "std": 1.58
    },
    "overall": {
      "mean": 7.74,
      "std": 1.28,
      "min": 4.5,
      "max": 9.8
    }
  }
}
```

### Reading the Stats
- **mean**: Average score across all samples
- **std**: Standard deviation (variability)
- **min/max**: Range of scores

Lower std = more consistent. Higher mean = better quality.

## ⚙️ Configuration

### Environment Variables
```bash
# Required for Google Judge
export GOOGLE_API_KEY="your-api-key"

# Optional - already in .env handling
# export OPENAI_API_KEY="for RAGAS fallback"
```

### Command Line Options
```bash
python3 evaluate.py --train-csv [limit]    # Evaluate with train.csv
python3 evaluate.py --benchmark            # Use benchmark data (default)
python3 evaluate.py --google-judge         # Force Google Judge
python3 evaluate.py --no-google-judge      # Force fallback evaluation
python3 evaluate.py --help                 # Show help
```

## 🔄 Evaluation Pipeline Flow

```
START
  ↓
Load Data (train.csv or benchmark)
  ↓
Setup RAG Pipeline
  ├─ Document Processing
  ├─ Vector Store Creation
  ├─ Generate Embeddings
  └─ Initialize Retriever & Generator
  ↓
Generate Answers
  ├─ For each query
  ├─ Retrieve context
  └─ Generate answer
  ↓
Evaluate Answers (Try in order)
  ├─ Google Judge (if available)
  ├─ RAGAS (if available)
  └─ Local Metrics (always available)
  ↓
Compute Statistics
  ├─ Mean, std, min, max for each dimension
  ├─ Custom metrics
  └─ Success rates
  ↓
Save Results
  ├─ metrics.json
  ├─ generated_answers.json
  ├─ evaluation_report.txt
  └─ config.yaml
  ↓
END
```

## 🛠️ Troubleshooting

### Issue 1: "Google API key not found"
**Solution:**
```bash
# Check if it's set
echo $GOOGLE_API_KEY

# Set it
export GOOGLE_API_KEY="your-key"

# Or add to .env file
echo "GOOGLE_API_KEY=your-key" >> .env
```

### Issue 2: "google-generativeai not installed"
**Solution:**
```bash
pip install google-generativeai
# Or
pip install -e .
```

### Issue 3: "No module named src.rag.evaluation"
**Solution:** Verify the evaluation module exists:
```bash
ls -la src/rag/evaluation/
# Should show: __init__.py  google_judge.py  metrics.py
```

### Issue 4: Evaluation is slow
**Solution:** Use smaller sample size for testing:
```bash
# Test with 5 samples first
python3 evaluate.py --train-csv 5

# Then run full if OK
python3 evaluate.py --train-csv
```

### Issue 5: Results don't show Google Judge scores
**Solution:** Check in this order:
1. `echo $GOOGLE_API_KEY` - Is it set?
2. `pip show google-generativeai` - Is it installed?
3. Check logs for fallback messages
4. System will use RAGAS if Google Judge unavailable

## 📚 Documentation Reference

### For Quick Start
→ **GOOGLE_JUDGE_QUICKSTART.md**

### For Full Details
→ **LLM_JUDGE_EVALUATION.md**

### For Implementation Details
→ **EVALUATION_IMPLEMENTATION_SUMMARY.md**

## 🔐 Security Notes

- API keys stored in environment variables (not in code)
- No credentials in config files
- Proper error handling without exposing sensitive data
- Google API only called when needed

## 💰 Cost Considerations

### Free Tier (Gemini)
- Sufficient for most evaluation tasks
- Usually adequate for 100+ evaluations/day
- Check [Google AI Studio](https://aistudio.google.com/app/apikey) for quotas

### Pricing
- Gemini 1.5 Flash: Very affordable
- Gemini 1.5 Pro: Higher quality, more expensive
- Check Google's pricing for current rates

### Optimization Tips
1. Use smaller limits for testing: `--train-csv 5`
2. Batch processing reduces overhead
3. Re-use results (don't re-evaluate same data)
4. Monitor usage in Google AI Studio

## 🔄 Data Flow

### train.csv → Evaluation
```
train.csv (columns: query, answer, context, ...)
    ↓
Read with pandas
    ↓
Extract: queries, answers, contexts
    ↓
Generate RAG answers (or use from CSV)
    ↓
Prepare evaluation items with:
    - Original query
    - Generated answer (from RAG)
    - Reference answer (from CSV)
    - Context (from CSV)
    ↓
Pass to Google Judge
    ↓
Receive scores (0-10 scale)
    ↓
Aggregate statistics
    ↓
Save results
```

## 📊 Metrics Comparison

### Before (Old System)
| Aspect | Value |
|--------|-------|
| Primary Judge | RAGAS + Local |
| Score Scale | Various (%) |
| Dimensions | 2-3 |
| Reasoning | None |
| Data Support | Benchmark only |

### After (New System)
| Aspect | Value |
|--------|-------|
| Primary Judge | Google Generative AI |
| Score Scale | 0-10 (unified) |
| Dimensions | 4 (+ overall) |
| Reasoning | Full reasoning |
| Data Support | train.csv + benchmark |

## ✅ Backward Compatibility

✓ Old `python3 evaluate.py` still works  
✓ Benchmark data still supported  
✓ RAGAS metrics still available as fallback  
✓ Local metrics always available  
✓ No breaking changes to existing code

## 🎓 Learning Path

1. **Start here**: Run a quick test
   ```bash
   python3 evaluate.py --train-csv 5
   ```

2. **Understand scores**: Read scoring explanation above

3. **Check results**: Look in `results/eval_TIMESTAMP/`

4. **Run full evaluation**: 
   ```bash
   python3 evaluate.py --train-csv
   ```

5. **Explore details**:
   - Read `metrics.json` for statistics
   - Read `evaluation_report.txt` for summary
   - Check `generated_answers.json` for details

6. **Advanced usage**: See LLM_JUDGE_EVALUATION.md

## 🚀 Next Steps

1. **Immediate**: Run evaluation
   ```bash
   python3 evaluate.py --train-csv 10
   ```

2. **Monitor**: Check results directory
   ```bash
   ls -la results/eval_*/
   cat results/eval_*/metrics.json
   ```

3. **Iterate**: Improve RAG system based on scores

4. **Automate**: Add to CI/CD pipeline

5. **Scale**: Evaluate full dataset when ready

## 📞 Support

- **Quick questions**: See GOOGLE_JUDGE_QUICKSTART.md
- **Technical details**: See LLM_JUDGE_EVALUATION.md
- **Implementation**: See EVALUATION_IMPLEMENTATION_SUMMARY.md
- **Google API**: https://ai.google.dev/

---

**Status**: ✅ Implementation Complete

Ready to evaluate? Start with:
```bash
python3 evaluate.py --train-csv 5
```
