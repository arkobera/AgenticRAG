#!/usr/bin/env python3
"""
Example script demonstrating Google Generative AI Judge evaluation.

This script shows how to:
1. Load data from train.csv
2. Initialize the Google Judge
3. Evaluate answers
4. View detailed results
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Check for dependencies
try:
    from src.rag.evaluation.google_judge import GoogleGenerativeAIJudge
    print("✓ Google Judge available")
except ImportError:
    print("✗ google-generativeai not installed")
    print("  Install with: pip install google-generativeai")
    exit(1)


def load_sample_data(csv_path: str = "raw/train.csv", limit: int = 5) -> List[Dict]:
    """
    Load sample data from train.csv.
    
    Args:
        csv_path: Path to train.csv
        limit: Maximum samples to load
        
    Returns:
        List of evaluation items
    """
    print(f"\n[1] Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} total rows")
    
    # Limit samples
    df = df.head(limit)
    print(f"   Using first {len(df)} samples")
    
    # Prepare evaluation items
    items = []
    for idx, row in df.iterrows():
        items.append({
            "query": str(row['query']),
            "generated_answer": f"Generated answer for: {row['query'][:50]}...",  # In real use, this comes from RAG
            "reference_answer": str(row['answer']),
            "context": str(row['context'])[:500],  # Truncate for display
        })
    
    return items


def evaluate_with_google_judge(items: List[Dict]) -> List[Dict]:
    """
    Evaluate items using Google Generative AI Judge.
    
    Args:
        items: List of evaluation items
        
    Returns:
        List of evaluation results
    """
    print(f"\n[2] Initializing Google Judge...")
    
    try:
        judge = GoogleGenerativeAIJudge(model_name="gemini-1.5-flash")
        print("   ✓ Judge initialized")
    except Exception as e:
        print(f"   ✗ Error initializing judge: {e}")
        print("   Make sure GOOGLE_API_KEY is set")
        return []
    
    print(f"\n[3] Evaluating {len(items)} items...")
    
    # Evaluate all items
    judge_scores = judge.evaluate_batch(
        items,
        batch_size=3,
        delay_seconds=1.0
    )
    
    # Convert to results
    results = [score.to_dict() for score in judge_scores]
    
    return results


def display_results(results: List[Dict]) -> None:
    """
    Display evaluation results in a formatted way.
    
    Args:
        results: List of evaluation results
    """
    print(f"\n[4] EVALUATION RESULTS")
    print("=" * 80)
    
    if not results:
        print("No results to display")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Query: {result['query'][:80]}...")
        print(f"Scores:")
        print(f"  Relevance:     {result['answer_relevance_score']:.1f}/10")
        print(f"  Faithfulness:  {result['answer_faithfulness_score']:.1f}/10")
        print(f"  Completeness:  {result['answer_completeness_score']:.1f}/10")
        print(f"  Context Usage: {result['context_utilization_score']:.1f}/10")
        print(f"  Overall:       {result['overall_score']:.1f}/10")
        print(f"Reasoning: {result['reasoning'][:150]}...")
        if result['strengths']:
            print(f"Strengths: {', '.join(result['strengths'][:2])}")
        if result['weaknesses']:
            print(f"Weaknesses: {', '.join(result['weaknesses'][:2])}")


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics from results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with statistics
    """
    if not results:
        return {}
    
    def get_stats(scores):
        return {
            "mean": round(sum(scores) / len(scores), 2),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
        }
    
    return {
        "total_evaluated": len(results),
        "answer_relevance": get_stats([r['answer_relevance_score'] for r in results]),
        "answer_faithfulness": get_stats([r['answer_faithfulness_score'] for r in results]),
        "answer_completeness": get_stats([r['answer_completeness_score'] for r in results]),
        "context_utilization": get_stats([r['context_utilization_score'] for r in results]),
        "overall": get_stats([r['overall_score'] for r in results]),
    }


def save_results(results: List[Dict], stats: Dict) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: List of evaluation results
        stats: Statistics dictionary
    """
    output = {
        "evaluation_summary": stats,
        "detailed_results": results,
    }
    
    output_file = Path("results") / "judge_example_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Main execution"""
    print("=" * 80)
    print("GOOGLE GENERATIVE AI JUDGE - EXAMPLE")
    print("=" * 80)
    
    # Load sample data
    try:
        items = load_sample_data(limit=3)
        if not items:
            print("✗ No items to evaluate")
            return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Evaluate with Google Judge
    results = evaluate_with_google_judge(items)
    
    if not results:
        print("✗ Evaluation failed")
        return
    
    # Display results
    display_results(results)
    
    # Calculate and display statistics
    stats = calculate_statistics(results)
    print(f"\n[5] AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"Total Evaluated: {stats['total_evaluated']}")
    print(f"Answer Relevance:     {stats['answer_relevance']}")
    print(f"Answer Faithfulness:  {stats['answer_faithfulness']}")
    print(f"Answer Completeness:  {stats['answer_completeness']}")
    print(f"Context Utilization:  {stats['context_utilization']}")
    print(f"Overall Score:        {stats['overall']}")
    
    # Save results
    save_results(results, stats)
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
