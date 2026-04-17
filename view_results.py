#!/usr/bin/env python3
"""
View and compare RAG pipeline evaluation results
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys


def format_metrics(metrics: Dict[str, Any], prefix: str = "") -> str:
    """Format metrics dictionary for display"""
    lines = []
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    lines.append(f"{prefix}  {sub_key}: {sub_value:.4f}")
                else:
                    lines.append(f"{prefix}  {sub_key}: {sub_value}")
        elif isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, str) and key == "method":
            lines.append(f"{prefix}{key}: {value}")
        elif key not in ["note", "method"]:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)


def view_latest_results():
    """View the latest evaluation results"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("✗ No results directory found. Run 'python3 evaluate.py' first.")
        return False
    
    # Find all eval directories
    eval_dirs = sorted(results_dir.glob("eval_*"), reverse=True)
    
    if not eval_dirs:
        print("✗ No evaluation results found in results/")
        return False
    
    latest = eval_dirs[0]
    print("=" * 80)
    print(f"LATEST EVALUATION RESULTS: {latest.name}")
    print("=" * 80)
    
    # Load and display metrics
    metrics_file = latest / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nTimestamp: {metrics.get('evaluation_timestamp', 'N/A')}")
        
        print("\n[CUSTOM METRICS]")
        print("-" * 80)
        if "custom_metrics" in metrics:
            print(format_metrics(metrics["custom_metrics"], "  "))
        
        print("\n[EVALUATION METRICS]")
        print("-" * 80)
        if "ragas_metrics" in metrics:
            ragas = metrics["ragas_metrics"]
            if isinstance(ragas, dict):
                print(format_metrics(ragas, "  "))
            else:
                print(f"  {ragas}")
        
        print("\n[PIPELINE CONFIGURATION]")
        print("-" * 80)
        if "evaluation_config" in metrics:
            print(format_metrics(metrics["evaluation_config"], "  "))
    
    # Show report
    report_file = latest / "evaluation_report.txt"
    if report_file.exists():
        print("\n[FULL REPORT]")
        print("-" * 80)
        with open(report_file, 'r') as f:
            print(f.read())
    
    return True


def list_all_results():
    """List all evaluation results"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("✗ No results directory found.")
        return False
    
    eval_dirs = sorted(results_dir.glob("eval_*"), reverse=True)
    
    if not eval_dirs:
        print("✗ No evaluation results found in results/")
        return False
    
    print("=" * 80)
    print("ALL EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{'Timestamp':<20} {'Success Rate':<15} {'Faithfulness':<15} {'Relevancy':<15}")
    print("-" * 80)
    
    for eval_dir in eval_dirs:
        metrics_file = eval_dir / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            timestamp = metrics.get("evaluation_timestamp", "Unknown")
            
            # Calculate success rate
            custom = metrics.get("custom_metrics", {})
            total = custom.get("total_queries", 1)
            success = custom.get("successful_answers", 0)
            success_rate = f"{(success/total)*100:.1f}%" if total > 0 else "N/A"
            
            # Get faithfulness and relevancy
            ragas = metrics.get("ragas_metrics", {})
            if isinstance(ragas, dict):
                if "faithfulness_proxy" in ragas:
                    faith = f"{ragas['faithfulness_proxy']['mean']:.3f}"
                elif "faithfulness" in ragas:
                    faith = f"{ragas['faithfulness']['mean']:.3f}"
                else:
                    faith = "N/A"
                
                if "relevancy_proxy" in ragas:
                    relev = f"{ragas['relevancy_proxy']['mean']:.3f}"
                elif "answer_relevancy" in ragas:
                    relev = f"{ragas['answer_relevancy']['mean']:.3f}"
                else:
                    relev = "N/A"
            else:
                faith = "N/A"
                relev = "N/A"
            
            print(f"{timestamp:<20} {success_rate:<15} {faith:<15} {relev:<15}")
    
    print("-" * 80)
    print(f"Total evaluation runs: {len(eval_dirs)}")
    
    return True


def compare_results(run1: str, run2: str):
    """Compare two evaluation runs"""
    results_dir = Path("results")
    
    eval_dir1 = results_dir / f"eval_{run1}"
    eval_dir2 = results_dir / f"eval_{run2}"
    
    if not eval_dir1.exists() or not eval_dir2.exists():
        print(f"✗ One or both evaluation directories not found")
        return False
    
    print("=" * 80)
    print("EVALUATION COMPARISON")
    print("=" * 80)
    
    with open(eval_dir1 / "metrics.json") as f:
        metrics1 = json.load(f)
    with open(eval_dir2 / "metrics.json") as f:
        metrics2 = json.load(f)
    
    print(f"\nComparing: {run1} vs {run2}")
    print("-" * 80)
    
    # Compare custom metrics
    custom1 = metrics1.get("custom_metrics", {})
    custom2 = metrics2.get("custom_metrics", {})
    
    metrics_to_compare = ["average_confidence", "average_context_chunks", "context_coverage"]
    
    print(f"\n{'Metric':<30} {'Run 1':<20} {'Run 2':<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in metrics_to_compare:
        val1 = custom1.get(metric, 0)
        val2 = custom2.get(metric, 0)
        diff = val2 - val1
        diff_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+}"
        
        print(f"{metric:<30} {val1:<20.4f} {val2:<20.4f} {diff_str:<15}")
    
    # Compare evaluation metrics
    ragas1 = metrics1.get("ragas_metrics", {})
    ragas2 = metrics2.get("ragas_metrics", {})
    
    if isinstance(ragas1, dict) and isinstance(ragas2, dict):
        print("\n" + "-" * 80)
        print("Evaluation Metrics Comparison:")
        print("-" * 80)
        
        # Faithfulness
        for key in ["faithfulness", "faithfulness_proxy"]:
            if key in ragas1 and key in ragas2:
                val1 = ragas1[key]["mean"]
                val2 = ragas2[key]["mean"]
                diff = val2 - val1
                print(f"{key:<30} {val1:<20.4f} {val2:<20.4f} {diff:+15.4f}")
        
        # Relevancy
        for key in ["answer_relevancy", "relevancy_proxy"]:
            if key in ragas1 and key in ragas2:
                val1 = ragas1[key]["mean"]
                val2 = ragas2[key]["mean"]
                diff = val2 - val1
                print(f"{key:<30} {val1:<20.4f} {val2:<20.4f} {diff:+15.4f}")
    
    return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Show latest results
        view_latest_results()
    elif sys.argv[1] == "list":
        list_all_results()
    elif sys.argv[1] == "compare" and len(sys.argv) >= 4:
        compare_results(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python3 view_results.py              - Show latest evaluation results")
        print("  python3 view_results.py list         - List all evaluation runs")
        print("  python3 view_results.py compare TIMESTAMP1 TIMESTAMP2 - Compare two runs")
        print()
        print("Examples:")
        print("  python3 view_results.py")
        print("  python3 view_results.py list")
        print("  python3 view_results.py compare 20260415_135302 20260415_135145")


if __name__ == "__main__":
    main()
