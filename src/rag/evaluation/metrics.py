"""
Evaluation metrics utilities for RAG system evaluation.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Scores from Google Judge (0-10 scale)
    answer_relevance_scores: List[float]
    answer_faithfulness_scores: List[float]
    answer_completeness_scores: List[float]
    context_utilization_scores: List[float]
    overall_scores: List[float]
    
    def __post_init__(self):
        """Validate that all lists have the same length."""
        lengths = [
            len(self.answer_relevance_scores),
            len(self.answer_faithfulness_scores),
            len(self.answer_completeness_scores),
            len(self.context_utilization_scores),
            len(self.overall_scores),
        ]
        if len(set(lengths)) > 1:
            raise ValueError("All score lists must have the same length")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with statistics for each metric
        """
        return {
            "answer_relevance": self._get_stats(self.answer_relevance_scores),
            "answer_faithfulness": self._get_stats(self.answer_faithfulness_scores),
            "answer_completeness": self._get_stats(self.answer_completeness_scores),
            "context_utilization": self._get_stats(self.context_utilization_scores),
            "overall": self._get_stats(self.overall_scores),
        }
    
    @staticmethod
    def _get_stats(scores: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of scores.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not scores:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        
        sorted_scores = sorted(scores)
        n = len(scores)
        mean = sum(scores) / n
        
        # Calculate median
        if n % 2 == 0:
            median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            median = sorted_scores[n // 2]
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in scores) / n
        std = variance ** 0.5 if variance > 0 else 0.0
        
        return {
            "mean": round(mean, 3),
            "median": round(median, 3),
            "std": round(std, 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer_relevance_scores": self.answer_relevance_scores,
            "answer_faithfulness_scores": self.answer_faithfulness_scores,
            "answer_completeness_scores": self.answer_completeness_scores,
            "context_utilization_scores": self.context_utilization_scores,
            "overall_scores": self.overall_scores,
            "summary": self.get_summary(),
        }
