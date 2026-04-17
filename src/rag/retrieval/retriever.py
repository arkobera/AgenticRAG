from typing import Dict, List, Optional, Callable
from src.rag.vector_store.base import VectorStore
from src.rag.doc_proc.models import RetrievalResult
from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Dense vector search (semantic similarity)
    2. Sparse BM25 search (keyword matching)
    
    Uses weighted combination for final ranking.
    Parameters are loaded from config.yaml
    """
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store backend
            embedding_fn: Function to embed text (required for dense search)
            dense_weight: Weight for dense search results (0-1), uses config if None
            sparse_weight: Weight for sparse search results (0-1), uses config if None
        """
        logger.info("Initializing HybridRetriever...")
        # Load from config if not provided
        if dense_weight is None:
            dense_weight = get_config("retriever.dense_weight")
        if sparse_weight is None:
            sparse_weight = get_config("retriever.sparse_weight")
        
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("Retriever weights must be non-negative")

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        total = dense_weight + sparse_weight
        if total == 0:
            raise ValueError("At least one retriever weight must be greater than zero")
        self.dense_weight /= total
        self.sparse_weight /= total
        
        logger.debug(f"Retriever weights: dense={self.dense_weight:.2f}, sparse={self.sparse_weight:.2f}")
        logger.info("HybridRetriever initialized successfully")

    @staticmethod
    def _normalize_scores(results: List[RetrievalResult]) -> Dict[str, float]:
        """Normalize result scores into a stable 0-1 range per retrieval method."""
        if not results:
            return {}

        max_score = max(result.score for result in results)
        if max_score <= 0:
            return {result.chunk_id: 0.0 for result in results}

        return {
            result.chunk_id: min(result.score / max_score, 1.0)
            for result in results
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            use_dense: Include dense vector search
            use_sparse: Include sparse BM25 search
            
        Returns:
            List of RetrievalResult objects ranked by combined score
        """
        logger.debug(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
        
        if not use_dense and not use_sparse:
            logger.error("At least one retrieval method must be enabled")
            raise ValueError("At least one retrieval method must be enabled")

        candidate_k_multiplier = get_config("retriever.candidate_k_multiplier")
        candidate_k = max(top_k * candidate_k_multiplier, top_k)
        combined_results = {}

        if use_dense and self.embedding_fn:
            try:
                query_embd = self.embedding_fn(query)
                dense_res = self.vector_store.search(query_embd, top_k=candidate_k)
                dense_scores = self._normalize_scores(dense_res)
                for res in dense_res:
                    if res.chunk_id not in combined_results:
                        combined_results[res.chunk_id] = {
                            "chunk_id": res.chunk_id,
                            "content": res.content,
                            "source_doc": res.source_doc,
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                        }
                    combined_results[res.chunk_id]["dense_score"] = dense_scores[res.chunk_id]
            except Exception as e:
                print(f"Dense search failed : {e}")

        if use_sparse:
            try:
                sparse_results = self.vector_store.keyword_search(query, top_k=candidate_k)
                sparse_scores = self._normalize_scores(sparse_results)
                for result in sparse_results:
                    if result.chunk_id not in combined_results:
                        combined_results[result.chunk_id] = {
                            "chunk_id": result.chunk_id,
                            "content": result.content,
                            "source_doc": result.source_doc,
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                        }
                    combined_results[result.chunk_id]["sparse_score"] = sparse_scores[result.chunk_id]
            except Exception as e:
                print(f"Sparse search failed: {e}")

        final_results = []
        for chunk_id, data in combined_results.items():
            combined_score = (
                data["dense_score"] * self.dense_weight +
                data["sparse_score"] * self.sparse_weight
            )
            final_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=data["content"],
                    source_doc=data["source_doc"],
                    score=combined_score,
                    search_type="hybrid",
                )
            )
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    def retrieve_with_reasoning(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[List[RetrievalResult], dict]:
        """
        Retrieve with detailed reasoning about scores.
        
        Returns:
            (results, reasoning_dict)
        """
        results = self.retrieve(query, top_k=top_k)
        reasoning = {
            "query": query,
            "retrieval_method": "hybrid",
            "top_k_requested": top_k,
            "results_count": len(results),
            "weights": {
                "dense": self.dense_weight,
                "sparse": self.sparse_weight,
            },
        }
        return results, reasoning
