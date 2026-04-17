from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.rag.vector_store.base import VectorStore
from src.rag.doc_proc.models import DocumentChunk, RetrievalResult
from src.logger import get_logger

logger = get_logger(__name__)

class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store with:
    - Dense vector similarity search using cosine distance
    - BM25 keyword search
    """
    def __init__(self):
        logger.info("Initializing InMemoryVectorStore")
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        logger.debug("InMemory vector store initialized successfully")
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the store"""
        logger.info(f"Adding {len(chunks)} chunks to in-memory store")
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            if chunk.embedding:
                self.embeddings[chunk.chunk_id] = np.array(chunk.embedding)
        logger.debug(f"Total chunks: {len(self.chunks)}, chunks with embeddings: {len(self.embeddings)}")
        self._rebuild_bm25_index()
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all chunks"""
        self.tokenized_corpus = [
            chunk.content.lower().split()
            for chunk in self.chunks.values()
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    def search(self,query_embedding: List[float],top_k: int = 5,) -> List[RetrievalResult]:
        """
        Dense vector similarity search.
        Uses cosine similarity.
        """
        logger.debug(f"Performing dense vector search with top_k={top_k}")
        if not self.embeddings:
            logger.warning("No embeddings in store, returning empty results")
            return []
        query_vec = np.array(query_embedding)
        results = []
        for chunk_id, embedding in self.embeddings.items():
            similarity = np.dot(query_vec, embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding) + 1e-10
            )
            score = (similarity + 1) / 2
            chunk = self.chunks[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    source_doc=chunk.source_doc,
                    score=float(score),
                    search_type="dense_vector",
                )
            )
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Dense search returned {len(results)} results")
        return results[:top_k]    
    def keyword_search(self,query: str,top_k: int = 5,) -> List[RetrievalResult]:
        """
        BM25 keyword search.
        Good for exact terms and product codes.
        """
        logger.debug(f"Performing BM25 keyword search with top_k={top_k}")
        if not self.bm25:
            logger.warning("BM25 index not initialized, returning empty results")
            return []
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        chunk_ids = list(self.chunks.keys())
        results = []    
        for chunk_id, score in zip(chunk_ids, scores):
            if score > 0:  
                chunk = self.chunks[chunk_id]
                normalized_score = min(float(score) / 100.0, 1.0)
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        content=chunk.content,
                        source_doc=chunk.source_doc,
                        score=normalized_score,
                        search_type="bm25_keyword",
                    )
                )
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"BM25 search returned {len(results)} results")
        return results[:top_k]
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID"""
        logger.info(f"Deleting {len(chunk_ids)} chunks from in-memory store")
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
            self.embeddings.pop(chunk_id, None)
        logger.debug(f"Remaining chunks: {len(self.chunks)}")
        self._rebuild_bm25_index()
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk"""
        return self.chunks.get(chunk_id)
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            "total_chunks": len(self.chunks),
            "chunks_with_embeddings": len(self.embeddings),
            "total_tokens": sum(c.token_count for c in self.chunks.values()),
        }