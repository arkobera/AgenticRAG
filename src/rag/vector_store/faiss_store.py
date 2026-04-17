from typing import List, Dict, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.rag.vector_store.base import VectorStore
from src.rag.doc_proc.models import DocumentChunk, RetrievalResult
from src.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore(VectorStore):
    """
    Vector store using FAISS for fast similarity search with:
    - Dense vector similarity search using FAISS
    - BM25 keyword search
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default 384 for all-MiniLM-L6-v2)
        """
        logger.info(f"Initializing FAISSVectorStore with embedding_dim={embedding_dim}")
        self.embedding_dim = embedding_dim
        self.chunks: Dict[str, DocumentChunk] = {}
        self.chunk_ids: List[str] = []
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        logger.debug("FAISS vector store initialized successfully")
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the FAISS store"""
        logger.info(f"Adding {len(chunks)} chunks to FAISS vector store")
        embeddings_list = []
        
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.chunk_ids.append(chunk.chunk_id)
            
            if chunk.embedding:
                embedding = np.array(chunk.embedding, dtype=np.float32)
                # Normalize for better cosine similarity
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                embeddings_list.append(embedding)
            else:
                # Add zero vector if no embedding (will be skipped in search)
                embeddings_list.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.index.add(embeddings_array)
            logger.debug(f"Added {len(embeddings_list)} embeddings to FAISS index")
        
        self._rebuild_bm25_index()
        logger.debug(f"Total chunks in store: {len(self.chunks)}")
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all chunks"""
        self.tokenized_corpus = [
            chunk.content.lower().split()
            for chunk in self.chunks.values()
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """
        Dense vector similarity search using FAISS.
        """
        logger.debug(f"Performing dense vector search with top_k={top_k}")
        if len(self.chunk_ids) == 0:
            logger.warning("Vector store is empty, returning no results")
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        # Normalize query vector
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        # FAISS is L2 distance-based, we convert to similarity
        distances, indices = self.index.search(query_vec, min(top_k, len(self.chunk_ids)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.chunk_ids):  # Valid index
                chunk_id = self.chunk_ids[int(idx)]
                chunk = self.chunks[chunk_id]
                # Convert L2 distance to similarity score (0-1)
                # L2 distance ranges from 0 to 2 for normalized vectors
                similarity = 1 - (distance / 2.0)
                similarity = max(0, similarity)
                
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        content=chunk.content,
                        source_doc=chunk.source_doc,
                        score=float(similarity),
                        search_type="dense_vector",
                    )
                )
        
        logger.debug(f"Dense search returned {len(results)} results")
        return results[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        BM25 keyword search.
        """
        logger.debug(f"Performing BM25 keyword search with top_k={top_k}")
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty, returning no results")
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        results = []
        for chunk_id, score in zip(self.chunk_ids, scores):
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
        """Delete chunks by ID (FAISS doesn't support deletion, so we rebuild)"""
        logger.info(f"Deleting {len(chunk_ids)} chunks from vector store")
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
        
        # Rebuild index
        self.chunk_ids = list(self.chunks.keys())
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        embeddings_list = []
        for chunk_id in self.chunk_ids:
            chunk = self.chunks[chunk_id]
            if chunk.embedding:
                embedding = np.array(chunk.embedding, dtype=np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                embeddings_list.append(embedding)
            else:
                embeddings_list.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.index.add(embeddings_array)
        
        self._rebuild_bm25_index()
        logger.debug(f"Index rebuilt after deletion. Remaining chunks: {len(self.chunks)}")
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk"""
        return self.chunks.get(chunk_id)
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            "total_chunks": len(self.chunks),
            "chunks_with_embeddings": len([c for c in self.chunks.values() if c.embedding]),
            "total_tokens": sum(c.token_count for c in self.chunks.values()),
            "index_size": self.index.ntotal,
        }
