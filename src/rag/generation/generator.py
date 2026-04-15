from typing import Optional, Callable, List, Dict
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.prompts import GroundingPrompts, ResponseBuilder
from src.rag.doc_proc.models import RetrievalResult
from src.config import get_config


class RAGGenerator:
    """
    Complete RAG pipeline: retrieves context and generates grounded responses.
    Uses HuggingFace LLM for generation.
    Parameters are loaded from config.yaml
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_fn: Callable[[str], str],
        min_context_score: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize RAG Generator.
        
        Args:
            retriever: HybridRetriever instance
            llm_fn: Function that takes a prompt and returns generated text
            min_context_score: Minimum relevance score threshold, uses config if None
            top_k: Number of chunks to retrieve, uses config if None
        """
        # Load from config if not provided
        if min_context_score is None:
            min_context_score = get_config("rag_generator.min_context_score")
        if top_k is None:
            top_k = get_config("rag_generator.top_k")
        
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.min_context_score = min_context_score
        self.top_k = top_k
    
    def generate(self, query: str, use_verification: bool = False) -> Dict:
        """
        Generate response for a query using RAG pipeline.
        
        Args:
            query: User query
            use_verification: Whether to verify response grounding
            
        Returns:
            Dictionary with answer, sources, confidence, etc.
        """
        retrieved_chunks, retrieval_reasoning = self.retriever.retrieve_with_reasoning(
            query=query,
            top_k=self.top_k,
        )

        relevant_chunks = [
            c for c in retrieved_chunks
            if c.score >= self.min_context_score
        ]

        if not relevant_chunks:
            return ResponseBuilder.build_fallback_response(
                query,
                reason="Could not find relevant documentation.",
            )

        context_texts = [c.content for c in relevant_chunks]
        sources = [c.source_doc for c in relevant_chunks]

        prompt = GroundingPrompts.build_rag_prompt(
            query=query,
            context_chunks=context_texts,
            sources=sources,
        )

        # Generate using HuggingFace LLM
        answer = self.llm_fn(prompt)

        response = ResponseBuilder.build_response(
            answer=answer,
            sources=sources,
            confidence=min(
                1.0,
                sum(c.score for c in relevant_chunks) / len(relevant_chunks)
            ) if relevant_chunks else 0.0
        )
        
        response["retrieval_reasoning"] = retrieval_reasoning
        response["num_context_chunks"] = len(relevant_chunks)
        response["verification"] = use_verification
        response["chunk_scores"] = [
            {"chunk_id": c.chunk_id, "score": c.score}
            for c in relevant_chunks
        ]
        return response
    
    def generate_batch(
        self,
        queries: List[str],
        use_verification: bool = False,
    ) -> List[Dict]:
        """Generate responses for multiple queries"""
        return [
            self.generate(query, use_verification=use_verification)
            for query in queries
        ]
    
    def generate_with_followup(
        self,
        query: str,
        followup_questions: List[str],
    ) -> List[Dict]:
        """
        Generate response and handle follow-up questions,
        maintaining context relevance.
        """
        results = [self.generate(query)]
        
        # For follow-ups, generate independently
        for followup in followup_questions:
            followup_result = self.generate(followup)
            results.append(followup_result)
        
        return results
