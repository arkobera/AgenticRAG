"""
RAG Pipeline Evaluation Script with LLM Judge.
Evaluates the pipeline using Google Generative AI as a judge (primary)
and fallback to RAGAS library and local metrics.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import yaml
from dotenv import load_dotenv

import pandas as pd
from rouge_score import rouge_scorer

# Import RAGAS for evaluation (optional fallback)
try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠ RAGAS library not available or not configured. Using fallback metrics.")

# Import Google Judge
try:
    from src.rag.evaluation.google_judge import GoogleGenerativeAIJudge
    GOOGLE_JUDGE_AVAILABLE = True
except ImportError:
    GOOGLE_JUDGE_AVAILABLE = False
    print("⚠ google-generativeai not available. Install with: pip install google-generativeai")

# Import RAG pipeline components
from src.config import get_config, Config
from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.langchain_setup import setup_embedding_fn, setup_llm
from src.logger import setup_logging, get_logger

load_dotenv()

# Setup logging
logger = get_logger(__name__)


class BenchmarkEvaluator:
    """Evaluate RAG pipeline against benchmark dataset using multiple evaluation methods"""
    
    def __init__(self, benchmark_dir: str = "data/benchmark", results_dir: str = "results", raw_data_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            benchmark_dir: Directory containing benchmark JSON files
            results_dir: Directory to save evaluation results
            raw_data_path: Path to train.csv file in raw folder (optional)
        """
        logger.info(f"Initializing BenchmarkEvaluator with benchmark_dir={benchmark_dir}, results_dir={results_dir}")
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir)
        self.raw_data_path = raw_data_path
        self.results_dir.mkdir(exist_ok=True)
        
        # Benchmark data
        self.queries: List[str] = []
        self.answers: List[str] = []
        self.corpus: Dict[str, str] = {}
        
        # Results
        self.results: Dict[str, Any] = {}
        self.config_snapshot = None
        logger.debug("BenchmarkEvaluator initialized")
        
    def load_benchmark_data(self) -> bool:
        """
        Load benchmark data from JSON files.
        
        Expected structure:
        - queries.json: List of query strings
        - ansers.json: List of ground truth answers (note: typo in filename)
        - corpus.json: Dictionary mapping doc_ids to document content
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        logger.info("Loading benchmark data from JSON files")
        print("\n[1] LOADING BENCHMARK DATA")
        print("-" * 80)
        
        queries_file = self.benchmark_dir / "queries.json"
        answers_file = self.benchmark_dir / "ansers.json"  # Note the typo in original
        corpus_file = self.benchmark_dir / "corpus.json"
        
        try:
            # Load queries
            if queries_file.exists():
                with open(queries_file, 'r') as f:
                    data = json.load(f)
                    self.queries = data if isinstance(data, list) else list(data.values())
                logger.info(f"Loaded {len(self.queries)} queries from {queries_file}")
                print(f"✓ Loaded {len(self.queries)} queries")
            else:
                logger.error(f"Queries file not found: {queries_file}")
                print(f"✗ Queries file not found: {queries_file}")
                return False
            
            # Load answers
            if answers_file.exists():
                with open(answers_file, 'r') as f:
                    data = json.load(f)
                    self.answers = data if isinstance(data, list) else list(data.values())
                logger.info(f"Loaded {len(self.answers)} answers from {answers_file}")
                print(f"✓ Loaded {len(self.answers)} ground truth answers")
            else:
                logger.error(f"Answers file not found: {answers_file}")
                print(f"✗ Answers file not found: {answers_file}")
                return False
            
            # Load corpus
            if corpus_file.exists():
                with open(corpus_file, 'r') as f:
                    data = json.load(f)
                    self.corpus = data if isinstance(data, dict) else {str(i): str(v) for i, v in enumerate(data)}
                logger.info(f"Loaded {len(self.corpus)} corpus documents from {corpus_file}")
                print(f"✓ Loaded {len(self.corpus)} corpus documents")
            else:
                logger.error(f"Corpus file not found: {corpus_file}")
                print(f"✗ Corpus file not found: {corpus_file}")
                return False
            
            # Validate data consistency
            if len(self.queries) != len(self.answers):
                logger.warning(f"Data size mismatch: {len(self.queries)} queries but {len(self.answers)} answers")
                print(f"✗ Mismatch: {len(self.queries)} queries but {len(self.answers)} answers")
                print("  (This is OK if answers are reference answers, not 1-to-1 mapping)")
            
            logger.info(f"Benchmark data loaded successfully")
            print(f"\n✓ Benchmark data loaded successfully")
            print(f"  Queries: {len(self.queries)}")
            print(f"  Answers: {len(self.answers)}")
            print(f"  Corpus: {len(self.corpus)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}", exc_info=True)
            print(f"✗ Error loading benchmark data: {e}")
            return False
    
    def load_from_train_csv(self, csv_path: str = "raw/train.csv", limit: Optional[int] = 2) -> bool:
        """
        Load evaluation data from train.csv in raw folder.
        
        Expected columns: query, answer, context, and optional metadata
        
        Args:
            csv_path: Path to train.csv file
            limit: Maximum number of samples to load (None = all)
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        logger.info(f"Loading evaluation data from train.csv: {csv_path}")
        print("\n[1] LOADING DATA FROM TRAIN.CSV")
        print("-" * 80)
        
        try:
            csv_file = Path(csv_path)
            
            if not csv_file.exists():
                logger.error(f"CSV file not found: {csv_path}")
                print(f"✗ CSV file not found: {csv_path}")
                return False
            
            # Load CSV
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"✓ Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Validate required columns
            required_cols = ['query', 'answer', 'context']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                print(f"✗ Missing required columns: {missing_cols}")
                print(f"  Available columns: {list(df.columns)}")
                return False
            
            # Limit samples if specified
            if limit:
                logger.debug(f"Limiting to {limit} samples")
                df = df.head(limit)
            
            # Extract data
            self.queries = df['query'].astype(str).tolist()
            self.answers = df['answer'].astype(str).tolist()
            contexts = df['context'].astype(str).tolist()
            
            # Build corpus from contexts
            self.corpus = {f"doc_{i}": ctx for i, ctx in enumerate(contexts)}
            
            logger.info(f"Train.csv data loaded successfully: {len(self.queries)} queries")
            print(f"✓ Train.csv data loaded successfully")
            print(f"  Queries: {len(self.queries)}")
            print(f"  Answers: {len(self.answers)}")
            print(f"  Corpus documents: {len(self.corpus)}")
            
            # Print sample
            if self.queries:
                print(f"\n  Sample query: {self.queries[0][:100]}...")
                print(f"  Sample answer: {self.answers[0][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading train.csv: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_with_google_judge(self, generated_answers: List[Dict]) -> Optional[Dict]:
        """
        Evaluate answers using Google Generative AI as judge.
        
        Args:
            generated_answers: List of generated answer dicts
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating answers with Google Generative AI Judge")
        print("\n[4] EVALUATION WITH GOOGLE GENERATIVE AI JUDGE")
        print("-" * 80)
        
        if not GOOGLE_JUDGE_AVAILABLE:
            logger.warning("Google Generative AI not available")
            print("✗ Google Generative AI not available")
            print("  Install with: pip install google-generativeai")
            print("  Set GOOGLE_API_KEY environment variable")
            return None
        
        try:
            # Initialize judge
            print("  Initializing Google Generative AI Judge...")
            logger.debug("Initializing Google Generative AI Judge with gemini-2.5-pro")
            judge = GoogleGenerativeAIJudge(model_name="gemini-2.5-pro")
            logger.info("Google Judge initialized")
            print("✓ Judge initialized")
            
            # Prepare evaluations
            evaluations = []
            for i, answer_dict in enumerate(generated_answers):
                if answer_dict.get('error'):
                    continue
                
                evaluation_item = {
                    "query": answer_dict.get("query", ""),
                    "generated_answer": answer_dict.get("generated_answer", ""),
                    "reference_answer": self.answers[answer_dict.get("query_id", 0)] if answer_dict.get("query_id", 0) < len(self.answers) else "",
                    "context": " ".join([
                        snippet.get("content", "") 
                        for snippet in answer_dict.get("context_snippets", [])
                    ])[:1000],
                }
                evaluations.append(evaluation_item)
            
            if not evaluations:
                print("✗ No valid evaluations to process")
                return None
            
            print(f"  Evaluating {len(evaluations)} answers with Google Judge...")
            
            # Batch evaluate
            judge_scores = judge.evaluate_batch(
                evaluations,
                batch_size=5,
                delay_seconds=1.0
            )
            
            # Extract scores
            scores_dict = {
                "answer_relevance_scores": [],
                "answer_faithfulness_scores": [],
                "answer_completeness_scores": [],
                "context_utilization_scores": [],
                "overall_scores": [],
                "detailed_judgments": []
            }
            
            for score in judge_scores:
                scores_dict["answer_relevance_scores"].append(score.answer_relevance_score)
                scores_dict["answer_faithfulness_scores"].append(score.answer_faithfulness_score)
                scores_dict["answer_completeness_scores"].append(score.answer_completeness_score)
                scores_dict["context_utilization_scores"].append(score.context_utilization_score)
                scores_dict["overall_scores"].append(score.overall_score)
                scores_dict["detailed_judgments"].append(score.to_dict())
            
            # Calculate statistics
            def get_stats(scores):
                if not scores:
                    return {}
                return {
                    "mean": round(sum(scores) / len(scores), 3),
                    "std": round(self._std(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                }
            
            eval_results = {
                "method": "Google Generative AI Judge",
                "answer_relevance": get_stats(scores_dict["answer_relevance_scores"]),
                "answer_faithfulness": get_stats(scores_dict["answer_faithfulness_scores"]),
                "answer_completeness": get_stats(scores_dict["answer_completeness_scores"]),
                "context_utilization": get_stats(scores_dict["context_utilization_scores"]),
                "overall": get_stats(scores_dict["overall_scores"]),
            }
            
            # Print summary
            if eval_results.get("overall"):
                print(f"✓ Google Judge Evaluation Complete")
                print(f"  Answer Relevance: {eval_results['answer_relevance'].get('mean', 0):.2f} ± {eval_results['answer_relevance'].get('std', 0):.2f}")
                print(f"  Answer Faithfulness: {eval_results['answer_faithfulness'].get('mean', 0):.2f} ± {eval_results['answer_faithfulness'].get('std', 0):.2f}")
                print(f"  Answer Completeness: {eval_results['answer_completeness'].get('mean', 0):.2f} ± {eval_results['answer_completeness'].get('std', 0):.2f}")
                print(f"  Context Utilization: {eval_results['context_utilization'].get('mean', 0):.2f} ± {eval_results['context_utilization'].get('std', 0):.2f}")
                print(f"  Overall Score: {eval_results['overall'].get('mean', 0):.2f} ± {eval_results['overall'].get('std', 0):.2f}")
            
            # Store detailed judgments for saving
            eval_results["judge_scores"] = scores_dict
            
            return eval_results
            
        except Exception as e:
            print(f"✗ Google Judge evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_rag_pipeline(self) -> Optional[Tuple]:
        """
        Setup RAG pipeline components.
        
        Returns:
            Tuple of (generator, embedding_fn) or None if setup fails
        """
        logger.info("Setting up RAG pipeline for evaluation")
        print("\n[2] SETTING UP RAG PIPELINE")
        print("-" * 80)
        
        try:
            # Load configuration
            config = Config()
            self.config_snapshot = Config.get_all()
            logger.debug("Configuration loaded and snapshotted")
            print("✓ Configuration loaded")
            
            # Document Processing
            processor = DocumentProcessor()
            document_dir = "data"  # Use the data directory
            docs = processor.load_documents(document_dir)
            logger.info(f"Loaded {len(docs)} documents")
            print(f"✓ Loaded {len(docs)} documents")
            
            chunks = processor.process()
            logger.info(f"Created {len(chunks)} chunks")
            print(f"✓ Created {len(chunks)} chunks")
            
            # Vector Store
            embedding_dim = get_config("embeddings.embedding_dim")
            vector_store = VectorStoreFactory.create("faiss", embedding_dim=embedding_dim)
            logger.info(f"Created FAISS vector store with {embedding_dim}-dim embeddings")
            print(f"✓ Created FAISS vector store")
            
            # Embeddings
            embedding_fn = setup_embedding_fn()
            logger.info("Embedding function initialized")
            print(f"✓ Embedding function initialized")
            
            # Embed chunks
            print("  Embedding chunks...")
            logger.info(f"Embedding {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                if i % max(1, len(chunks) // 4) == 0:
                    logger.debug(f"Embedding progress: {i}/{len(chunks)}")
                    print(f"    Progress: {i}/{len(chunks)}")
                try:
                    embedding = embedding_fn(chunk.content)
                    chunk.embedding = embedding
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {chunk.chunk_id}: {e}")
                    print(f"    Warning: Embedding failed for chunk {chunk.chunk_id}: {e}")
            
            vector_store.add_chunks(chunks)
            stats = vector_store.get_stats()
            logger.info(f"Vector store ready: {stats['total_chunks']} chunks indexed")
            print(f"✓ Vector store ready: {stats['total_chunks']} chunks indexed")
            
            # Retriever
            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_fn=embedding_fn,
            )
            logger.info("Hybrid retriever initialized")
            print(f"✓ Retriever initialized")
            
            # LLM
            llm_fn = setup_llm()
            logger.info("LLM initialized")
            print(f"✓ LLM initialized")
            
            # Generator
            generator = RAGGenerator(
                retriever=retriever,
                llm_fn=llm_fn,
            )
            logger.info("RAG Generator initialized")
            print(f"✓ RAG Generator initialized")
            
            return generator, embedding_fn
            
        except Exception as e:
            logger.error(f"Error setting up pipeline: {e}", exc_info=True)
            print(f"✗ Error setting up pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_answers(self, generator: RAGGenerator) -> List[Dict[str, Any]]:
        """
        Generate answers for all benchmark queries.
        
        Args:
            generator: RAGGenerator instance
            
        Returns:
            List of answer dictionaries with query, answer, and metadata
        """
        logger.info(f"Generating answers for {len(self.queries)} queries")
        print("\n[3] GENERATING ANSWERS")
        print("-" * 80)
        
        generated_answers = []
        
        for i, query in enumerate(self.queries, 1):
            logger.debug(f"Processing query {i}/{len(self.queries)}")
            print(f"  Query {i}/{len(self.queries)}: {query[:60]}...")
            try:
                response = generator.generate(query)
                logger.debug(f"Generated answer for query {i} (confidence: {response.get('confidence', 0):.2f})")
                
                generated_answers.append({
                    "query_id": i - 1,
                    "query": query,
                    "generated_answer": response.get('answer', ''),
                    "source_documents": response.get('sources', []),
                    "confidence": response.get('confidence', 0.0),
                    "num_context_chunks": response.get('num_context_chunks', 0),
                    "context_snippets": [
                        {
                            "chunk_id": c.get('chunk_id'),
                            "score": c.get('score'),
                            "content": c.get('content', '')[:200]
                        }
                        for c in response.get('chunk_scores', [])
                    ]
                })
            except Exception as e:
                logger.error(f"Error generating answer for query {i}: {e}", exc_info=True)
                print(f"    Error generating answer: {e}")
                generated_answers.append({
                    "query_id": i - 1,
                    "query": query,
                    "generated_answer": f"Error: {str(e)}",
                    "error": True
                })
        
        logger.info(f"Generated {len(generated_answers)} answers")
        print(f"\n✓ Generated {len(generated_answers)} answers")
        return generated_answers
    
    def evaluate_with_ragas(self, generated_answers: List[Dict]) -> Optional[Dict]:
        """
        Evaluate answers using metrics (Faithfulness and AnswerRelevancy).
        
        Uses RAGAS if available and OpenAI API key is set,
        otherwise uses local fallback metrics.
        
        Args:
            generated_answers: List of generated answer dicts
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n[4] EVALUATION METRICS")
        print("-" * 80)
        
        eval_results = {}
        
        # Try RAGAS evaluation if available and OpenAI key is set
        if RAGAS_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                print("  Using RAGAS metrics with OpenAI...")
                
                # Prepare data for RAGAS
                eval_data = []
                for answer_dict in generated_answers:
                    if answer_dict.get('error'):
                        continue
                    
                    eval_data.append({
                        "question": answer_dict["query"],
                        "answer": answer_dict["generated_answer"],
                        "contexts": [
                            snippet["content"] 
                            for snippet in answer_dict.get("context_snippets", [])
                        ][:3],
                        "ground_truth": self.answers[answer_dict["query_id"]] if answer_dict["query_id"] < len(self.answers) else ""
                    })
                
                if not eval_data:
                    print("  No valid data for RAGAS evaluation")
                    return self.compute_fallback_metrics(generated_answers)
                
                df = pd.DataFrame(eval_data)
                print(f"  Evaluating {len(df)} answers with RAGAS...")
                
                metrics = [
                    Faithfulness(),
                    AnswerRelevancy(),
                ]
                
                results = evaluate(df, metrics=metrics)
                
                eval_results = {
                    "method": "RAGAS with OpenAI",
                    "faithfulness": {
                        "mean": float(results['faithfulness'].mean()),
                        "std": float(results['faithfulness'].std()),
                        "min": float(results['faithfulness'].min()),
                        "max": float(results['faithfulness'].max()),
                    },
                    "answer_relevancy": {
                        "mean": float(results['answer_relevancy'].mean()),
                        "std": float(results['answer_relevancy'].std()),
                        "min": float(results['answer_relevancy'].min()),
                        "max": float(results['answer_relevancy'].max()),
                    }
                }
                
                print(f"✓ RAGAS Evaluation Complete")
                print(f"  Faithfulness: {eval_results['faithfulness']['mean']:.3f} ± {eval_results['faithfulness']['std']:.3f}")
                print(f"  Answer Relevancy: {eval_results['answer_relevancy']['mean']:.3f} ± {eval_results['answer_relevancy']['std']:.3f}")
                
                return eval_results
                
            except Exception as e:
                print(f"  RAGAS evaluation failed: {e}")
                print("  Falling back to local metrics...")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                print("  OPENAI_API_KEY not set - using fallback metrics")
        
        # Use fallback metrics
        return self.compute_fallback_metrics(generated_answers)
    
    def compute_fallback_metrics(self, generated_answers: List[Dict]) -> Dict:
        """
        Compute fallback metrics when RAGAS/OpenAI is not available.
        
        Metrics include:
        - Faithfulness (based on answer length and confidence)
        - Answer Relevancy (based on ROUGE scores and context usage)
        
        Args:
            generated_answers: List of generated answer dicts
            
        Returns:
            Dictionary with fallback metrics
        """
        print("  Computing local evaluation metrics...")
        
        try:
            faithfulness_scores = []
            relevancy_scores = []
            
            rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            
            for i, answer_dict in enumerate(generated_answers):
                if answer_dict.get('error'):
                    continue
                
                generated = answer_dict.get("generated_answer", "")
                ground_truth = self.answers[answer_dict["query_id"]] if answer_dict["query_id"] < len(self.answers) else ""
                
                # Faithfulness proxy: based on answer length, confidence, and context
                # Higher confidence + more context chunks = higher faithfulness
                confidence = answer_dict.get('confidence', 0)
                context_chunks = answer_dict.get('num_context_chunks', 0)
                answer_length = len(generated.split())
                
                # Faithfulness: Context usage is key
                faithfulness = min(
                    1.0,
                    (confidence * 0.5) +  # 50% from confidence
                    (min(1.0, context_chunks / 3) * 0.5)  # 50% from context usage
                )
                faithfulness_scores.append(max(0, min(1.0, faithfulness)))
                
                # Relevancy: Use ROUGE score against ground truth
                if ground_truth and generated:
                    try:
                        rouge_scores = rouge_scorer_instance.score(ground_truth, generated)
                        relevancy = rouge_scores['rougeL'].fmeasure  # Use ROUGE-L F1 score
                    except:
                        relevancy = 0.5
                else:
                    relevancy = 0.0 if not generated else 0.5
                
                relevancy_scores.append(max(0, min(1.0, relevancy)))
            
            eval_results = {
                "method": "Local Fallback Metrics",
                "faithfulness_proxy": {
                    "mean": float(sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0,
                    "std": float(self._std(faithfulness_scores)) if len(faithfulness_scores) > 1 else 0,
                    "min": float(min(faithfulness_scores)) if faithfulness_scores else 0,
                    "max": float(max(faithfulness_scores)) if faithfulness_scores else 0,
                },
                "relevancy_proxy": {
                    "mean": float(sum(relevancy_scores) / len(relevancy_scores)) if relevancy_scores else 0,
                    "std": float(self._std(relevancy_scores)) if len(relevancy_scores) > 1 else 0,
                    "min": float(min(relevancy_scores)) if relevancy_scores else 0,
                    "max": float(max(relevancy_scores)) if relevancy_scores else 0,
                },
                "note": "These are proxy metrics computed locally without external API calls"
            }
            
            print(f"✓ Local Metrics Computed")
            print(f"  Faithfulness Proxy: {eval_results['faithfulness_proxy']['mean']:.3f} ± {eval_results['faithfulness_proxy']['std']:.3f}")
            print(f"  Relevancy Proxy: {eval_results['relevancy_proxy']['mean']:.3f} ± {eval_results['relevancy_proxy']['std']:.3f}")
            
            return eval_results
            
        except Exception as e:
            print(f"  Error computing fallback metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _std(values: List[float]) -> float:
        """Compute standard deviation"""
        if not values or len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def compute_custom_metrics(self, generated_answers: List[Dict]) -> Dict:
        """
        Compute custom metrics (precision-like metrics).
        
        Args:
            generated_answers: List of generated answer dicts
            
        Returns:
            Dictionary with custom metrics
        """
        logger.info(f"Computing custom metrics for {len(generated_answers)} answers")
        print("\n[5] CUSTOM METRICS")
        print("-" * 80)
        
        try:
            metrics = {
                "total_queries": len(self.queries),
                "successful_answers": sum(1 for a in generated_answers if not a.get('error')),
                "failed_answers": sum(1 for a in generated_answers if a.get('error')),
                "average_confidence": sum(a.get('confidence', 0) for a in generated_answers) / len(generated_answers) if generated_answers else 0,
                "average_context_chunks": sum(a.get('num_context_chunks', 0) for a in generated_answers) / sum(1 for a in generated_answers if not a.get('error')) if generated_answers else 0,
            }
            
            # Calculate retrieval metrics
            with_context = [a for a in generated_answers if a.get('num_context_chunks', 0) > 0]
            metrics["queries_with_context"] = len(with_context)
            metrics["context_coverage"] = len(with_context) / len(generated_answers) if generated_answers else 0
            
            logger.info(f"Custom metrics computed: {metrics['successful_answers']}/{metrics['total_queries']} successful")
            print("✓ Custom Metrics Computed")
            print(f"  Success Rate: {metrics['successful_answers']}/{metrics['total_queries']}")
            print(f"  Avg Confidence: {metrics['average_confidence']:.3f}")
            print(f"  Avg Context Chunks: {metrics['average_context_chunks']:.2f}")
            print(f"  Context Coverage: {metrics['context_coverage']:.1%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing custom metrics: {e}", exc_info=True)
            print(f"✗ Error computing custom metrics: {e}")
            return {}
    
    def save_results(self, generated_answers: List[Dict], ragas_results: Optional[Dict], custom_metrics: Dict) -> str:
        """
        Save evaluation results with configuration snapshot.
        
        Args:
            generated_answers: Generated answers list
            ragas_results: RAGAS evaluation results
            custom_metrics: Custom metrics
            
        Returns:
            Path to results directory
        """
        logger.info("Saving evaluation results")
        print("\n[6] SAVING RESULTS")
        print("-" * 80)
        
        # Create timestamped results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_results_dir = self.results_dir / f"eval_{timestamp}"
        eval_results_dir.mkdir(exist_ok=True)
        logger.debug(f"Created results directory: {eval_results_dir}")
        
        try:
            # Save configuration
            config_file = eval_results_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config_snapshot, f, default_flow_style=False)
            logger.info(f"Configuration saved: {config_file}")
            print(f"✓ Configuration saved: {config_file}")
            
            # Save generated answers
            answers_file = eval_results_dir / "generated_answers.json"
            with open(answers_file, 'w') as f:
                json.dump(generated_answers, f, indent=2)
            logger.info(f"Generated answers saved: {answers_file}")
            print(f"✓ Generated answers saved: {answers_file}")
            
            # Save evaluation metrics
            metrics_file = eval_results_dir / "metrics.json"
            metrics_summary = {
                "evaluation_timestamp": timestamp,
                "custom_metrics": custom_metrics,
                "ragas_metrics": ragas_results if ragas_results else "Not available",
                "evaluation_config": {
                    "embedding_model": get_config("embeddings.model_name"),
                    "llm_model": get_config("llm.model_id"),
                    "top_k": get_config("rag_generator.top_k"),
                    "min_context_score": get_config("rag_generator.min_context_score"),
                    "retriever_dense_weight": get_config("retriever.dense_weight"),
                    "retriever_sparse_weight": get_config("retriever.sparse_weight"),
                }
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info(f"Metrics saved: {metrics_file}")
            print(f"✓ Metrics saved: {metrics_file}")
            
            # Save summary report
            report_file = eval_results_dir / "evaluation_report.txt"
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("RAG PIPELINE EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("EVALUATION CONFIGURATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Embedding Model: {get_config('embeddings.model_name')}\n")
                f.write(f"LLM Model: {get_config('llm.model_id')}\n")
                f.write(f"Top-K: {get_config('rag_generator.top_k')}\n")
                f.write(f"Min Context Score: {get_config('rag_generator.min_context_score')}\n\n")
                
                f.write("CUSTOM METRICS\n")
                f.write("-" * 80 + "\n")
                for key, value in custom_metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
                
                if ragas_results:
                    f.write(f"EVALUATION METRICS ({ragas_results.get('method', 'Unknown method')})\n")
                    f.write("-" * 80 + "\n")
                    
                    if 'faithfulness' in ragas_results:
                        f.write(f"Faithfulness:\n")
                        f.write(f"  Mean: {ragas_results['faithfulness']['mean']:.4f}\n")
                        f.write(f"  Std Dev: {ragas_results['faithfulness']['std']:.4f}\n")
                        f.write(f"  Range: [{ragas_results['faithfulness']['min']:.4f}, {ragas_results['faithfulness']['max']:.4f}]\n\n")
                    
                    if 'faithfulness_proxy' in ragas_results:
                        f.write(f"Faithfulness Proxy:\n")
                        f.write(f"  Mean: {ragas_results['faithfulness_proxy']['mean']:.4f}\n")
                        f.write(f"  Std Dev: {ragas_results['faithfulness_proxy']['std']:.4f}\n")
                        f.write(f"  Range: [{ragas_results['faithfulness_proxy']['min']:.4f}, {ragas_results['faithfulness_proxy']['max']:.4f}]\n\n")
                    
                    if 'answer_relevancy' in ragas_results:
                        f.write(f"Answer Relevancy:\n")
                        f.write(f"  Mean: {ragas_results['answer_relevancy']['mean']:.4f}\n")
                        f.write(f"  Std Dev: {ragas_results['answer_relevancy']['std']:.4f}\n")
                        f.write(f"  Range: [{ragas_results['answer_relevancy']['min']:.4f}, {ragas_results['answer_relevancy']['max']:.4f}]\n\n")
                    
                    if 'relevancy_proxy' in ragas_results:
                        f.write(f"Relevancy Proxy (ROUGE-L):\n")
                        f.write(f"  Mean: {ragas_results['relevancy_proxy']['mean']:.4f}\n")
                        f.write(f"  Std Dev: {ragas_results['relevancy_proxy']['std']:.4f}\n")
                        f.write(f"  Range: [{ragas_results['relevancy_proxy']['min']:.4f}, {ragas_results['relevancy_proxy']['max']:.4f}]\n\n")
                    
                    if 'note' in ragas_results:
                        f.write(f"Note: {ragas_results['note']}\n\n")
                
                f.write("=" * 80 + "\n")
            
            print(f"✓ Report saved: {report_file}")
            
            logger.info(f"All results saved to: {eval_results_dir}")
            print(f"\n✓ All results saved to: {eval_results_dir}")
            return str(eval_results_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)
            print(f"✗ Error saving results: {e}")
            return str(eval_results_dir)
    
    def run_evaluation(self) -> bool:
        """Execute full evaluation pipeline"""
        logger.info("Starting full evaluation pipeline")
        print("=" * 80)
        print("RAG PIPELINE EVALUATION")
        print("=" * 80)
        
        # Load benchmark data
        if not self.load_benchmark_data():
            logger.error("Failed to load benchmark data")
            print("\n✗ Failed to load benchmark data")
            print("  Expected files in data/benchmark/:")
            print("  - queries.json")
            print("  - ansers.json")
            print("  - corpus.json")
            return False
        
        # Setup pipeline
        setup_result = self.setup_rag_pipeline()
        if not setup_result:
            logger.error("Failed to setup RAG pipeline")
            print("\n✗ Failed to setup RAG pipeline")
            return False
        
        generator, embedding_fn = setup_result
        
        # Generate answers
        generated_answers = self.generate_answers(generator)
        
        # Try Google Judge first, then fallback to RAGAS
        logger.info("Starting evaluation with Google Judge")
        google_results = self.evaluate_with_google_judge(generated_answers)
        ragas_results = None
        
        if not google_results:
            logger.info("Google Judge evaluation failed, falling back to RAGAS")
            print("\n  Falling back to RAGAS evaluation...")
            ragas_results = self.evaluate_with_ragas(generated_answers)
        else:
            ragas_results = google_results
        
        # Compute custom metrics
        custom_metrics = self.compute_custom_metrics(generated_answers)
        
        # Save results
        results_path = self.save_results(generated_answers, ragas_results, custom_metrics)
        
        logger.info(f"Evaluation complete, results saved to: {results_path}")
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {results_path}")
        print("=" * 80)
        
        return True
    
    def run_evaluation_with_train_csv(self, csv_path: str = "raw/train.csv", generate_answers: bool = True, limit: Optional[int] = None) -> bool:
        """
        Run evaluation using train.csv data.
        
        Args:
            csv_path: Path to train.csv file
            generate_answers: Whether to generate answers using RAG pipeline
            limit: Maximum number of samples to evaluate
            
        Returns:
            True if evaluation was successful
        """
        logger.info(f"Starting evaluation with train.csv: {csv_path}")
        print("=" * 80)
        print("RAG PIPELINE EVALUATION WITH TRAIN.CSV")
        print("=" * 80)
        
        # Load train.csv data
        if not self.load_from_train_csv(csv_path, limit=limit):
            logger.error("Failed to load train.csv data")
            print("\n✗ Failed to load train.csv data")
            return False
        
        # Setup pipeline
        if generate_answers:
            logger.info("Setting up RAG pipeline for answer generation")
            print("\n[2] SETTING UP RAG PIPELINE")
            print("-" * 80)
            setup_result = self.setup_rag_pipeline()
            if not setup_result:
                logger.error("Failed to setup RAG pipeline")
                print("\n✗ Failed to setup RAG pipeline")
                return False
            
            generator, embedding_fn = setup_result
            
            # Generate answers for queries
            generated_answers = self.generate_answers(generator)
        else:
            # Use provided answers from CSV as "generated" answers
            logger.info("Using answers from CSV")
            print("\n[3] PREPARING ANSWERS FROM CSV")
            print("-" * 80)
            generated_answers = []
            for i, query in enumerate(self.queries):
                generated_answers.append({
                    "query_id": i,
                    "query": query,
                    "generated_answer": self.answers[i] if i < len(self.answers) else "",
                    "source_documents": [],
                    "confidence": 1.0,
                    "num_context_chunks": 0,
                    "context_snippets": [{"content": self.corpus.get(f"doc_{i}", "")}]
                })
            logger.info(f"Prepared {len(generated_answers)} answers from CSV")
            print(f"✓ Prepared {len(generated_answers)} answers from CSV")
        
        # Evaluate with Google Judge (primary), fallback to RAGAS
        print("\n[4] EVALUATION")
        print("-" * 80)
        logger.info("Starting evaluation with Google Judge")
        google_results = self.evaluate_with_google_judge(generated_answers)
        
        if google_results:
            eval_results = google_results
        else:
            logger.info("Google Judge evaluation failed, falling back to RAGAS/Local evaluation")
            print("  Falling back to RAGAS/Local evaluation...")
            eval_results = self.evaluate_with_ragas(generated_answers)
        
        # Compute custom metrics
        custom_metrics = self.compute_custom_metrics(generated_answers)
        
        # Save results
        results_path = self.save_results(generated_answers, eval_results, custom_metrics)
        
        logger.info(f"Evaluation complete, results saved to: {results_path}")
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {results_path}")
        print("=" * 80)
        
        return True


def main():
    """Main evaluation entry point with multiple modes"""
    import sys
    
    # Check for command line arguments
    eval_mode = "benchmark"  # default
    use_google_judge = GOOGLE_JUDGE_AVAILABLE
    csv_limit = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--train-csv", "--csv", "-csv"]:
            eval_mode = "train-csv"
            # Check for limit argument
            if len(sys.argv) > 2:
                try:
                    csv_limit = int(sys.argv[2])
                except ValueError:
                    print(f"Invalid limit: {sys.argv[2]}")
        elif sys.argv[1] in ["--help", "-h"]:
            print_help()
            return True
        elif sys.argv[1] in ["--google-judge"]:
            use_google_judge = True
        elif sys.argv[1] in ["--no-google-judge"]:
            use_google_judge = False
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        benchmark_dir="data/benchmark",
        results_dir="results"
    )
    
    # Run appropriate evaluation mode
    if eval_mode == "train-csv":
        print("\n📊 Running evaluation with TRAIN.CSV data")
        print(f"   Mode: {'With RAG generation' if True else 'Direct answer evaluation'}")
        if csv_limit:
            print(f"   Limit: {csv_limit} samples")
        print(f"   Using Google Judge: {use_google_judge}")
        success = evaluator.run_evaluation_with_train_csv(
            csv_path="raw/train.csv",
            generate_answers=True,
            limit=csv_limit
        )
    else:
        print("\n📊 Running evaluation with BENCHMARK data")
        success = evaluator.run_evaluation()
    
    if not success:
        print("\n⚠ Evaluation did not complete successfully")
        if eval_mode == "benchmark":
            print("\nTo create sample benchmark data, you can use:")
            print("  python3 create_sample_benchmark.py")
        print("\nTo evaluate with train.csv, use:")
        print("  python3 evaluate.py --train-csv [limit]")
        return False
    
    return True


def print_help():
    """Print help message"""
    print("""
RAG Pipeline Evaluation Script

Usage:
  python3 evaluate.py                    # Evaluate with benchmark data (default)
  python3 evaluate.py --train-csv [n]    # Evaluate with train.csv (optional: limit to n samples)
  python3 evaluate.py --google-judge     # Force using Google Generative AI Judge
  python3 evaluate.py --no-google-judge  # Force using fallback evaluation
  python3 evaluate.py --help              # Show this help message

Examples:
  python3 evaluate.py                         # Use benchmark data with Google Judge
  python3 evaluate.py --train-csv 10          # Evaluate first 10 samples from train.csv
  python3 evaluate.py --train-csv 100         # Evaluate first 100 samples from train.csv
  python3 evaluate.py --train-csv             # Evaluate all samples from train.csv

Requirements:
  - For Google Generative AI Judge: pip install google-generativeai
  - Set GOOGLE_API_KEY environment variable with your API key

Evaluation Methods:
  1. Google Generative AI Judge (Gemini) - Primary method
     - Score on 0-10 scale
     - Evaluates: Relevance, Faithfulness, Completeness, Context Utilization
     - Requires: google-generativeai, GOOGLE_API_KEY

  2. RAGAS Metrics - Fallback method
     - Faithfulness and AnswerRelevancy metrics
     - Requires: ragas, OpenAI API key (optional)

  3. Local Fallback Metrics
     - ROUGE scores, confidence-based metrics
     - No external dependencies required
""")



if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
