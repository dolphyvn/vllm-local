"""
financial_memory_manager.py - Enhanced Memory Manager for Trading Analysis
Integrates financial-specific embedding models for better trading context retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np

from memory import MemoryManager
from financial_embedding_config import FinancialEmbeddingConfig, preprocess_financial_text

logger = logging.getLogger(__name__)

class FinancialMemoryManager(MemoryManager):
    """
    Enhanced memory manager with financial-specific embedding models
    """

    def __init__(self,
                 collection_name: str = "financial_trading_memory",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "finance-news-v1"):
        """
        Initialize financial memory manager with custom embedding model

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Financial embedding model key
        """
        super().__init__(collection_name, persist_directory)
        self.embedding_model_config = FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS.get(
            embedding_model, FinancialEmbeddingConfig.get_recommended_model()
        )
        self.financial_embedding_model = None
        self._init_financial_embeddings()

    def _init_financial_embeddings(self):
        """Initialize financial-specific embedding model"""
        try:
            # Import sentence-transformers
            from sentence_transformers import SentenceTransformer

            model_name = self.embedding_model_config["model_name"]
            logger.info(f"Loading financial embedding model: {model_name}")

            # Load the financial embedding model
            self.financial_embedding_model = SentenceTransformer(model_name)
            logger.info(f"✅ Financial embedding model loaded: {model_name}")
            logger.info(f"   Dimensions: {self.embedding_model_config['dimensions']}")
            logger.info(f"   Max sequence length: {self.embedding_model_config['max_sequence_length']}")

        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to default ChromaDB embeddings.")
            logger.warning("Install with: pip install sentence-transformers")
            self.financial_embedding_model = None
        except Exception as e:
            logger.error(f"Failed to load financial embedding model: {e}")
            self.financial_embedding_model = None

    def _get_embedding_function(self):
        """
        Get the embedding function for ChromaDB

        Returns:
            Embedding function or None
        """
        if self.financial_embedding_model is None:
            return None

        def financial_embed(texts: List[str]) -> List[List[float]]:
            """
            Financial embedding function

            Args:
                texts: List of texts to embed

            Returns:
                List of embedding vectors
            """
            try:
                # Preprocess financial texts
                preprocessed_texts = [preprocess_financial_text(text) for text in texts]

                # Generate embeddings
                embeddings = self.financial_embedding_model.encode(
                    preprocessed_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32
                )

                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Financial embedding failed: {e}")
                # Fallback to simple embedding
                return [[0.0] * self.embedding_model_config["dimensions"] for _ in texts]

        return financial_embed

    def _ensure_initialized(self):
        """Override initialization with financial embeddings"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get custom embedding function
            embedding_function = self._get_embedding_function()

            # Create collection with custom embedding function
            if embedding_function:
                from chromadb.utils import embedding_functions

                # Create custom embedding function wrapper
                class FinancialEmbeddingFunction:
                    def __init__(self, embed_func):
                        self.embed_func = embed_func

                    def __call__(self, input: List[str]) -> List[List[float]]:
                        return self.embed_func(input)

                custom_embedding = FinancialEmbeddingFunction(embedding_function)

                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=custom_embedding,
                    metadata={
                        "description": "Financial trading memory storage",
                        "embedding_model": self.embedding_model_config["model_name"],
                        "dimensions": self.embedding_model_config["dimensions"]
                    }
                )
            else:
                # Fallback to default ChromaDB embeddings
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Financial trading memory storage (default embeddings)"}
                )

            self._initialized = True
            logger.info(f"✅ Financial memory manager initialized with collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize financial memory manager: {e}")
            raise

    def add_trading_memory(self,
                          user_query: str,
                          ai_response: str,
                          market_context: Optional[Dict[str, Any]] = None,
                          trading_signals: Optional[List[str]] = None,
                          confidence_score: Optional[float] = None):
        """
        Add trading-specific memory with enhanced financial context

        Args:
            user_query: User's trading question
            ai_response: AI's trading analysis
            market_context: Market data context (prices, indicators, etc.)
            trading_signals: List of trading signals mentioned
            confidence_score: Confidence score of the analysis
        """
        try:
            self._ensure_initialized()
            if not self._initialized:
                return

            # Create enhanced metadata for trading
            timestamp = datetime.now().isoformat()
            memory_id = f"trading_{timestamp}_{hash(user_query) % 10000}"

            # Enhanced metadata with trading-specific information
            trading_metadata = {
                "timestamp": timestamp,
                "type": "trading_analysis",
                "user_query_length": len(user_query),
                "ai_response_length": len(ai_response),
                "has_market_context": market_context is not None,
                "signal_count": len(trading_signals) if trading_signals else 0,
                "confidence_score": confidence_score or 0.0
            }

            # Add market context information
            if market_context:
                trading_metadata.update({
                    "market_context": json.dumps(market_context),
                    "symbols_mentioned": len(market_context.get("symbols", [])),
                    "timeframes": market_context.get("timeframes", [])
                })

            # Add trading signals
            if trading_signals:
                trading_metadata["trading_signals"] = json.dumps(trading_signals)

            # Create enhanced text for embedding
            enhanced_text = self._create_enhanced_trading_text(
                user_query, ai_response, market_context, trading_signals
            )

            # Add to collection
            self.collection.add(
                documents=[enhanced_text],
                metadatas=[trading_metadata],
                ids=[memory_id]
            )

            logger.info(f"✅ Added trading memory: {memory_id}")
            logger.debug(f"   Query: {user_query[:100]}...")
            logger.debug(f"   Signals: {trading_signals[:3] if trading_signals else []}")

        except Exception as e:
            logger.error(f"Failed to add trading memory: {e}")
            raise

    def _create_enhanced_trading_text(self,
                                     user_query: str,
                                     ai_response: str,
                                     market_context: Optional[Dict[str, Any]],
                                     trading_signals: Optional[List[str]]) -> str:
        """
        Create enhanced text for better financial embeddings

        Args:
            user_query: User's query
            ai_response: AI's response
            market_context: Market data
            trading_signals: Trading signals

        Returns:
            Enhanced text for embedding
        """
        parts = [f"Trading Query: {user_query}", f"Analysis: {ai_response}"]

        # Add market context
        if market_context:
            if market_context.get("symbols"):
                parts.append(f"Symbols: {', '.join(market_context['symbols'])}")
            if market_context.get("indicators"):
                parts.append(f"Indicators: {', '.join(market_context['indicators'])}")
            if market_context.get("timeframes"):
                parts.append(f"Timeframes: {', '.join(market_context['timeframes'])}")

        # Add trading signals
        if trading_signals:
            parts.append(f"Signals: {', '.join(trading_signals)}")

        return " | ".join(parts)

    def search_trading_memories(self,
                               query: str,
                               market_context: Optional[Dict[str, Any]] = None,
                               min_confidence: float = 0.0,
                               n: int = 5) -> List[Dict[str, Any]]:
        """
        Search trading memories with enhanced financial filtering

        Args:
            query: Trading query
            market_context: Current market context for better matching
            min_confidence: Minimum confidence score filter
            n: Number of results to return

        Returns:
            List of relevant trading memories
        """
        try:
            self._ensure_initialized()
            if not self._initialized:
                return []

            # Build enhanced query
            enhanced_query = query
            if market_context:
                context_parts = []
                if market_context.get("symbols"):
                    context_parts.append(f"Symbols: {', '.join(market_context['symbols'])}")
                if market_context.get("indicators"):
                    context_parts.append(f"Indicators: {', '.join(market_context['indicators'])}")
                if context_parts:
                    enhanced_query += " | " + " | ".join(context_parts)

            # Query with confidence filter
            where_filter = {"confidence_score": {"$gte": min_confidence}} if min_confidence > 0 else None

            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n,
                where=where_filter
            )

            # Format results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distances = results['distances'][0] if results['distances'] else []

                    memories.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity_score': 1.0 - distances[i] if i < len(distances) else 0.0,
                        'memory_type': metadata.get('type', 'trading_analysis')
                    })

            logger.info(f"🔍 Found {len(memories)} trading memories for query: {query[:50]}...")
            return memories

        except Exception as e:
            logger.error(f"Failed to search trading memories: {e}")
            return []

    def get_similar_trading_scenarios(self,
                                     current_scenario: str,
                                     timeframe: Optional[str] = None,
                                     symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find similar trading scenarios from historical memories

        Args:
            current_scenario: Current trading scenario description
            timeframe: Timeframe filter (e.g., "daily", "weekly")
            symbols: List of symbols to filter by

        Returns:
            List of similar historical scenarios
        """
        try:
            self._ensure_initialized()
            if not self._initialized:
                return []

            # Build where filters
            where_filter = {"type": "trading_analysis"}

            if timeframe or symbols:
                and_conditions = []
                if timeframe:
                    and_conditions.append({"timeframes": {"$in": [timeframe]}})
                if symbols:
                    and_conditions.append({"symbols_mentioned": {"$gte": len(symbols)}})

                if and_conditions:
                    where_filter = {"$and": [where_filter] + and_conditions}

            results = self.collection.query(
                query_texts=[current_scenario],
                n_results=10,
                where=where_filter
            )

            # Format and rank by similarity
            scenarios = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distances = results['distances'][0] if results['distances'] else []

                    scenarios.append({
                        'scenario_text': doc,
                        'metadata': metadata,
                        'similarity_score': 1.0 - distances[i] if i < len(distances) else 0.0,
                        'timestamp': metadata.get('timestamp', ''),
                        'confidence': metadata.get('confidence_score', 0.0)
                    })

            # Sort by similarity score
            scenarios.sort(key=lambda x: x['similarity_score'], reverse=True)

            logger.info(f"📊 Found {len(scenarios)} similar scenarios for current trading setup")
            return scenarios[:5]  # Return top 5 most similar

        except Exception as e:
            logger.error(f"Failed to find similar trading scenarios: {e}")
            return []

    def get_trading_performance_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get statistics about trading memory performance

        Args:
            days_back: Number of days to look back

        Returns:
            Trading performance statistics
        """
        try:
            self._ensure_initialized()
            if not self._initialized:
                return {}

            # Calculate date threshold
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()

            # Query recent trading memories
            results = self.collection.query(
                query_texts=["trading analysis"],
                n_results=1000,
                where={
                    "$and": [
                        {"type": "trading_analysis"},
                        {"timestamp": {"$gte": threshold_date}}
                    ]
                }
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                return {}

            metadatas = results['metadatas'][0]

            # Calculate statistics
            total_memories = len(metadatas)
            avg_confidence = np.mean([m.get('confidence_score', 0) for m in metadatas])
            memories_with_signals = sum(1 for m in metadatas if m.get('signal_count', 0) > 0)
            memories_with_context = sum(1 for m in metadatas if m.get('has_market_context', False))

            stats = {
                'total_trading_memories': total_memories,
                'average_confidence_score': round(avg_confidence, 3),
                'memories_with_signals': memories_with_signals,
                'memories_with_market_context': memories_with_context,
                'signal_coverage_ratio': round(memories_with_signals / total_memories, 3) if total_memories > 0 else 0,
                'context_coverage_ratio': round(memories_with_context / total_memories, 3) if total_memories > 0 else 0,
                'period_days': days_back,
                'embedding_model': self.embedding_model_config['model_name']
            }

            logger.info(f"📈 Trading memory stats for last {days_back} days: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get trading performance stats: {e}")
            return {}

if __name__ == "__main__":
    # Demonstrate financial memory manager
    print("🎯 Financial Memory Manager Test")
    print("=" * 40)

    try:
        # Initialize financial memory manager
        fm_manager = FinancialMemoryManager()
        print("✅ Financial memory manager initialized")

        # Test adding a trading memory
        fm_manager.add_trading_memory(
            user_query="What's the outlook for AAPL stock?",
            ai_response="Based on technical indicators, AAPL shows bullish momentum with RSI at 65 and MACD showing positive crossover.",
            market_context={
                "symbols": ["AAPL"],
                "indicators": ["RSI", "MACD"],
                "timeframes": ["daily"]
            },
            trading_signals=["bullish", "momentum"],
            confidence_score=0.85
        )
        print("✅ Test trading memory added")

        # Test searching
        results = fm_manager.search_trading_memories("AAPL stock analysis")
        print(f"✅ Found {len(results)} trading memories")

        # Get performance stats
        stats = fm_manager.get_trading_performance_stats(30)
        print(f"✅ Performance stats: {stats}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()