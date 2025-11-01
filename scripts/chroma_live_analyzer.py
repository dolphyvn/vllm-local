#!/usr/bin/env python3
"""
ChromaDB Live Analysis Storage - Phase 3 Implementation
Handles storage and retrieval of live trading analysis in ChromaDB

Features:
- live_analysis collection management
- Trade setup storage with metadata
- Symbol/timeframe based queries
- Historical analysis retrieval
- Automatic document formatting

Usage:
    from scripts.chroma_live_analyzer import ChromaLiveAnalyzer

    analyzer = ChromaLiveAnalyzer()
    analyzer.store_live_analysis(analysis_data)
    results = analyzer.get_latest_analysis("XAUUSD", "M15")
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid


class ChromaLiveAnalyzer:
    """ChromaDB integration for live trading analysis"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB live analyzer

        Args:
            persist_directory: Directory to persist ChromaDB
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "live_analysis"
        self._initialized = False
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create live_analysis collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Live trading analysis and trade recommendations",
                    "version": "1.0"
                }
            )

            self._initialized = True
            self.logger.info(f"✅ ChromaDB '{self.collection_name}' collection initialized")

        except ImportError:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise

    def format_analysis_document(self, analysis_data: Dict[str, Any]) -> str:
        """
        Format analysis data for ChromaDB embedding

        Args:
            analysis_data: Complete analysis from live_trading_analyzer

        Returns:
            Formatted document string for embedding
        """
        try:
            symbol = analysis_data.get('symbol', 'UNKNOWN')
            timeframe = analysis_data.get('timeframe', 'UNKNOWN')
            timestamp = analysis_data.get('timestamp', datetime.now().isoformat())
            current_price = analysis_data.get('current_price', 0)

            pattern = analysis_data.get('pattern', {})
            pattern_name = pattern.get('name', 'unknown')
            pattern_confidence = pattern.get('confidence', 0)

            context = analysis_data.get('context', {})
            trend = context.get('trend', 'unknown')
            rsi_state = context.get('rsi_state', 'unknown')
            volume_state = context.get('volume_state', 'unknown')
            session = context.get('session', 'unknown')

            recommendation = analysis_data.get('recommendation', {})
            direction = recommendation.get('direction', 'HOLD')
            confidence = recommendation.get('confidence', 0)
            entry_price = recommendation.get('entry_price', 0)
            stop_loss = recommendation.get('stop_loss', 0)
            take_profit = recommendation.get('take_profit', 0)
            risk_reward = recommendation.get('risk_reward_ratio', 0)

            # Create comprehensive document for embedding
            document = f"""
Live Analysis - {symbol} {timeframe}
Timestamp: {timestamp}
Current Price: {current_price:.2f}

Pattern Analysis:
- Pattern: {pattern_name.replace('_', ' ').title()}
- Confidence: {pattern_confidence}%
- Direction: {pattern.get('direction', 'unknown')}
- Type: {pattern.get('type', 'unknown')}

Market Context:
- Trend: {trend}
- RSI State: {rsi_state}
- Volume: {volume_state}
- Session: {session}

Technical Indicators:
- RSI: {analysis_data.get('indicators', {}).get('momentum', {}).get('rsi', 0):.0f}
- MACD: {analysis_data.get('indicators', {}).get('momentum', {}).get('macd', 0):.2f}
- EMA 9/20/50: {analysis_data.get('indicators', {}).get('trend', {}).get('ema_9', 0):.2f}/{analysis_data.get('indicators', {}).get('trend', {}).get('ema_20', 0):.2f}/{analysis_data.get('indicators', {}).get('trend', {}).get('ema_50', 0):.2f}
- Volume Ratio: {analysis_data.get('indicators', {}).get('volume', {}).get('volume_ratio', 0):.1f}x
- ATR: {analysis_data.get('indicators', {}).get('volatility', {}).get('atr', 0):.2f}

Trade Recommendation:
- Direction: {direction}
- Confidence: {confidence}%
- Entry: {entry_price:.2f}
- Stop Loss: {stop_loss:.2f}
- Take Profit: {take_profit:.2f}
- Risk/Reward: {risk_reward}:1

Support/Resistance Levels:
- Support: {context.get('levels', {}).get('support', [])[:3]}
- Resistance: {context.get('levels', {}).get('resistance', [])[:3]}

Reasoning: {recommendation.get('reasoning', 'No reasoning provided')}

Summary: {analysis_data.get('summary', 'No summary available')}
            """.strip()

            return document

        except Exception as e:
            self.logger.error(f"❌ Error formatting analysis document: {e}")
            return f"Live Analysis - {symbol} {timeframe} - Error formatting document"

    def create_metadata(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for ChromaDB document

        Args:
            analysis_data: Complete analysis data

        Returns:
            Metadata dictionary for filtering
        """
        try:
            pattern = analysis_data.get('pattern', {})
            context = analysis_data.get('context', {})
            recommendation = analysis_data.get('recommendation', {})

            metadata = {
                # Core identifiers
                "symbol": analysis_data.get('symbol', 'UNKNOWN'),
                "timeframe": analysis_data.get('timeframe', 'UNKNOWN'),
                "timestamp": analysis_data.get('timestamp', datetime.now().isoformat()),
                "analysis_type": "live",

                # Pattern information
                "pattern_name": pattern.get('name', 'unknown'),
                "pattern_direction": pattern.get('direction', 'unknown'),
                "pattern_type": pattern.get('type', 'unknown'),
                "pattern_confidence": pattern.get('confidence', 0),
                "pattern_quality": pattern.get('quality', 0),

                # Trade recommendation
                "trade_direction": recommendation.get('direction', 'HOLD'),
                "entry_price": float(recommendation.get('entry_price', 0)),
                "current_price": float(analysis_data.get('current_price', 0)),
                "confidence": int(recommendation.get('confidence', 0)),
                "risk_reward_ratio": float(recommendation.get('risk_reward_ratio', 0)),

                # Market context
                "trend": context.get('trend', 'unknown'),
                "rsi_state": context.get('rsi_state', 'unknown'),
                "volume_state": context.get('volume_state', 'unknown'),
                "session": context.get('session', 'unknown'),

                # Technical indicators (for numeric filtering)
                "rsi": float(analysis_data.get('indicators', {}).get('momentum', {}).get('rsi', 50)),
                "macd": float(analysis_data.get('indicators', {}).get('momentum', {}).get('macd', 0)),
                "volume_ratio": float(analysis_data.get('indicators', {}).get('volume', {}).get('volume_ratio', 1.0)),
                "atr": float(analysis_data.get('indicators', {}).get('volatility', {}).get('atr', 0)),

                # Storage info
                "created_at": datetime.now().isoformat(),
                "document_id": str(uuid.uuid4())
            }

            return metadata

        except Exception as e:
            self.logger.error(f"❌ Error creating metadata: {e}")
            return {
                "symbol": "ERROR",
                "analysis_type": "live",
                "error": str(e),
                "created_at": datetime.now().isoformat()
            }

    def store_live_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Store live analysis in ChromaDB

        Args:
            analysis_data: Complete analysis from live_trading_analyzer

        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_initialized()

            # Format document and create metadata
            document = self.format_analysis_document(analysis_data)
            metadata = self.create_metadata(analysis_data)

            # Generate unique ID
            doc_id = f"{metadata['symbol']}_{metadata['timeframe']}_{metadata['timestamp']}"

            # Store in ChromaDB
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )

            self.logger.info(f"✅ Stored live analysis: {metadata['symbol']} {metadata['timeframe']} (confidence: {metadata['confidence']}%)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store live analysis: {e}")
            return False

    def get_latest_analysis(self, symbol: str, timeframe: str, min_confidence: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get latest analysis for symbol/timeframe

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            min_confidence: Minimum confidence threshold

        Returns:
            Latest analysis data or None if not found
        """
        try:
            self._ensure_initialized()

            # Query for latest analysis
            results = self.collection.query(
                query_texts=[f"Latest analysis for {symbol} {timeframe}"],
                where={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "analysis_type": "live",
                    "confidence": {"$gte": min_confidence}
                },
                n_results=1
            )

            if not results['ids'][0]:
                self.logger.info(f"📭 No analysis found for {symbol} {timeframe}")
                return None

            # Get the result
            doc_id = results['ids'][0][0]
            document = results['documents'][0][0]
            metadata = results['metadatas'][0][0]

            self.logger.info(f"📄 Found analysis for {symbol} {timeframe}: {metadata['confidence']}% confidence")

            return {
                "document_id": doc_id,
                "document": document,
                "metadata": metadata
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get latest analysis: {e}")
            return None

    def get_recent_analyses(self, symbol: str, timeframe: str = None, hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent analyses for a symbol

        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe filter
            hours: Hours to look back
            limit: Maximum results

        Returns:
            List of recent analyses
        """
        try:
            self._ensure_initialized()

            # Calculate timestamp threshold
            threshold_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            # Build where clause
            where_clause = {
                "symbol": symbol,
                "analysis_type": "live",
                "timestamp": {"$gte": threshold_time}
            }

            if timeframe:
                where_clause["timeframe"] = timeframe

            # Query for recent analyses
            results = self.collection.query(
                query_texts=[f"Recent analyses for {symbol}"],
                where=where_clause,
                n_results=limit
            )

            if not results['ids'][0]:
                return []

            # Format results
            analyses = []
            for i in range(len(results['ids'][0])):
                analyses.append({
                    "document_id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })

            # Sort by timestamp (most recent first)
            analyses.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)

            self.logger.info(f"📄 Found {len(analyses)} recent analyses for {symbol}")
            return analyses

        except Exception as e:
            self.logger.error(f"❌ Failed to get recent analyses: {e}")
            return []

    def get_high_confidence_setups(self, symbol: str = None, min_confidence: int = 70, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get high confidence trade setups

        Args:
            symbol: Optional symbol filter
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of high confidence setups
        """
        try:
            self._ensure_initialized()

            # Build where clause
            where_clause = {
                "analysis_type": "live",
                "confidence": {"$gte": min_confidence},
                "trade_direction": {"$ne": "HOLD"}
            }

            if symbol:
                where_clause["symbol"] = symbol

            # Query for high confidence setups
            results = self.collection.query(
                query_texts=["High confidence trade setups"],
                where=where_clause,
                n_results=limit
            )

            if not results['ids'][0]:
                return []

            # Format results
            setups = []
            for i in range(len(results['ids'][0])):
                setups.append({
                    "document_id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })

            # Sort by confidence (highest first)
            setups.sort(key=lambda x: x['metadata']['confidence'], reverse=True)

            self.logger.info(f"🎯 Found {len(setups)} high confidence setups (≥{min_confidence}%)")
            return setups

        except Exception as e:
            self.logger.error(f"❌ Failed to get high confidence setups: {e}")
            return []

    def cleanup_old_analyses(self, symbol: str = None, days: int = 7) -> int:
        """
        Clean up old analyses to manage storage

        Args:
            symbol: Optional symbol filter
            days: Days to keep data

        Returns:
            Number of analyses removed
        """
        try:
            self._ensure_initialized()

            # Calculate cutoff time
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

            # Build where clause
            where_clause = {
                "analysis_type": "live",
                "timestamp": {"$lt": cutoff_time}
            }

            if symbol:
                where_clause["symbol"] = symbol

            # Get old analyses
            results = self.collection.get(
                where=where_clause
            )

            if not results['ids']:
                return 0

            # Delete old analyses
            self.collection.delete(ids=results['ids'])

            count = len(results['ids'])
            self.logger.info(f"🗑️  Cleaned up {count} old analyses (older than {days} days)")
            return count

        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old analyses: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics

        Returns:
            Statistics about the collection
        """
        try:
            self._ensure_initialized()

            # Get total count
            total_count = self.collection.count()

            # Get unique symbols
            all_results = self.collection.get()
            symbols = set()
            timeframes = set()
            patterns = set()

            for metadata in all_results.get('metadatas', []):
                if metadata:
                    symbols.add(metadata.get('symbol', 'unknown'))
                    timeframes.add(metadata.get('timeframe', 'unknown'))
                    patterns.add(metadata.get('pattern_name', 'unknown'))

            return {
                "collection_name": self.collection_name,
                "total_analyses": total_count,
                "unique_symbols": sorted(list(symbols)),
                "unique_timeframes": sorted(list(timeframes)),
                "unique_patterns": sorted(list(patterns)),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}


def main():
    """Example usage of ChromaLiveAnalyzer"""
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create example analysis data
    example_analysis = {
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "timestamp": datetime.now().isoformat(),
        "current_price": 1985.50,
        "pattern": {
            "name": "bullish_engulfing",
            "type": "reversal",
            "direction": "bullish",
            "quality": 0.85,
            "confidence": 80
        },
        "context": {
            "trend": "bullish",
            "rsi_state": "oversold",
            "volume_state": "high",
            "session": "london",
            "levels": {
                "support": [1980.0, 1975.0, 1970.0],
                "resistance": [1990.0, 1995.0, 2000.0]
            }
        },
        "indicators": {
            "momentum": {"rsi": 32.5, "macd": 12.3},
            "trend": {"ema_9": 1982.0, "ema_20": 1978.0, "ema_50": 1965.0},
            "volume": {"volume_ratio": 1.8},
            "volatility": {"atr": 15.2}
        },
        "recommendation": {
            "direction": "LONG",
            "confidence": 85,
            "entry_price": 1986.0,
            "stop_loss": 1978.0,
            "take_profit": 2000.0,
            "risk_reward_ratio": 2.0,
            "reasoning": "Strong bullish pattern with oversold RSI and high volume confirmation"
        },
        "summary": "XAUUSD M15 - Bullish engulfing pattern detected with 85% confidence - LONG setup"
    }

    # Test ChromaDB integration
    analyzer = ChromaLiveAnalyzer()

    print("🧪 Testing ChromaDB Live Analysis Storage")
    print("=" * 50)

    # Store analysis
    success = analyzer.store_live_analysis(example_analysis)
    print(f"Store analysis: {'✅ Success' if success else '❌ Failed'}")

    # Get latest analysis
    latest = analyzer.get_latest_analysis("XAUUSD", "M15")
    print(f"Get latest analysis: {'✅ Found' if latest else '❌ Not found'}")

    # Get collection stats
    stats = analyzer.get_collection_stats()
    print(f"Collection stats: {stats}")

    # Get high confidence setups
    setups = analyzer.get_high_confidence_setups(min_confidence=70)
    print(f"High confidence setups: {len(setups)} found")

    print("\n✅ ChromaDB Live Analysis Storage test complete")


if __name__ == "__main__":
    main()