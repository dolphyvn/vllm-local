#!/usr/bin/env python3
"""
Pattern Retriever - Query helper for LLM integration
Optimized for small LLM + ChromaDB RAG system

Retrieves and formats patterns from ChromaDB for LLM consumption:
- Semantic search with metadata filtering
- Statistical aggregation (win rate, avg P&L)
- Structured output optimized for small LLMs
- Context-aware pattern matching

Usage:
    # Query patterns
    python scripts/pattern_retriever.py \\
        --query "bullish reversal oversold RSI" \\
        --limit 5

    # Query with filters
    python scripts/pattern_retriever.py \\
        --query "support bounce" \\
        --symbol XAUUSD --timeframe M15 \\
        --outcome WIN --limit 10

    # As library
    from pattern_retriever import PatternRetriever
    retriever = PatternRetriever()
    results = retriever.search(query="breakout", filters={"symbol": "XAUUSD"})
"""

import json
import argparse
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime


class PatternRetriever:
    """Query and retrieve patterns from ChromaDB for LLM"""

    def __init__(self, chroma_dir: str = "./chroma_db", collection_name: str = "trading_patterns"):
        """
        Initialize pattern retriever

        Args:
            chroma_dir: ChromaDB persistence directory
            collection_name: Name of the collection
        """
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        self._initialize_chromadb()

    def _initialize_chromadb(self) -> bool:
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=self.chroma_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            return True

        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}", file=sys.stderr)
            return False

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar patterns

        Args:
            query: Natural language query
            filters: Optional metadata filters (e.g., {"symbol": "XAUUSD", "outcome_result": "WIN"})
            limit: Maximum number of results

        Returns:
            Dictionary with patterns and statistics
        """
        if not self.collection:
            return {"error": "ChromaDB not initialized"}

        try:
            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": limit
            }

            if filters:
                # ChromaDB requires multiple filters to be wrapped in $and
                if len(filters) > 1:
                    query_params["where"] = {
                        "$and": [{k: v} for k, v in filters.items()]
                    }
                else:
                    query_params["where"] = filters

            # Query ChromaDB
            results = self.collection.query(**query_params)

            if not results['documents'] or not results['documents'][0]:
                return {
                    "query": query,
                    "filters": filters,
                    "patterns": [],
                    "statistics": {},
                    "message": "No patterns found"
                }

            # Parse results
            patterns = []
            for i, doc_str in enumerate(results['documents'][0]):
                try:
                    # Parse JSON document
                    doc = json.loads(doc_str)
                    pattern_data = doc.get('data', {})

                    # Add distance/similarity score
                    if results.get('distances') and results['distances'][0]:
                        pattern_data['similarity_score'] = 1.0 - results['distances'][0][i]

                    patterns.append(pattern_data)

                except json.JSONDecodeError as e:
                    print(f"⚠️  Failed to parse pattern: {e}", file=sys.stderr)
                    continue

            # Calculate statistics
            statistics = self._calculate_statistics(patterns)

            return {
                "query": query,
                "filters": filters,
                "total_found": len(patterns),
                "patterns": patterns,
                "statistics": statistics,
                "retrieved_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "filters": filters
            }

    def search_by_current_market(
        self,
        current_indicators: Dict[str, Any],
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for patterns similar to current market conditions

        Args:
            current_indicators: Current market indicators
                {
                    "symbol": "XAUUSD",
                    "rsi": 45,
                    "trend": "bullish",
                    "volume_ratio": 1.5,
                    "session": "london"
                }
            limit: Maximum number of results

        Returns:
            Dictionary with relevant patterns and analysis
        """
        # Build query from indicators
        query_parts = []
        filters = {}

        # Symbol and timeframe (exact match)
        if 'symbol' in current_indicators:
            filters['symbol'] = current_indicators['symbol']

        if 'timeframe' in current_indicators:
            filters['timeframe'] = current_indicators['timeframe']

        # RSI state (bucketed)
        if 'rsi' in current_indicators:
            rsi = current_indicators['rsi']
            if rsi <= 30:
                filters['rsi_bucket'] = "oversold"
                query_parts.append("oversold RSI reversal")
            elif rsi <= 40:
                filters['rsi_bucket'] = "bearish"
                query_parts.append("bearish RSI recovery")
            elif rsi <= 60:
                filters['rsi_bucket'] = "neutral"
                query_parts.append("neutral RSI continuation")
            elif rsi <= 70:
                filters['rsi_bucket'] = "bullish"
                query_parts.append("bullish RSI momentum")
            else:
                filters['rsi_bucket'] = "overbought"
                query_parts.append("overbought RSI reversal")

        # Trend
        if 'trend' in current_indicators:
            trend = current_indicators['trend']
            query_parts.append(f"{trend} trend")
            filters['trend'] = trend

        # Volume
        if 'volume_ratio' in current_indicators:
            vol_ratio = current_indicators['volume_ratio']
            if vol_ratio < 0.8:
                filters['volume_bucket'] = "low"
                query_parts.append("low volume")
            elif vol_ratio < 1.5:
                filters['volume_bucket'] = "normal"
            elif vol_ratio < 2.0:
                filters['volume_bucket'] = "high"
                query_parts.append("high volume")
            else:
                filters['volume_bucket'] = "very_high"
                query_parts.append("very high volume")

        # Session
        if 'session' in current_indicators:
            filters['session'] = current_indicators['session']

        # Build query string
        query = " ".join(query_parts) if query_parts else "pattern"

        # Search
        return self.search(query=query, filters=filters, limit=limit)

    def _calculate_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from patterns"""
        if not patterns:
            return {}

        total = len(patterns)

        # Outcomes
        wins = sum(1 for p in patterns if p.get('outcome', {}).get('result') == 'WIN')
        losses = sum(1 for p in patterns if p.get('outcome', {}).get('result') == 'LOSS')
        neutrals = total - wins - losses

        # P&L
        pnl_points = [p.get('outcome', {}).get('pnl_points', 0) for p in patterns]
        avg_pnl = sum(pnl_points) / total if total > 0 else 0
        total_pnl = sum(pnl_points)

        winning_pnl = [p for p in pnl_points if p > 0]
        losing_pnl = [p for p in pnl_points if p < 0]

        avg_win = sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0
        avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0

        # Duration
        durations = [p.get('outcome', {}).get('duration_bars', 0) for p in patterns]
        avg_duration = sum(durations) / total if total > 0 else 0

        # Quality scores
        quality_scores = [p.get('quality_score', 0) for p in patterns]
        avg_quality = sum(quality_scores) / total if total > 0 else 0

        # Pattern distribution
        pattern_types = {}
        for p in patterns:
            ptype = p.get('pattern', {}).get('name', 'Unknown')
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

        # Session distribution
        sessions = {}
        for p in patterns:
            session = p.get('context', {}).get('session', 'unknown')
            sessions[session] = sessions.get(session, 0) + 1

        # Most common session
        most_common_session = max(sessions.items(), key=lambda x: x[1])[0] if sessions else None

        return {
            "total_patterns": total,
            "outcomes": {
                "wins": wins,
                "losses": losses,
                "neutrals": neutrals,
                "win_rate_pct": (wins / total * 100) if total > 0 else 0
            },
            "pnl": {
                "avg_pnl_points": round(avg_pnl, 2),
                "total_pnl_points": round(total_pnl, 2),
                "avg_win_points": round(avg_win, 2),
                "avg_loss_points": round(avg_loss, 2),
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0
            },
            "duration": {
                "avg_duration_bars": round(avg_duration, 1)
            },
            "quality": {
                "avg_quality_score": round(avg_quality, 2)
            },
            "distribution": {
                "by_pattern_type": pattern_types,
                "by_session": sessions,
                "most_common_session": most_common_session
            }
        }

    def format_for_llm(self, search_results: Dict[str, Any], max_patterns: int = 3) -> str:
        """
        Format search results for LLM consumption

        Creates a structured summary optimized for small LLMs

        Args:
            search_results: Results from search()
            max_patterns: Maximum number of detailed patterns to include

        Returns:
            Formatted string for LLM context
        """
        if 'error' in search_results:
            return f"Error retrieving patterns: {search_results['error']}"

        if not search_results.get('patterns'):
            return "No similar patterns found in historical data."

        stats = search_results.get('statistics', {})
        patterns = search_results['patterns'][:max_patterns]

        # Build formatted output
        output = []

        # Header
        output.append("=== HISTORICAL PATTERN ANALYSIS ===")
        output.append(f"Query: {search_results.get('query', 'N/A')}")
        if search_results.get('filters'):
            output.append(f"Filters: {json.dumps(search_results['filters'])}")
        output.append("")

        # Statistics
        output.append("📊 AGGREGATE STATISTICS")
        output.append(f"Total similar patterns: {stats.get('total_patterns', 0)}")

        if 'outcomes' in stats:
            outcomes = stats['outcomes']
            output.append(f"Win rate: {outcomes.get('win_rate_pct', 0):.1f}% ({outcomes.get('wins', 0)}W / {outcomes.get('losses', 0)}L / {outcomes.get('neutrals', 0)}N)")

        if 'pnl' in stats:
            pnl = stats['pnl']
            output.append(f"Average P&L: {pnl.get('avg_pnl_points', 0):+.2f} points")
            output.append(f"Avg Win: {pnl.get('avg_win_points', 0):+.2f} points | Avg Loss: {pnl.get('avg_loss_points', 0):+.2f} points")
            output.append(f"Profit Factor: {pnl.get('profit_factor', 0):.2f}")

        if 'distribution' in stats:
            dist = stats['distribution']
            if dist.get('most_common_session'):
                output.append(f"Most common: {dist['most_common_session']} session")

        output.append("")

        # Detailed patterns
        output.append(f"📈 TOP {len(patterns)} SIMILAR PATTERNS")
        output.append("")

        for i, pattern in enumerate(patterns, 1):
            output.append(f"Pattern {i}: {pattern.get('pattern', {}).get('name', 'Unknown')}")
            output.append(f"  Symbol: {pattern.get('symbol')} {pattern.get('timeframe')}")
            output.append(f"  Date: {pattern.get('timestamp', 'Unknown')[:16]}")

            # Pattern details
            pattern_info = pattern.get('pattern', {})
            output.append(f"  Direction: {pattern_info.get('direction', 'N/A')}")
            output.append(f"  Quality: {pattern_info.get('quality', 0):.2f}")

            # Indicators
            indicators = pattern.get('indicators', {})
            if 'momentum' in indicators:
                rsi = indicators['momentum'].get('rsi', 0)
                output.append(f"  RSI: {rsi:.0f}")

            # Context
            context = pattern.get('context', {})
            output.append(f"  Trend: {context.get('trend', 'N/A')}")
            output.append(f"  Volume: {indicators.get('volume', {}).get('volume_ratio', 0):.1f}x")
            output.append(f"  Session: {context.get('session', 'N/A')}")

            # Outcome
            outcome = pattern.get('outcome', {})
            result = outcome.get('result', 'UNKNOWN')
            pnl_pct = outcome.get('pnl_pct', 0)
            duration = outcome.get('duration_bars', 0)

            output.append(f"  ➜ Outcome: {result} ({pnl_pct:+.2f}% in {duration} bars)")

            if i < len(patterns):
                output.append("")

        output.append("")
        output.append("=== END HISTORICAL ANALYSIS ===")

        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Query patterns from ChromaDB for LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for patterns
  python scripts/pattern_retriever.py \\
      --query "bullish reversal oversold RSI"

  # Search with filters
  python scripts/pattern_retriever.py \\
      --query "support bounce" \\
      --symbol XAUUSD --timeframe M15 --outcome WIN

  # Format for LLM
  python scripts/pattern_retriever.py \\
      --query "breakout high volume" \\
      --format llm

  # Search by current market conditions (JSON input)
  echo '{"symbol":"XAUUSD","rsi":32,"trend":"bullish","volume_ratio":1.8}' | \\
      python scripts/pattern_retriever.py --current-market
        """
    )

    parser.add_argument('--query', help='Natural language query')
    parser.add_argument('--current-market', action='store_true', help='Read current market conditions from stdin (JSON)')
    parser.add_argument('--symbol', help='Filter by symbol (e.g., XAUUSD)')
    parser.add_argument('--timeframe', help='Filter by timeframe (e.g., M15)')
    parser.add_argument('--outcome', help='Filter by outcome (WIN, LOSS, NEUTRAL)')
    parser.add_argument('--session', help='Filter by session (asia, london, newyork)')
    parser.add_argument('--limit', type=int, default=5, help='Maximum results (default: 5)')
    parser.add_argument('--format', choices=['json', 'llm'], default='json', help='Output format (default: json)')
    parser.add_argument('--chroma-dir', default='./chroma_db', help='ChromaDB directory')
    parser.add_argument('--collection', default='trading_patterns', help='Collection name')

    args = parser.parse_args()

    # Initialize retriever
    retriever = PatternRetriever(args.chroma_dir, args.collection)

    if not retriever.collection:
        print("❌ Failed to initialize ChromaDB")
        return 1

    # Build filters
    filters = {}
    if args.symbol:
        filters['symbol'] = args.symbol
    if args.timeframe:
        filters['timeframe'] = args.timeframe
    if args.outcome:
        filters['outcome_result'] = args.outcome.upper()
    if args.session:
        filters['session'] = args.session.lower()

    # Search
    if args.current_market:
        # Read JSON from stdin
        try:
            import json
            current_indicators = json.load(sys.stdin)
            results = retriever.search_by_current_market(current_indicators, limit=args.limit)
        except Exception as e:
            print(f"❌ Failed to parse current market JSON: {e}")
            return 1
    else:
        if not args.query:
            print("❌ Either --query or --current-market is required")
            return 1

        results = retriever.search(args.query, filters=filters if filters else None, limit=args.limit)

    # Output
    if args.format == 'llm':
        output = retriever.format_for_llm(results)
        print(output)
    else:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
