#!/usr/bin/env python3
"""
RAG Structured Feeder - Feed patterns to ChromaDB
Optimized for small LLM + ChromaDB RAG system

Feeds detected patterns to ChromaDB with:
- Structured JSON storage (summary for embedding, full data for retrieval)
- Flattened metadata for efficient filtering
- Batch processing support
- Duplicate detection

Usage:
    python scripts/rag_structured_feeder.py \
        --input data/patterns/XAUUSD_M15_patterns.json \
        --chroma-dir ./chroma_db \
        --collection trading_patterns
"""

import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class RAGStructuredFeeder:
    """Feed structured patterns to ChromaDB for RAG"""

    def __init__(self, chroma_dir: str = "./chroma_db", collection_name: str = "trading_patterns"):
        """
        Initialize ChromaDB feeder

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
        print(f"🔌 Connecting to ChromaDB...")
        print(f"   Directory: {self.chroma_dir}")
        print(f"   Collection: {self.collection_name}")

        try:
            import chromadb
            from chromadb.config import Settings

            # Create client
            self.client = chromadb.PersistentClient(path=self.chroma_dir)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Trading patterns with structured data for LLM RAG",
                    "format_version": "2.0_structured",
                    "created_at": datetime.now().isoformat()
                }
            )

            count = self.collection.count()
            print(f"✅ Connected! Current entries: {count}")
            return True

        except ImportError:
            print(f"❌ ChromaDB not installed. Install: pip install chromadb")
            return False
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_patterns(self, json_path: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load patterns from JSON file"""
        print(f"\n📂 Loading patterns: {json_path}")

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            patterns = data.get('patterns', [])

            print(f"✅ Loaded {len(patterns)} patterns")
            print(f"   Symbol: {metadata.get('source', {}).get('symbol')}")
            print(f"   Timeframe: {metadata.get('source', {}).get('timeframe')}")
            print(f"   Win rate: {metadata.get('statistics', {}).get('win_rate', 0):.1f}%")

            return metadata, patterns

        except Exception as e:
            print(f"❌ Load failed: {e}")
            return {}, []

    def _flatten_metadata(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten pattern metadata for ChromaDB filtering

        ChromaDB only supports: str, int, float, bool in metadata
        """
        metadata = {
            # Basic info
            "pattern_id": pattern['pattern_id'],
            "symbol": pattern['symbol'],
            "timeframe": pattern['timeframe'],
            "timestamp": pattern['timestamp'],

            # Pattern info
            "pattern_name": pattern['pattern']['name'],
            "pattern_type": pattern['pattern']['type'],
            "pattern_direction": pattern['pattern']['direction'],
            "pattern_quality": float(pattern['pattern']['quality']),

            # Entry
            "entry_price": float(pattern['entry']['price']),
            "candle_index": int(pattern['entry']['candle_index']),

            # Context
            "trend": pattern['context']['trend'],
            "rsi_state": pattern['context']['rsi_state'],
            "volume_state": pattern['context']['volume_state'],
            "session": pattern['context']['session'],
            "day_of_week": pattern['context']['day_of_week'],
            "hour": int(pattern['context']['hour']),

            # Indicators (key ones for filtering)
            "rsi": float(pattern['indicators']['momentum']['rsi']),
            "volume_ratio": float(pattern['indicators']['volume']['volume_ratio']),

            # Outcome
            "outcome_result": pattern['outcome']['result'],
            "outcome_pnl_points": float(pattern['outcome']['pnl_points']),
            "outcome_pnl_pct": float(pattern['outcome']['pnl_pct']),
            "outcome_duration_bars": int(pattern['outcome']['duration_bars']),

            # Quality
            "quality_score": float(pattern['quality_score']),

            # Metadata
            "detected_at": pattern['metadata']['detected_at']
        }

        # Add RSI buckets for easier filtering
        rsi = metadata['rsi']
        if rsi <= 30:
            metadata['rsi_bucket'] = "oversold"
        elif rsi <= 40:
            metadata['rsi_bucket'] = "bearish"
        elif rsi <= 60:
            metadata['rsi_bucket'] = "neutral"
        elif rsi <= 70:
            metadata['rsi_bucket'] = "bullish"
        else:
            metadata['rsi_bucket'] = "overbought"

        # Add volume buckets
        vol_ratio = metadata['volume_ratio']
        if vol_ratio < 0.8:
            metadata['volume_bucket'] = "low"
        elif vol_ratio < 1.5:
            metadata['volume_bucket'] = "normal"
        elif vol_ratio < 2.0:
            metadata['volume_bucket'] = "high"
        else:
            metadata['volume_bucket'] = "very_high"

        return metadata

    def _prepare_document(self, pattern: Dict[str, Any]) -> str:
        """
        Prepare document for ChromaDB

        Format: JSON string with summary for embedding + full data for retrieval
        """
        document = {
            "summary": pattern['summary'],  # This gets embedded
            "data": pattern  # Full structured data for retrieval
        }

        return json.dumps(document, ensure_ascii=False)

    def feed_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Feed a single pattern to ChromaDB"""
        try:
            # Check if already exists
            existing = self.collection.get(ids=[pattern['pattern_id']])
            if existing['ids']:
                print(f"   ⏭️  Skipping {pattern['pattern_id']} (already exists)")
                return False

            # Prepare data
            metadata = self._flatten_metadata(pattern)
            document = self._prepare_document(pattern)

            # Add to collection
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[pattern['pattern_id']]
            )

            return True

        except Exception as e:
            print(f"   ❌ Failed to feed {pattern.get('pattern_id', 'unknown')}: {e}")
            return False

    def feed_patterns_batch(self, patterns: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """Feed patterns in batches"""
        print(f"\n📤 Feeding patterns (batch size: {batch_size})...")

        stats = {
            'total': len(patterns),
            'added': 0,
            'skipped': 0,
            'failed': 0
        }

        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(patterns))})...")

            batch_ids = []
            batch_documents = []
            batch_metadatas = []

            for pattern in batch:
                try:
                    # Check if exists
                    pattern_id = pattern['pattern_id']
                    existing = self.collection.get(ids=[pattern_id])
                    if existing['ids']:
                        stats['skipped'] += 1
                        continue

                    # Prepare data
                    metadata = self._flatten_metadata(pattern)
                    document = self._prepare_document(pattern)

                    batch_ids.append(pattern_id)
                    batch_documents.append(document)
                    batch_metadatas.append(metadata)

                except Exception as e:
                    print(f"      ⚠️  Error preparing {pattern.get('pattern_id', 'unknown')}: {e}")
                    stats['failed'] += 1

            # Add batch to ChromaDB
            if batch_ids:
                try:
                    self.collection.add(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    stats['added'] += len(batch_ids)
                    print(f"      ✅ Added {len(batch_ids)} patterns")
                except Exception as e:
                    print(f"      ❌ Batch insert failed: {e}")
                    stats['failed'] += len(batch_ids)

        return stats

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()

            # Get sample to understand data distribution
            sample = self.collection.get(limit=1000)

            if sample['metadatas']:
                # Count by outcome
                outcomes = {}
                patterns_by_type = {}
                symbols = set()
                timeframes = set()

                for meta in sample['metadatas']:
                    # Outcomes
                    outcome = meta.get('outcome_result', 'UNKNOWN')
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1

                    # Pattern types
                    pattern_name = meta.get('pattern_name', 'Unknown')
                    patterns_by_type[pattern_name] = patterns_by_type.get(pattern_name, 0) + 1

                    # Symbols and timeframes
                    symbols.add(meta.get('symbol', 'Unknown'))
                    timeframes.add(meta.get('timeframe', 'Unknown'))

                return {
                    'total_entries': count,
                    'outcomes': outcomes,
                    'patterns_by_type': patterns_by_type,
                    'symbols': list(symbols),
                    'timeframes': list(timeframes),
                    'sample_size': len(sample['metadatas'])
                }
            else:
                return {
                    'total_entries': count,
                    'note': 'Collection is empty'
                }

        except Exception as e:
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Feed structured patterns to ChromaDB for RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feed patterns to ChromaDB
  python scripts/rag_structured_feeder.py \\
      --input data/patterns/XAUUSD_M15_patterns.json \\
      --chroma-dir ./chroma_db \\
      --collection trading_patterns

  # Feed with smaller batch size
  python scripts/rag_structured_feeder.py \\
      --input data/patterns/XAUUSD_M15_patterns.json \\
      --batch-size 50

  # Show stats only (no feeding)
  python scripts/rag_structured_feeder.py \\
      --stats-only
        """
    )

    parser.add_argument('--input', help='Input patterns JSON file')
    parser.add_argument('--chroma-dir', default='./chroma_db', help='ChromaDB directory (default: ./chroma_db)')
    parser.add_argument('--collection', default='trading_patterns', help='Collection name (default: trading_patterns)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for feeding (default: 100)')
    parser.add_argument('--stats-only', action='store_true', help='Show stats only, do not feed')

    args = parser.parse_args()

    print("=" * 70)
    print("RAG STRUCTURED FEEDER")
    print("=" * 70)
    print(f"ChromaDB:    {args.chroma_dir}")
    print(f"Collection:  {args.collection}")
    print("=" * 70)

    # Initialize feeder
    feeder = RAGStructuredFeeder(args.chroma_dir, args.collection)

    if not feeder.client or not feeder.collection:
        print("❌ Failed to initialize ChromaDB")
        return 1

    # Show stats if requested
    if args.stats_only or not args.input:
        print("\n📊 Collection Statistics:")
        print("-" * 70)
        stats = feeder.get_collection_stats()
        print(json.dumps(stats, indent=2))
        return 0

    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Load patterns
    metadata, patterns = feeder.load_patterns(args.input)

    if not patterns:
        print("❌ No patterns to feed")
        return 1

    # Feed patterns
    stats = feeder.feed_patterns_batch(patterns, batch_size=args.batch_size)

    print("\n" + "=" * 70)
    print("✅ FEEDING COMPLETE")
    print("=" * 70)
    print(f"📊 Results:")
    print(f"   Total:   {stats['total']}")
    print(f"   Added:   {stats['added']}")
    print(f"   Skipped: {stats['skipped']} (already exists)")
    print(f"   Failed:  {stats['failed']}")

    if stats['added'] > 0:
        print(f"\n💡 Next step:")
        print(f"   python scripts/pattern_retriever.py --query \"bullish reversal oversold\"")

    # Show updated stats
    print("\n📊 Updated Collection Statistics:")
    print("-" * 70)
    collection_stats = feeder.get_collection_stats()
    print(f"Total patterns: {collection_stats.get('total_entries', 0)}")
    if 'outcomes' in collection_stats:
        print(f"Outcomes: {collection_stats['outcomes']}")
    if 'patterns_by_type' in collection_stats:
        top_patterns = sorted(collection_stats['patterns_by_type'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top patterns: {dict(top_patterns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
