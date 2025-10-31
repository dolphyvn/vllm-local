#!/usr/bin/env python3
"""
Direct integration with RAG system (no API needed)
Feeds processed trading patterns directly to ChromaDB via MemoryManager
This is much faster for bulk operations
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memory import MemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("❌ Error: memory module not available")
    sys.exit(1)

class DirectTradingPatternFeeder:
    """Feed trading patterns directly to ChromaDB"""

    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize direct feeder

        Args:
            persist_directory: Path to ChromaDB directory
        """
        self.memory_manager = MemoryManager(
            collection_name="financial_memory",
            persist_directory=persist_directory
        )
        print(f"✅ Connected to ChromaDB at {persist_directory}")

    def feed_pattern(self, pattern):
        """Feed a single pattern directly to memory"""
        try:
            # Extract metadata
            metadata = pattern['metadata']

            # Create lesson title
            lesson_title = f"Trading Pattern: {metadata['pattern']} - {metadata['symbol']} {metadata['timeframe']}"

            # Use the pattern text as content (already comprehensive)
            lesson_content = pattern['text']

            # Determine tags
            tags = [
                metadata['pattern'],
                metadata['symbol'],
                metadata['timeframe'],
                metadata['outcome'],
                metadata['trend'],
                metadata.get('session', ''),
                metadata.get('day_of_week', ''),
                metadata.get('market_regime', ''),
                'pattern_detection',
                'historical_analysis'
            ]
            # Remove empty tags
            tags = [tag for tag in tags if tag]

            # Determine confidence based on outcome
            confidence = 0.9 if metadata['outcome'] == 'WIN' else 0.7

            # Add to memory using add_lesson_memory
            self.memory_manager.add_lesson_memory(
                lesson_title=lesson_title,
                lesson_content=lesson_content,
                category="trading",
                confidence=confidence,
                tags=tags,
                source_conversation=None
            )

            return True

        except Exception as e:
            print(f"❌ Error feeding pattern: {e}")
            return False

    def feed_level(self, level):
        """Feed a support/resistance level directly to memory"""
        try:
            # Extract metadata
            metadata = level['metadata']

            # Create lesson title
            lesson_title = f"{metadata['type'].title()} Level: {metadata['level']:.2f} - {metadata['symbol']}"

            # Use the level text as content
            lesson_content = level['text']

            # Determine tags
            tags = [
                metadata['type'],
                metadata['symbol'],
                metadata['timeframe'],
                metadata['strength'],
                'support_resistance',
                'key_level',
                'price_action'
            ]

            # Determine confidence based on strength
            confidence = 0.85 if metadata['strength'] == 'strong' else 0.7

            # Add to memory
            self.memory_manager.add_lesson_memory(
                lesson_title=lesson_title,
                lesson_content=lesson_content,
                category="technical",
                confidence=confidence,
                tags=tags,
                source_conversation=None
            )

            return True

        except Exception as e:
            print(f"❌ Error feeding level: {e}")
            return False

    def feed_from_file(self, json_file):
        """Feed patterns and levels from a processed JSON file"""
        print(f"\nProcessing: {json_file}")
        print("-"*60)

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            patterns = data.get('patterns', [])
            levels = data.get('levels', [])

            print(f"Found {len(patterns)} patterns and {len(levels)} levels")

            # Feed patterns
            pattern_success = 0
            for i, pattern in enumerate(patterns):
                if self.feed_pattern(pattern):
                    pattern_success += 1
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(patterns)} patterns...")

            # Feed levels
            level_success = 0
            for i, level in enumerate(levels):
                if self.feed_level(level):
                    level_success += 1
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(levels)} levels...")

            print(f"\n✅ Successfully fed {pattern_success}/{len(patterns)} patterns")
            print(f"✅ Successfully fed {level_success}/{len(levels)} levels")

            return {
                'patterns_total': len(patterns),
                'patterns_success': pattern_success,
                'levels_total': len(levels),
                'levels_success': level_success
            }

        except Exception as e:
            print(f"❌ Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Feed processed trading data directly to ChromaDB')
    parser.add_argument('--file', help='Process single JSON file')
    parser.add_argument('--dir', default='./data/processed', help='Process all JSON files in directory')
    parser.add_argument('--chroma-dir', default='./chroma_db', help='ChromaDB directory')

    args = parser.parse_args()

    print("="*60)
    print("DIRECT CHROMADB TRADING PATTERN FEEDER")
    print("="*60)

    feeder = DirectTradingPatternFeeder(args.chroma_dir)

    if args.file:
        # Process single file
        result = feeder.feed_from_file(args.file)
        if result:
            print(f"\n✅ Successfully processed {args.file}")
    else:
        # Process all JSON files in directory
        json_files = list(Path(args.dir).glob("*_processed.json"))
        if not json_files:
            print(f"\n❌ No processed JSON files found in {args.dir}")
            print("Run the data processor first: python scripts/data_processor.py <csv_file>")
            sys.exit(1)

        print(f"\nFound {len(json_files)} processed files")
        print("="*60)

        results = {}
        for json_file in json_files:
            result = feeder.feed_from_file(str(json_file))
            if result:
                results[json_file.name] = result

        # Summary
        print("\n" + "="*60)
        print("FEEDING SUMMARY")
        print("="*60)

        total_patterns = sum(r['patterns_success'] for r in results.values())
        total_levels = sum(r['levels_success'] for r in results.values())

        print(f"Total patterns fed: {total_patterns}")
        print(f"Total levels fed: {total_levels}")
        print(f"Total documents: {total_patterns + total_levels}")
        print("\n✅ All data fed to ChromaDB successfully!")

if __name__ == "__main__":
    main()
