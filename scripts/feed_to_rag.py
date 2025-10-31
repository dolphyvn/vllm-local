#!/usr/bin/env python3
"""
Feed processed trading patterns to the RAG system
Integrates with the existing Financial Assistant knowledge feeding system
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from knowledge_feeder import KnowledgeEntry, KnowledgeCategory
    KNOWLEDGE_FEEDER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_FEEDER_AVAILABLE = False
    print("Warning: knowledge_feeder module not available")

class TradingPatternFeeder:
    """Feed trading patterns to RAG system"""

    def __init__(self, base_url="http://localhost:8080", password="admin123"):
        self.base_url = base_url
        self.password = password
        self.token = None
        self.session = requests.Session()

        # Authenticate
        self.authenticate()

    def authenticate(self):
        """Authenticate with the API"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"password": self.password}
            )

            if response.status_code == 200:
                self.token = response.json().get("session_token")
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def feed_pattern(self, pattern):
        """Feed a single pattern to the knowledge base"""
        try:
            # Extract metadata
            metadata = pattern['metadata']

            # Create topic string
            topic = f"{metadata['pattern']} - {metadata['symbol']} {metadata['timeframe']}"

            # Determine tags from metadata
            tags = [
                metadata['pattern'],
                metadata['symbol'],
                metadata['timeframe'],
                metadata['outcome'],
                metadata['trend'],
                metadata.get('session', ''),
                metadata.get('market_regime', '')
            ]
            # Remove empty tags
            tags = [tag for tag in tags if tag]

            # Create knowledge entry compatible with your existing API
            knowledge_data = {
                "topic": topic,
                "content": pattern['text'],
                "category": "trading",  # Using existing category
                "confidence": 0.9 if metadata['outcome'] == 'WIN' else 0.7,
                "tags": tags,
                "source": "MT5 CSV Analysis",
                "priority": 8 if metadata['outcome'] == 'WIN' else 5
            }

            response = self.session.post(
                f"{self.base_url}/api/knowledge/add",
                json=knowledge_data
            )

            if response.status_code == 200:
                return True
            else:
                print(f"❌ Failed to add pattern: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error feeding pattern: {e}")
            return False

    def feed_level(self, level):
        """Feed a support/resistance level to the knowledge base"""
        try:
            # Extract metadata
            metadata = level['metadata']

            # Create topic string
            topic = f"{metadata['type'].title()} Level: {metadata['level']} - {metadata['symbol']}"

            # Determine tags
            tags = [
                metadata['type'],
                metadata['symbol'],
                metadata['timeframe'],
                metadata['strength'],
                'support_resistance',
                'key_level'
            ]

            # Create knowledge entry
            knowledge_data = {
                "topic": topic,
                "content": level['text'],
                "category": "technical",  # Using existing category
                "confidence": 0.85 if metadata['strength'] == 'strong' else 0.7,
                "tags": tags,
                "source": "MT5 CSV Analysis",
                "priority": 7 if metadata['strength'] == 'strong' else 5
            }

            response = self.session.post(
                f"{self.base_url}/api/knowledge/add",
                json=knowledge_data
            )

            if response.status_code == 200:
                return True
            else:
                print(f"❌ Failed to add level: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error feeding level: {e}")
            return False

    def feed_from_file(self, json_file):
        """Feed patterns and levels from a processed JSON file"""
        print(f"\nFeeding from: {json_file}")
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
                if (i + 1) % 100 == 0:
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
            return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Feed processed trading data to RAG system')
    parser.add_argument('--file', help='Process single JSON file')
    parser.add_argument('--dir', default='./data/processed', help='Process all JSON files in directory')
    parser.add_argument('--base-url', default='http://localhost:8080', help='API base URL')
    parser.add_argument('--password', default='admin123', help='API password')

    args = parser.parse_args()

    feeder = TradingPatternFeeder(args.base_url, args.password)

    if not feeder.token:
        print("❌ Failed to authenticate. Exiting.")
        sys.exit(1)

    if args.file:
        # Process single file
        feeder.feed_from_file(args.file)
    else:
        # Process all JSON files in directory
        json_files = list(Path(args.dir).glob("*_processed.json"))
        print(f"Found {len(json_files)} processed files")
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

if __name__ == "__main__":
    main()
