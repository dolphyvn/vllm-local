#!/usr/bin/env python3
"""
Simple MT5 Data Processor
Converts MT5 CSV export to RAG-compatible format
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class MT5DataProcessor:
    """Process MT5 CSV data for RAG system"""

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None

    def load_data(self):
        """Load and clean MT5 data"""
        print(f"Loading MT5 data: {self.csv_file}")

        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_file, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any supported encoding")

        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

        # Standardize column names for MT5 format
        column_mapping = {
            'DateTime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume',
            'Candle': 'candle_id',
            'TimeFrame': 'timeframe',
            'Symbol': 'symbol'
        }

        self.df = self.df.rename(columns={k: v for k, v in column_mapping.items() if k in self.df.columns})

        # Parse timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        print(f"Loaded {len(self.df)} candles from {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        return True

    def calculate_basic_indicators(self):
        """Calculate basic technical indicators"""
        print("Calculating indicators...")

        # Simple moving averages
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()

        # Price changes
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['price_change_abs'] = abs(self.df['price_change'])

        # High-Low spread
        self.df['hl_range'] = self.df['high'] - self.df['low']
        self.df['hl_range_pct'] = (self.df['hl_range'] / self.df['close']) * 100

        # Volume analysis
        self.df['volume_sma'] = self.df['tick_volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['tick_volume'] / self.df['volume_sma']

        # Drop NaN values from calculations
        self.df = self.df.dropna().reset_index(drop=True)

        print(f"Indicators calculated. {len(self.df)} candles ready.")
        return self.df

    def detect_simple_patterns(self):
        """Detect basic patterns"""
        print("Detecting patterns...")

        patterns = []

        for i in range(50, len(self.df) - 10):  # Need history and future

            current = self.df.iloc[i]
            prev_candles = self.df.iloc[i-10:i]
            future_candles = self.df.iloc[i+1:i+11]

            # Large move pattern
            if current['hl_range_pct'] > 0.3:  # Range > 0.3%
                pattern = self.create_pattern_document(
                    i,
                    "Large Range Candle",
                    current,
                    prev_candles,
                    future_candles
                )
                patterns.append(pattern)

            # High volume pattern
            if current['volume_ratio'] > 2.0:  # Volume > 2x average
                pattern = self.create_pattern_document(
                    i,
                    "High Volume Event",
                    current,
                    prev_candles,
                    future_candles
                )
                patterns.append(pattern)

            # SMA crossover pattern
            if (self.df.iloc[i-1]['sma_20'] < self.df.iloc[i-1]['sma_50'] and
                current['sma_20'] > current['sma_50']):
                pattern = self.create_pattern_document(
                    i,
                    "SMA Bullish Crossover",
                    current,
                    prev_candles,
                    future_candles
                )
                patterns.append(pattern)
            elif (self.df.iloc[i-1]['sma_20'] > self.df.iloc[i-1]['sma_50'] and
                  current['sma_20'] < current['sma_50']):
                pattern = self.create_pattern_document(
                    i,
                    "SMA Bearish Crossover",
                    current,
                    prev_candles,
                    future_candles
                )
                patterns.append(pattern)

        print(f"Found {len(patterns)} patterns")
        return patterns

    def create_pattern_document(self, idx, pattern_name, current, prev_candles, future_candles):
        """Create RAG document for pattern"""

        # Calculate outcome
        max_high = future_candles['high'].max()
        min_low = future_candles['low'].min()
        entry_price = current['close']

        if max_high - entry_price > entry_price - min_low:
            outcome = "Bullish"
            pnl = max_high - entry_price
        else:
            outcome = "Bearish"
            pnl = entry_price - min_low

        # Determine trend
        if current['close'] > current['sma_20'] > current['sma_50']:
            trend = "Bullish"
        elif current['close'] < current['sma_20'] < current['sma_50']:
            trend = "Bearish"
        else:
            trend = "Neutral"

        # Get session
        hour = current['timestamp'].hour
        if 8 <= hour < 16:
            session = "London"
        elif 13 <= hour < 22:
            session = "New York"
        else:
            session = "Asia"

        # Build narrative text
        text = f"""Pattern: {pattern_name}
Symbol: XAUUSD M15
Date: {current['timestamp'].strftime('%Y-%m-%d %H:%M')}

=== MARKET CONDITIONS ===
Price: {current['close']:.2f}
Range: {current['hl_range']:.2f} points ({current['hl_range_pct']:.2f}%)
Volume: {current['tick_volume']} ({current['volume_ratio']:.1f}x average)
Session: {session}
Trend: {trend}

=== TECHNICAL SETUP ===
SMA20: {current['sma_20']:.2f}
SMA50: {current['sma_50']:.2f}
Price vs SMA20: {current['close'] - current['sma_20']:+.2f}
Previous 10-candle range: {prev_candles['low'].min():.2f} to {prev_candles['high'].max():.2f}

=== TRADING OPPORTUNITY ===
Signal: {pattern_name}
Entry: {current['close']:.2f}
Direction: {outcome.lower()}
Expected move: {pnl:.2f} points
Risk level: {'High' if current['volume_ratio'] > 2 else 'Medium' if current['volume_ratio'] > 1.5 else 'Low'}

=== OUTCOME ANALYSIS ===
Next 10 candles: High {max_high:.2f}, Low {min_low:.2f}
Actual outcome: {outcome}
P&L potential: {pnl:+.2f} points
Success probability: Based on similar {trend.lower()} conditions in {session} session

=== MARKET CONTEXT ===
This {pattern_name.lower()} occurred during {trend.lower()} conditions with
{'high' if current['volume_ratio'] > 2 else 'normal'} volume participation.
The pattern suggests {outcome.lower()} momentum with {pnl:.2f} points potential.
"""

        # Metadata for filtering
        metadata = {
            'pattern': pattern_name,
            'timestamp': current['timestamp'].isoformat(),
            'date': current['timestamp'].strftime('%Y-%m-%d'),
            'time': current['timestamp'].strftime('%H:%M'),
            'hour': current['timestamp'].hour,
            'session': session,
            'symbol': 'XAUUSD',
            'timeframe': 'M15',
            'entry': float(current['close']),
            'outcome': outcome,
            'pnl': float(pnl),
            'trend': trend,
            'volume_ratio': float(current['volume_ratio']),
            'range_pct': float(current['hl_range_pct'])
        }

        return {
            'text': text,
            'metadata': metadata
        }

    def process_all(self):
        """Run complete processing pipeline"""
        print("\n" + "="*60)
        print("MT5 DATA PROCESSING PIPELINE")
        print("="*60 + "\n")

        # Load data
        if not self.load_data():
            return None

        # Calculate indicators
        self.calculate_basic_indicators()

        # Detect patterns
        patterns = self.detect_simple_patterns()

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total patterns found: {len(patterns)}")

        return {
            'patterns': patterns,
            'summary': {
                'total_candles': len(self.df),
                'patterns_found': len(patterns),
                'date_range': f"{self.df['timestamp'].min()} to {self.df['timestamp'].max()}",
                'symbol': 'XAUUSD',
                'timeframe': 'M15'
            }
        }

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mt5_data_processor.py <mt5_csv_file>")
        print("Example: python mt5_data_processor.py data/XAUUSD_PERIOD_M15_200.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Process
    processor = MT5DataProcessor(csv_file)
    results = processor.process_all()

    if results:
        # Save output
        output_file = csv_file.replace('.csv', '_rag_ready.json')

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nOutput saved to: {output_file}")
        print(f"Ready for RAG system integration!")

        # Show sample
        if results['patterns']:
            print(f"\nSample pattern document:")
            print("-" * 60)
            print(results['patterns'][0]['text'][:300] + "...")

if __name__ == "__main__":
    main()