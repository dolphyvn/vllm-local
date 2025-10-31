#!/usr/bin/env python3
"""
Convert processed JSON back to RAG-compatible CSV format
"""

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime

def convert_json_to_rag_csv(json_file, csv_output):
    """Convert processed JSON to RAG CSV format"""

    print(f"🔄 Converting {json_file} to RAG CSV format...")

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        candles = data.get('data', [])

        print(f"📊 Processing {len(candles)} candles")

        # Convert to DataFrame
        rows = []
        for candle in candles:
            indicators = candle.get('indicators', {})

            row = {
                'TimeFrame': metadata.get('timeframe', 'M15'),
                'Symbol': metadata.get('symbol', 'XAUUSD'),
                'Candle': f"{candle['open']}-{candle['high']}-{candle['low']}-{candle['close']}",
                'DateTime': candle['timestamp'],
                'Open': candle['open'],
                'High': candle['high'],
                'Low': candle['low'],
                'Close': candle['close'],
                'Volume': candle.get('volume', 0),
                'HL': candle['high'] - candle['low'],
                'Body': abs(candle['close'] - candle['open']),
                'RSI': indicators.get('rsi', 50),
                'EMA20': indicators.get('ema_20', candle['close']),
                'EMA50': indicators.get('ema_50', candle['close']),
                'VWAP': indicators.get('vwap', candle['close']),
                'MACD': indicators.get('macd', 0),
                'ATR': indicators.get('atr', 0),
                'BB_Upper': indicators.get('bb_upper', candle['close'] * 1.02),
                'BB_Lower': indicators.get('bb_lower', candle['close'] * 0.98)
            }

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv(csv_output, index=False)

        print(f"✅ Successfully converted to {csv_output}")
        print(f"📊 Saved {len(df)} rows with {len(df.columns)} columns")

        return True

    except Exception as e:
        print(f"❌ Error converting: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert processed JSON to RAG CSV format')
    parser.add_argument('--file', required=True, help='Input JSON file')
    parser.add_argument('--output', help='Output CSV file (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Input file not found: {args.file}")
        return 1

    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        args.output = f"data/{base_name}_rag_format.csv"

    print("="*60)
    print("🔄 JSON TO RAG CSV CONVERTER")
    print("="*60)

    success = convert_json_to_rag_csv(args.file, args.output)

    if success:
        print(f"\n✅ Conversion completed!")
        print(f"📁 Output file: {args.output}")
        print(f"💡 You can now feed this to RAG: python scripts/feed_to_rag_direct.py --file {args.output}")
    else:
        print(f"\n❌ Conversion failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())