#!/usr/bin/env python3
"""
Create RAG-compatible JSON from processed data
"""

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime
import numpy as np

def create_rag_json_from_csv(csv_file, json_output):
    """Create RAG-compatible JSON from CSV"""

    print(f"🔄 Creating RAG JSON from {csv_file}...")

    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} records")

        trading_patterns = []
        support_resistance_levels = []

        # Process each row to create trading patterns
        for i, row in df.iterrows():
            if i < 50:  # Skip first 50 rows to have enough history for indicators
                continue

            if i % 100 != 0:  # Sample every 100th row to avoid too many patterns
                continue

            try:
                # Calculate trend direction
                rsi = row.get('RSI', 50)
                ema20 = row.get('EMA20', row['Close'])
                ema50 = row.get('EMA50', row['Close'])
                close_price = row['Close']

                trend_direction = "NEUTRAL"
                if rsi > 70:
                    trend_direction = "OVERBOUGHT"
                elif rsi < 30:
                    trend_direction = "OVERSOLD"
                elif close_price > ema20 > ema50:
                    trend_direction = "BULLISH"
                elif close_price < ema20 < ema50:
                    trend_direction = "BEARISH"

                # Determine trading session
                timestamp = pd.to_datetime(row['DateTime'])
                hour = timestamp.hour
                if 0 <= hour < 8:
                    session = "ASIAN"
                elif 8 <= hour < 13:
                    session = "LONDON"
                elif 13 <= hour < 17:
                    session = "NEW_YORK"
                else:
                    session = "US_LATE"

                # Create pattern
                pattern = {
                    "timestamp": row['DateTime'],
                    "timeframe": row.get('TimeFrame', 'M15'),
                    "symbol": row.get('Symbol', 'XAUUSD'),
                    "pattern": f"{trend_direction}_TREND",
                    "direction": "BUY" if trend_direction == "BULLISH" else "SELL" if trend_direction == "BEARISH" else "HOLD",
                    "session": session,
                    "rsi": float(rsi),
                    "ema_20": float(ema20),
                    "ema_50": float(ema50),
                    "vwap": float(row.get('VWAP', close_price)),
                    "entry_trigger": float(close_price),
                    "stop_loss": float(close_price * 0.98 if trend_direction == "BULLISH" else close_price * 1.02),
                    "confidence": float(max(30, min(90, 70 - abs(rsi - 50) * 0.4))),
                    "volume_ratio": float(row.get('Volume', 1000) / 1000),
                    "vwap_deviation": float(((close_price - row.get('VWAP', close_price)) / row.get('VWAP', close_price)) * 100)
                }

                trading_patterns.append(pattern)

                # Create support/resistance levels from recent highs/lows
                if i % 500 == 0:  # Every 500th row
                    high_20 = df.iloc[max(0, i-20):i+1]['High'].max()
                    low_20 = df.iloc[max(0, i-20):i+1]['Low'].min()

                    support_resistance_levels.append({
                        "timestamp": row['DateTime'],
                        "level_type": "RESISTANCE",
                        "price": float(high_20),
                        "strength": float(np.random.uniform(0.6, 0.9)),
                        "timeframe": row.get('TimeFrame', 'M15'),
                        "symbol": row.get('Symbol', 'XAUUSD')
                    })

                    support_resistance_levels.append({
                        "timestamp": row['DateTime'],
                        "level_type": "SUPPORT",
                        "price": float(low_20),
                        "strength": float(np.random.uniform(0.6, 0.9)),
                        "timeframe": row.get('TimeFrame', 'M15'),
                        "symbol": row.get('Symbol', 'XAUUSD')
                    })

            except Exception as e:
                print(f"⚠️ Error processing row {i}: {e}")
                continue

        # Create final JSON structure
        rag_data = {
            "patterns": trading_patterns,
            "levels": support_resistance_levels,
            "metadata": {
                "total_patterns": len(trading_patterns),
                "total_levels": len(support_resistance_levels),
                "source_file": os.path.basename(csv_file),
                "created_at": datetime.now().isoformat()
            }
        }

        # Save to JSON
        with open(json_output, 'w') as f:
            json.dump(rag_data, f, indent=2)

        print(f"✅ Successfully created RAG JSON: {json_output}")
        print(f"📊 Created {len(trading_patterns)} trading patterns")
        print(f"📈 Created {len(support_resistance_levels)} support/resistance levels")

        return True

    except Exception as e:
        print(f"❌ Error creating RAG JSON: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create RAG-compatible JSON from CSV')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', help='Output JSON file (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"data/{base_name}_rag.json"

    print("="*60)
    print("🔄 CSV TO RAG JSON CONVERTER")
    print("="*60)

    success = create_rag_json_from_csv(args.input, args.output)

    if success:
        print(f"\n✅ RAG JSON creation completed!")
        print(f"📁 Output file: {args.output}")
        print(f"💡 You can now feed this to RAG: python scripts/feed_to_rag_direct.py --file {args.output}")
    else:
        print(f"\n❌ RAG JSON creation failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())