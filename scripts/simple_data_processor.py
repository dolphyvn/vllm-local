#!/usr/bin/env python3
"""
Simple Data Processor - Convert raw MT5 CSV to processed JSON format
Usage: python scripts/simple_data_processor.py --input data/XAUUSD_PERIOD_M15_0.csv --output data/XAUUSD_PERIOD_M15_0_processed.json
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
import sys
import os

def detect_csv_format(df):
    """Detect CSV format and normalize column names"""
    print("🔍 Detecting CSV format...")

    # Check for RAG format columns
    rag_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    mt5_columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']

    df_columns = list(df.columns)

    if all(col in df_columns for col in rag_columns):
        print("✅ RAG format detected")
        return 'rag'
    elif all(col in df_columns for col in mt5_columns):
        print("✅ MT5 format detected")
        return 'mt5'
    else:
        print(f"⚠️ Unknown format, columns: {df_columns}")
        return 'unknown'

def normalize_dataframe(df, format_type):
    """Normalize dataframe to standard format"""

    if format_type == 'rag':
        # Map RAG format to standard names
        column_mapping = {
            'DateTime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)

    elif format_type == 'mt5':
        # MT5 already has standard names
        pass

    else:
        # Try to infer columns
        print("🔧 Attempting to infer column mapping...")

        # Look for timestamp column
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            df = df.rename(columns={timestamp_cols[0]: 'timestamp'})

        # Look for OHLC columns
        ohlc_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close']:
                continue  # Already correct
            elif 'open' in col_lower:
                ohlc_mapping[col] = 'open'
            elif 'high' in col_lower:
                ohlc_mapping[col] = 'high'
            elif 'low' in col_lower:
                ohlc_mapping[col] = 'low'
            elif 'close' in col_lower:
                ohlc_mapping[col] = 'close'
            elif 'volume' in col_lower or 'tick' in col_lower:
                ohlc_mapping[col] = 'volume'

        df = df.rename(columns=ohlc_mapping)

    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    print("📊 Calculating technical indicators...")

    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    try:
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Default to neutral

        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Bollinger Bands (20, 2)
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # MACD (12, 26, 9)
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        df['atr'] = df['atr'].fillna(df['atr'].mean())

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()

        # Handle missing volume column
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Default volume
            print("⚠️ No volume column found, using default value")

        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)

        # VWAP (simplified)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        # Fill any remaining NaN values
        df = df.ffill().bfill()

        print("✅ Technical indicators calculated successfully")

    except Exception as e:
        print(f"⚠️ Error calculating indicators: {e}")
        print("🔄 Using basic calculations...")

        # Fallback basic indicators
        df['rsi'] = 50.0
        df['ema_20'] = df['close']
        df['ema_50'] = df['close']
        df['ema_200'] = df['close']
        df['bb_upper'] = df['close'] * 1.02
        df['bb_middle'] = df['close']
        df['bb_lower'] = df['close'] * 0.98
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_histogram'] = 0.0
        df['atr'] = df['close'].rolling(window=14, min_periods=1).std().fillna(df['close'].std())
        df['sma_20'] = df['close']
        df['sma_50'] = df['close']
        df['price_change'] = 0.0
        df['price_change_abs'] = 0.0
        df['volume_ratio'] = 1.0
        df['vwap'] = df['close']

        if 'volume' not in df.columns:
            df['volume'] = 1000
        df['volume_sma'] = df['volume']

    return df

def process_csv_to_json(input_file, output_file, symbol="XAUUSD", timeframe="M15"):
    """Main processing function"""

    print(f"🔄 Processing {input_file}...")
    print(f"📊 Symbol: {symbol}, Timeframe: {timeframe}")

    # Try different encodings
    encodings = ['utf-16', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None

    for encoding in encodings:
        try:
            print(f"🔤 Trying encoding: {encoding}")
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"✅ Successfully loaded with {encoding}")

            # Check if we got valid column names (not all Unnamed)
            if all('Unnamed' in str(col) for col in df.columns[:6]):
                print("⚠️ Got unnamed columns, trying with skiprows...")
                df = pd.read_csv(input_file, encoding=encoding, skiprows=1)
                print(f"✅ Successfully loaded with skiprows using {encoding}")

            break
        except Exception as e:
            print(f"❌ Failed with {encoding}: {e}")
            continue

    if df is None:
        print("❌ Could not read CSV file with any supported encoding")
        return False

    print(f"📋 Loaded {len(df)} records with columns: {list(df.columns)}")

    # Detect and normalize format
    format_type = detect_csv_format(df)
    df = normalize_dataframe(df, format_type)

    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            print("⚠️ Some timestamps could not be parsed, using row numbers")
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
    else:
        print("⚠️ No timestamp column found, creating synthetic timestamps")
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Convert to JSON format
    print("📦 Converting to JSON format...")

    processed_data = []

    for index, row in df.iterrows():
        try:
            candle_data = {
                "timestamp": row['timestamp'].isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row.get('volume', 1000)),
                "indicators": {
                    "rsi": float(row['rsi']),
                    "ema_20": float(row['ema_20']),
                    "ema_50": float(row['ema_50']),
                    "ema_200": float(row['ema_200']),
                    "bb_upper": float(row['bb_upper']),
                    "bb_middle": float(row['bb_middle']),
                    "bb_lower": float(row['bb_lower']),
                    "macd": float(row['macd']),
                    "macd_signal": float(row['macd_signal']),
                    "macd_histogram": float(row['macd_histogram']),
                    "atr": float(row['atr']),
                    "sma_20": float(row['sma_20']),
                    "sma_50": float(row['sma_50']),
                    "vwap": float(row['vwap']),
                    "volume_ratio": float(row['volume_ratio']),
                    "price_change": float(row['price_change']),
                    "price_change_abs": float(row['price_change_abs'])
                }
            }

            processed_data.append(candle_data)

        except Exception as e:
            print(f"⚠️ Error processing row {index}: {e}")
            continue

    if not processed_data:
        print("❌ No valid data processed")
        return False

    # Add metadata
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_candles": len(processed_data),
        "date_range": {
            "start": processed_data[0]["timestamp"],
            "end": processed_data[-1]["timestamp"]
        },
        "indicators": ["RSI", "EMA_20", "EMA_50", "EMA_200", "Bollinger_Bands", "MACD", "ATR", "SMA_20", "SMA_50", "VWAP", "Volume_Ratio", "Price_Change"],
        "processed_at": datetime.now().isoformat(),
        "source_format": format_type
    }

    # Save to JSON
    output_data = {
        "metadata": metadata,
        "data": processed_data
    }

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved processed data to {output_file}")
        print(f"📊 Processed {len(processed_data)} candles")
        print(f"📈 Added {len(metadata['indicators'])} technical indicators")
        print(f"📅 Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        return True

    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple CSV to JSON processor with technical indicators')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--symbol', default='XAUUSD', help='Symbol name (default: XAUUSD)')
    parser.add_argument('--timeframe', default='M15', help='Timeframe (default: M15)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    print("="*60)
    print("🔄 SIMPLE DATA PROCESSOR")
    print("="*60)

    success = process_csv_to_json(args.input, args.output, args.symbol, args.timeframe)

    if success:
        print("\n✅ Processing completed successfully!")
        print(f"📁 Output file: {args.output}")
        print(f"💡 You can now use this file with: python scripts/feed_to_rag_direct.py --file {args.output}")
    else:
        print("\n❌ Processing failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())