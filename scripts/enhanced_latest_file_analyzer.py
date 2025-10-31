#!/usr/bin/env python3
"""
Enhanced Latest File Analysis Script
Converts CSV to JSON/RAG format and sends detailed data to LLM for analysis
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

def list_csv_files(data_dir="./data", pattern="*.csv"):
    """
    List all CSV files in the data directory with modification times
    """
    print(f"🔍 Scanning CSV files in: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return []

    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(data_dir, pattern))

    if not csv_files:
        print(f"❌ No CSV files found in: {data_dir}")
        return []

    # Sort by modification time (newest first) and collect info
    files_info = []
    for file_path in csv_files:
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        file_size = os.path.getsize(file_path)
        files_info.append({
            'path': file_path,
            'name': os.path.basename(file_path),
            'modified': mod_time,
            'size': file_size
        })

    files_info.sort(key=lambda x: x['modified'], reverse=True)

    print(f"✅ Found {len(files_info)} CSV files:")
    for i, info in enumerate(files_info, 1):
        print(f"   {i:2d}. {info['name']:<30} | {info['modified']} | {info['size']:,} bytes")

    return files_info

def find_latest_csv_file(data_dir="./data", pattern="*.csv"):
    """
    Find the most recently modified CSV file in the data directory
    """
    files_info = list_csv_files(data_dir, pattern)

    if not files_info:
        return None

    # Return the latest file
    latest_info = files_info[0]
    print(f"🎯 Selected latest file:")
    print(f"   📁 File: {latest_info['name']}")
    print(f"   📅 Modified: {latest_info['modified']}")
    print(f"   📊 Size: {latest_info['size']:,} bytes")

    return latest_info['path']

def convert_csv_to_rag_json(file_path):
    """
    Convert CSV file to detailed RAG format with all candlestick data
    """
    print(f"\n🔄 Converting to RAG JSON format...")

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Detect format and normalize columns
        if all(col in df.columns for col in ['DateTime', 'Open', 'High', 'Low', 'Close']):
            format_type = "RAG Format"
            # Already in RAG format
            df_clean = df.copy()
        elif all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
            format_type = "MT5 Format"
            # Convert MT5 to RAG format
            df_clean = pd.DataFrame({
                'DateTime': pd.to_datetime(df['timestamp']),
                'Open': df['open'],
                'High': df['high'],
                'Low': df['low'],
                'Close': df['close'],
                'Volume': df.get('tick_volume', 0)
            })
        else:
            print(f"❌ Unknown format: {df.columns}")
            return None

        # Detect timeframe from filename
        filename = os.path.basename(file_path).upper()
        timeframe = "UNKNOWN"
        if 'M1' in filename:
            timeframe = "M1"
        elif 'M5' in filename:
            timeframe = "M5"
        elif 'M15' in filename:
            timeframe = "M15"
        elif 'M30' in filename:
            timeframe = "M30"
        elif 'H1' in filename:
            timeframe = "H1"
        elif 'H4' in filename:
            timeframe = "H4"
        elif 'D1' in filename:
            timeframe = "D1"

        # Create detailed RAG JSON structure
        rag_data = {
            "metadata": {
                "symbol": "XAUUSD",
                "timeframe": timeframe,
                "total_candles": len(df_clean),
                "source_file": os.path.basename(file_path),
                "format": format_type,
                "processed_at": datetime.now().isoformat()
            },
            "price_analysis": {
                "current_price": float(df_clean['Close'].iloc[-1]),
                "price_range": {
                    "high": float(df_clean['High'].max()),
                    "low": float(df_clean['Low'].min())
                },
                "average_price": float(df_clean['Close'].mean()),
                "price_volatility": float(df_clean['Close'].std())
            },
            "all_candles": []
        }

        # Add ALL candlestick data (not just the last 20)
        for idx, row in df_clean.iterrows():
            candle = {
                "datetime": str(row['DateTime']),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row.get('Volume', 0)),
                "price_range": float(row['High'] - row['Low']),
                "body_size": abs(float(row['Close'] - row['Open'])),
                "candle_type": "bullish" if row['Close'] > row['Open'] else "bearish"
            }
            rag_data["all_candles"].append(candle)

        # Add comprehensive technical analysis including Market Profile and VWAP
        close_prices = df_clean['Close'].values
        volumes = df_clean['Volume'].values if 'Volume' in df_clean.columns else df_clean.get('tick_volume', pd.Series([1]*len(df_clean))).values

        # Calculate VWAP (Volume Weighted Average Price)
        cumulative_volume = np.cumsum(volumes)
        cumulative_price_volume = np.cumsum(close_prices * volumes)
        vwap = cumulative_price_volume / cumulative_volume
        current_vwap = vwap[-1]
        vwap_deviation = ((close_prices[-1] - current_vwap) / current_vwap) * 100

        # Calculate Market Profile metrics
        price_ranges = df_clean['High'] - df_clean['Low']
        avg_range = price_ranges.mean()
        current_range = price_ranges.iloc[-1]

        # Find POC (Point of Control) - price with highest volume
        if len(df_clean) > 20:
            recent_window = df_clean.tail(20)
            poc_idx = recent_window['Volume'].idxmax() if 'Volume' in recent_window.columns else recent_window.index[len(recent_window)//2]
            poc_price = recent_window.loc[poc_idx, 'Close']
        else:
            poc_price = close_prices[len(close_prices)//2]

        # Calculate Value Area (simplified - 70% of volume around POC)
        price_volume_map = {}
        for idx, row in df_clean.iterrows():
            price = row['Close']
            volume = row.get('Volume', row.get('tick_volume', 1))
            price_volume_map[price] = price_volume_map.get(price, 0) + volume

        sorted_prices = sorted(price_volume_map.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volumes)
        cumulative_volume_pct = 0
        value_area_high = None
        value_area_low = None

        for price, volume in sorted_prices:
            cumulative_volume_pct += (volume / total_volume) * 100
            if value_area_high is None:
                value_area_high = price
            value_area_low = price
            if cumulative_volume_pct >= 70:  # 70% value area
                break

        rag_data["technical_patterns"] = {
            "trend_direction": "uptrend" if close_prices[-1] > close_prices[-min(10, len(close_prices)-1)] else "downtrend",
            "volatility_trend": "increasing" if df_clean['Close'].std() > df_clean['Close'].head(min(10, len(df_clean)-1)).std() else "stable",
            "price_momentum": float(close_prices[-1] - close_prices[-min(5, len(close_prices)-1)]) if len(close_prices) >= 5 else 0,

            # VWAP Analysis
            "vwap_analysis": {
                "current_vwap": float(current_vwap),
                "vwap_deviation_percent": float(vwap_deviation),
                "price_above_vwap": close_prices[-1] > current_vwap,
                "vwap_trend": "rising" if len(vwap) > 1 and vwap[-1] > vwap[-min(5, len(vwap)-1)] else "falling"
            },

            # Market Profile Analysis
            "market_profile": {
                "poc_price": float(poc_price),
                "value_area_high": float(value_area_high) if value_area_high else None,
                "value_area_low": float(value_area_low) if value_area_low else None,
                "value_area_width": float(value_area_high - value_area_low) if value_area_high and value_area_low else None,
                "current_price_in_value_area": value_area_low <= close_prices[-1] <= value_area_high if value_area_low and value_area_high else None,
                "avg_range": float(avg_range),
                "current_range": float(current_range),
                "range_expansion": current_range > avg_range * 1.2
            },

            # Volume Analysis
            "volume_analysis": {
                "avg_volume": float(np.mean(volumes)),
                "current_volume": float(volumes[-1]),
                "volume_ratio": float(volumes[-1] / np.mean(volumes)),
                "volume_trend": "increasing" if len(volumes) > 5 and np.mean(volumes[-3:]) > np.mean(volumes[-6:-3]) else "decreasing"
            }
        }

        print(f"✅ Converted {len(df_clean)} candles to RAG format")
        print(f"   Timeframe: {timeframe}")
        print(f"   Current Price: {rag_data['price_analysis']['current_price']}")
        print(f"   All Candles: {len(rag_data['all_candles'])}")

        return rag_data

    except Exception as e:
        print(f"❌ Error converting CSV to RAG: {e}")
        return None

def create_detailed_llm_prompt(rag_data, filename):
    """
    Create a detailed prompt with all the candlestick data for LLM analysis
    """
    current_price = rag_data['price_analysis']['current_price']
    timeframe = rag_data['metadata']['timeframe']

    prompt = f"""
You are an expert trader with deep expertise in technical analysis, market data analysis, and risk management.

COMPREHENSIVE MARKET DATA ANALYSIS:
======================================

File: {filename}
Timeframe: {timeframe}
Total Candles: {rag_data['metadata']['total_candles']}
Current Price: {current_price}

PRICE ANALYSIS:
- Current Price: ${current_price:.2f}
- Price Range: ${rag_data['price_analysis']['price_range']['low']:.2f} - ${rag_data['price_analysis']['price_range']['high']:.2f}
- Average Price: ${rag_data['price_analysis']['average_price']:.2f}
- Price Volatility: {rag_data['price_analysis']['price_volatility']:.2f}

COMPLETE CANDLESTICK DATA (All {len(rag_data['all_candles'])} candles):
======================================================"""

    # Add detailed candlestick information for the last 20 candles (to not overwhelm LLM)
    # But note that the full dataset is available for analysis
    recent_for_display = rag_data['all_candles'][-20:]
    prompt += f"""
Note: Full dataset contains {len(rag_data['all_candles'])} candles. Showing last 20 for detailed analysis:

RECENT CANDLESTICK DATA (Last 20 candles):
=========================================="""

    # Add detailed candlestick information for recent candles
    for i, candle in enumerate(recent_for_display):
        prompt += f"""
Candle {len(rag_data['all_candles']) - 20 + i + 1}: {candle['datetime']}
- OHLC: {candle['open']:.2f} / {candle['high']:.2f} / {candle['low']:.2f} / {candle['close']:.2f}
- Range: {candle['price_range']:.2f} | Body: {candle['body_size']:.2f} | Type: {candle['candle_type']}
- Volume: {candle['volume']:.0f}"""

    # Add pattern analysis
    prompt += f"""

TECHNICAL PATTERNS:
==================
- Trend Direction: {rag_data['technical_patterns']['trend_direction']}
- Volatility Trend: {rag_data['technical_patterns']['volatility_trend']}
- Price Momentum: {rag_data['technical_patterns']['price_momentum']:.2f}

VWAP ANALYSIS:
===============
- Current VWAP: ${rag_data['technical_patterns']['vwap_analysis']['current_vwap']:.2f}
- VWAP Deviation: {rag_data['technical_patterns']['vwap_analysis']['vwap_deviation_percent']:.2f}%
- Price vs VWAP: {'Above' if rag_data['technical_patterns']['vwap_analysis']['price_above_vwap'] else 'Below'} VWAP
- VWAP Trend: {rag_data['technical_patterns']['vwap_analysis']['vwap_trend']}

MARKET PROFILE ANALYSIS:
========================
- Point of Control (POC): ${rag_data['technical_patterns']['market_profile']['poc_price']:.2f}
- Value Area High: ${rag_data['technical_patterns']['market_profile']['value_area_high']:.2f}
- Value Area Low: ${rag_data['technical_patterns']['market_profile']['value_area_low']:.2f}
- Value Area Width: {rag_data['technical_patterns']['market_profile']['value_area_width']:.2f} pips
- Current in Value Area: {'Yes' if rag_data['technical_patterns']['market_profile']['current_price_in_value_area'] else 'No'}
- Average Range: {rag_data['technical_patterns']['market_profile']['avg_range']:.2f} pips
- Current Range: {rag_data['technical_patterns']['market_profile']['current_range']:.2f} pips
- Range Expansion: {'Yes' if rag_data['technical_patterns']['market_profile']['range_expansion'] else 'No'}

VOLUME ANALYSIS:
==================
- Average Volume: {rag_data['technical_patterns']['volume_analysis']['avg_volume']:.0f}
- Current Volume: {rag_data['technical_patterns']['volume_analysis']['current_volume']:.0f}
- Volume Ratio: {rag_data['technical_patterns']['volume_analysis']['volume_ratio']:.2f}x
- Volume Trend: {rag_data['technical_patterns']['volume_analysis']['volume_trend']}

ANALYSIS REQUEST:
=================
Based on the detailed candlestick data above, provide comprehensive trading analysis:

1. **TRADE DIRECTION**: Clear BUY/SELL/WAIT recommendation with detailed reasoning

2. **ENTRY PRICE**: Optimal entry level with specific price and justification

3. **STOP LOSS**: Strategic stop-loss placement with reasoning based on recent lows/highs

4. **TAKE PROFIT**: Realistic profit targets (primary & secondary) based on resistance/support

5. **RISK/REWARD**: Risk/reward ratio calculation with specific numbers

6. **TECHNICAL REASONING**:
   - Candlestick pattern analysis
   - Support and resistance levels (including Value Area)
   - Trend strength and momentum
   - Volume analysis and VWAP relationship
   - Market Profile context (POC, Value Area)
   - Key price levels from the data
   - VWAP bounce/breakout scenarios
   - Value Area acceptance/rejection patterns

7. **RISK MANAGEMENT**: Specific stop-loss and position sizing recommendations

**CRITICAL ANALYSIS POINTS**:
- How does current price relate to VWAP (support/resistance)?
- Is price inside or outside the Value Area?
- Any POC retests or Value Area violations?
- Volume confirms the price action?
- Range expansion suggests what about market strength?

Focus on actionable trading signals from the actual price action data provided. Consider the {timeframe} timeframe characteristics and provide realistic, tradable recommendations with specific price levels.
"""

    return prompt

def process_with_enhanced_api(file_path, rag_data, api_base="http://localhost:8080", model="gemma3:1b"):
    """
    Process the enhanced RAG data with the API for detailed analysis
    """
    print(f"\n🔄 Processing enhanced data with API using model: {model}")
    print("=" * 60)

    try:
        import requests

        # First, login to get session token
        print("🔐 Authenticating with API...")
        auth_response = requests.post(f"{api_base}/auth/login",
                                    json={"password": "admin123"},
                                    timeout=30,
                                    headers={'Connection': 'close'})

        if auth_response.status_code != 200:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            return False

        session_token = auth_response.json().get('session_token')
        if not session_token:
            print("❌ No session token received")
            return False

        print("✅ Authentication successful")

        # Add detailed analysis to knowledge base
        print("📚 Adding detailed RAG analysis to knowledge base...")

        # Convert RAG data to JSON string for content
        rag_content = json.dumps(rag_data, indent=2, default=str)

        knowledge_data = {
            "topic": f"Enhanced {rag_data['metadata']['timeframe']} Analysis - {rag_data['metadata']['source_file']}",
            "content": f"""
Detailed RAG Analysis from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

File: {rag_data['metadata']['source_file']}
Timeframe: {rag_data['metadata']['timeframe']}
Candles: {rag_data['metadata']['total_candles']}
Current Price: ${rag_data['price_analysis']['current_price']:.2f}

Complete RAG Data Structure:
{rag_content}

This includes all candlestick data, technical patterns, and detailed price analysis for comprehensive trading analysis.
            """.strip(),
            "metadata": {
                "source": rag_data['metadata']['source_file'],
                "file_path": file_path,
                "timeframe": rag_data['metadata']['timeframe'],
                "candles": rag_data['metadata']['total_candles'],
                "current_price": str(rag_data['price_analysis']['current_price']),
                "model_used": model,
                "analysis_type": "enhanced_rag_analysis",
                "processed_at": datetime.now().isoformat()
            }
        }

        headers = {"Authorization": f"Bearer {session_token}"}
        knowledge_response = requests.post(f"{api_base}/api/knowledge/add",
                                         json=knowledge_data,
                                         headers=headers)

        if knowledge_response.status_code == 200:
            print("✅ Enhanced RAG analysis added to knowledge base")
        else:
            print(f"⚠️ Knowledge addition failed: {knowledge_response.status_code}")

        # Get detailed trade recommendation
        print(f"💡 Getting detailed trade recommendation using {model}...")

        detailed_prompt = create_detailed_llm_prompt(rag_data, os.path.basename(file_path))

        chat_data = {
            "message": detailed_prompt,
            "model": model,
            "use_rag": True,
            "memory_context": 15  # Increased for more context
        }

        chat_response = requests.post(f"{api_base}/chat",
                                    json=chat_data,
                                    headers=headers)

        if chat_response.status_code == 200:
            result = chat_response.json()
            print("✅ Detailed trade recommendation received:")
            print("-" * 40)
            print(result.get('response', 'No response'))
            print("-" * 40)

            # Show model info
            print(f"📊 Model used: {result.get('model', model)}")
            print(f"🧠 Memory used: {'Yes' if result.get('memory_used') else 'No'}")
            print(f"📈 Data processed: {rag_data['metadata']['total_candles']} candles")

            return True
        else:
            print(f"❌ Trade recommendation failed: {chat_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error processing with API: {e}")
        return False

def save_rag_to_json(rag_data, original_file_path, output_dir="./data/rag_processed", add_timestamp=False):
    """
    Save RAG format data to JSON file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        original_name = os.path.splitext(os.path.basename(original_file_path))[0]

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{original_name}_RAG_{timestamp}.json"
        else:
            output_filename = f"{original_name}.json"

        output_path = os.path.join(output_dir, output_filename)

        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False, default=str)

        file_size = os.path.getsize(output_path)
        print(f"✅ RAG data saved to: {output_path}")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Candles: {rag_data['metadata']['total_candles']:,}")

        return output_path

    except Exception as e:
        print(f"❌ Error saving RAG to JSON: {e}")
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Enhanced CSV analysis with RAG conversion and detailed LLM trading recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process latest file with default model
  python3 scripts/enhanced_latest_file_analyzer.py

  # Use specific file and model, save RAG to JSON (no timestamp)
  python3 scripts/enhanced_latest_file_analyzer.py --file data/XAUUSD_PERIOD_M15_200.csv --model qwen3:14b --save-rag

  # Save with timestamp
  python3 scripts/enhanced_latest_file_analyzer.py --file data/XAUUSD_PERIOD_M15_200.csv --save-rag --timestamp

  # List available files
  python3 scripts/enhanced_latest_file_analyzer.py --list

  # Convert to RAG and save only (no API)
  python3 scripts/enhanced_latest_file_analyzer.py --file data.csv --save-rag --no-api

  # Custom output directory
  python3 scripts/enhanced_latest_file_analyzer.py --file data.csv --save-rag --output-dir ./my_rag_files
        """
    )

    parser.add_argument('--file', '-f', type=str,
                       help='Specific CSV file to analyze (optional: uses latest if not specified)')
    parser.add_argument('--model', '-m', type=str, default='gemma3:1b',
                       help='LLM model to use (default: gemma3:1b)')
    parser.add_argument('--api-url', '-u', type=str, default='http://localhost:8080',
                       help='API base URL (default: http://localhost:8080)')
    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Data directory path (default: ./data)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available files')
    parser.add_argument('--no-api', action='store_true',
                       help='Skip API processing (analyze and convert only)')
    parser.add_argument('--save-rag', '-s', action='store_true',
                       help='Save RAG format to JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='./data/rag_processed',
                       help='Output directory for RAG JSON files (default: ./data/rag_processed)')
    parser.add_argument('--timestamp', '-t', action='store_true',
                       help='Add timestamp to filename (default: no timestamp)')

    args = parser.parse_args()

    print("🚀 Enhanced File Analysis & RAG Processing")
    print("=" * 60)
    print(f"🤖 Model: {args.model}")
    print(f"🔗 API URL: {args.api_url}")
    print(f"📁 Data Directory: {args.data_dir}")

    if args.list:
        print("\n📁 Available CSV Files:")
        list_csv_files(args.data_dir)
        return True

    # Step 1: Determine which file to use
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return False
        target_file = args.file
        print(f"\n📂 Using specified file: {os.path.basename(args.file)}")
    else:
        target_file = find_latest_csv_file(args.data_dir)
        if not target_file:
            print("\n❌ No files found to process")
            return False

    # Step 2: Convert CSV to RAG format
    rag_data = convert_csv_to_rag_json(target_file)

    if not rag_data:
        print("\n❌ RAG conversion failed")
        return False

    # Step 3: Save RAG to JSON (if requested)
    saved_rag_file = None
    if args.save_rag:
        saved_rag_file = save_rag_to_json(rag_data, target_file, args.output_dir, args.timestamp)

    # Step 4: Process with API (unless skipped)
    if args.no_api:
        success = True
        print(f"\n⏭️ API processing skipped (--no-api flag)")
    else:
        success = process_with_enhanced_api(target_file, rag_data, args.api_url, args.model)

    # Summary
    print("\n" + "=" * 60)
    print("📋 ENHANCED PROCESSING SUMMARY")
    print("=" * 60)
    print(f"File: {os.path.basename(target_file)}")
    print(f"Timeframe: {rag_data['metadata']['timeframe']}")
    print(f"Candles Processed: {rag_data['metadata']['total_candles']:,}")
    print(f"Current Price: ${rag_data['price_analysis']['current_price']:.2f}")
    print(f"Price Range: ${rag_data['price_analysis']['price_range']['low']:.2f} - ${rag_data['price_analysis']['price_range']['high']:.2f}")
    print(f"RAG Format: ✅ Complete")
    if saved_rag_file:
        print(f"RAG JSON Saved: ✅ {saved_rag_file}")
    else:
        print(f"RAG JSON Saved: ❌ (use --save-rag to enable)")
    print(f"Model: {args.model}")
    print(f"API Processing: {'✅ Success' if success else '❌ Failed'}")

    if success and not args.no_api:
        print("\n🎉 Enhanced processing completed successfully!")
        print("📊 Full candlestick data converted to RAG and detailed trade analysis obtained.")
        if saved_rag_file:
            print(f"💾 RAG data saved to JSON file for future use.")
    elif success:
        print("\n✅ RAG conversion completed (API processing skipped).")
        if saved_rag_file:
            print(f"💾 RAG data saved to JSON file.")
    else:
        print("\n⚠️ Processing completed with some issues.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)