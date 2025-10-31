#!/usr/bin/env python3
"""
Latest File Analysis Script
Finds and processes CSV files with file selection and model configuration
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

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

def analyze_csv_file(file_path):
    """
    Analyze the CSV file and provide basic information
    """
    print(f"\n📊 Analyzing: {os.path.basename(file_path)}")
    print("=" * 60)

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Basic information
        print(f"📈 Dataset Info:")
        print(f"   Total rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")

        # Detect format
        if all(col in df.columns for col in ['DateTime', 'Open', 'High', 'Low', 'Close']):
            format_type = "RAG Format"
            time_col = 'DateTime'
            price_cols = ['Open', 'High', 'Low', 'Close']
        elif all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
            format_type = "MT5 Format"
            time_col = 'timestamp'
            price_cols = ['open', 'high', 'low', 'close']
        else:
            format_type = "Unknown Format"
            time_col = df.columns[0]
            price_cols = df.columns[1:5] if len(df.columns) >= 5 else df.columns[1:]

        print(f"   Format: {format_type}")

        # Time range
        if time_col in df.columns:
            if time_col == 'DateTime':
                df['timestamp'] = pd.to_datetime(df['DateTime'])
                time_col = 'timestamp'

            first_time = df[time_col].iloc[0]
            last_time = df[time_col].iloc[-1]
            print(f"   Time range: {first_time} → {last_time}")

        # Price range
        if price_cols:
            if 'low' in df.columns:
                low_price = df['low'].min()
                high_price = df['high'].max()
                current_price = df[price_cols[-1]].iloc[-1]  # Last close price
            elif 'Low' in df.columns:
                low_price = df['Low'].min()
                high_price = df['High'].max()
                current_price = df[price_cols[-1]].iloc[-1]
            else:
                low_price = high_price = current_price = "N/A"

            print(f"   Price range: {low_price} - {high_price}")
            print(f"   Current price: {current_price}")

        # Detect timeframe from filename
        filename = os.path.basename(file_path).upper()
        timeframe = "UNKNOWN"
        if 'M1' in filename:
            timeframe = "M1 (1-minute)"
        elif 'M5' in filename:
            timeframe = "M5 (5-minute)"
        elif 'M15' in filename:
            timeframe = "M15 (15-minute)"
        elif 'M30' in filename:
            timeframe = "M30 (30-minute)"
        elif 'H1' in filename:
            timeframe = "H1 (1-hour)"
        elif 'H4' in filename:
            timeframe = "H4 (4-hour)"
        elif 'D1' in filename:
            timeframe = "D1 (Daily)"

        print(f"   Timeframe: {timeframe}")

        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'format': format_type,
            'rows': len(df),
            'timeframe': timeframe,
            'price_range': f"{low_price} - {high_price}",
            'current_price': current_price,
            'columns': list(df.columns)
        }

    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

def process_with_api(file_info, api_base="http://192.168.0.107:8080", model="gemma3:1b"):
    """
    Process the file with the API for RAG integration and analysis
    """
    print(f"\n🔄 Processing with API using model: {model}")
    print("=" * 60)

    if not file_info:
        print("❌ No file info to process")
        return False

    try:
        import requests

        # First, login to get session token
        print("🔐 Authenticating with API...")

        # Add connection debugging and timeout
        try:
            auth_response = requests.post(f"{api_base}/auth/login",
                                        json={"password": "admin123"},
                                        timeout=30,
                                        headers={'Connection': 'close'})
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: {e}")
            print(f"💡 Tip: Try testing with curl: curl -X POST {api_base}/auth/login -H 'Content-Type: application/json' -d '{{\"password\": \"admin123\"}}'")
            return False
        except requests.exceptions.Timeout:
            print(f"❌ Connection timeout to {api_base}")
            return False

        if auth_response.status_code != 200:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            return False

        session_token = auth_response.json().get('session_token')
        if not session_token:
            print("❌ No session token received")
            return False

        print("✅ Authentication successful")

        # Add file analysis to knowledge base
        print("📚 Adding file analysis to knowledge base...")

        knowledge_data = {
            "topic": f"{file_info['timeframe']} Analysis - {file_info['filename']}",
            "content": f"""
File analysis from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

File Details:
- Name: {file_info['filename']}
- Format: {file_info['format']}
- Rows: {file_info['rows']:,}
- Timeframe: {file_info['timeframe']}
- Price Range: {file_info['price_range']}
- Current Price: {file_info['current_price']}

Analysis performed with model: {model}
            """.strip(),
            "metadata": {
                "source": file_info['filename'],
                "file_path": file_info['file_path'],
                "format": file_info['format'],
                "rows": file_info['rows'],
                "timeframe": file_info['timeframe'],
                "price_range": file_info['price_range'],
                "current_price": str(file_info['current_price']),
                "model_used": model,
                "analysis_type": "file_analysis",
                "processed_at": datetime.now().isoformat()
            }
        }

        headers = {"Authorization": f"Bearer {session_token}"}
        knowledge_response = requests.post(f"{api_base}/api/knowledge/add",
                                         json=knowledge_data,
                                         headers=headers)

        if knowledge_response.status_code == 200:
            print("✅ File analysis added to knowledge base")
        else:
            print(f"⚠️ Knowledge addition failed: {knowledge_response.status_code}")

        # Get trade recommendation
        print(f"💡 Getting trade recommendation using {model}...")

        trade_query = f"""
You are an expert trader with deep expertise in technical analysis, market data analysis, and risk management.

Analyze the following {file_info['timeframe']} data from {file_info['filename']}:
- Current Price: {file_info['current_price']}
- Price Range: {file_info['price_range']}
- Data Points: {file_info['rows']:,} candles
- Format: {file_info['format']}

Based on your technical analysis expertise, please provide:
1. **TRADE DIRECTION**: Clear BUY/SELL/WAIT recommendation
2. **ENTRY PRICE**: Optimal entry level with justification
3. **STOP LOSS**: Strategic stop-loss placement with reasoning
4. **TAKE PROFIT**: Realistic profit targets (primary & secondary)
5. **RISK/REWARD**: Risk/reward ratio analysis
6. **TECHNICAL REASONING**: Key technical indicators, patterns, and market structure supporting your decision

Focus on practical, actionable trading advice based on price action, volatility, and market context. Consider the timeframe characteristics and provide realistic, tradable recommendations.
        """.strip()

        chat_data = {
            "message": trade_query,
            "model": model,
            "use_rag": True,
            "memory_context": 10
        }

        chat_response = requests.post(f"{api_base}/chat",
                                    json=chat_data,
                                    headers=headers)

        if chat_response.status_code == 200:
            result = chat_response.json()
            print("✅ Trade recommendation received:")
            print("-" * 40)
            print(result.get('response', 'No response'))
            print("-" * 40)

            # Show model info
            print(f"📊 Model used: {result.get('model', model)}")
            print(f"🧠 Memory used: {'Yes' if result.get('memory_used') else 'No'}")

            return True
        else:
            print(f"❌ Trade recommendation failed: {chat_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error processing with API: {e}")
        return False

def get_available_models(api_base="http://localhost:8080"):
    """Get list of available models from API or local Ollama"""
    # Try to get models from the API server first
    try:
        import requests
        # Try to get models from the API server (might have different models available)
        response = requests.get(f"{api_base}/api/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if isinstance(models, list):
                return [model if isinstance(model, str) else model.get('name', str(model)) for model in models]
    except:
        pass

    # Fallback to local Ollama if API doesn't provide models endpoint
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model['name'] for model in response.json().get('models', [])]
            return models
    except:
        pass

    return ['gemma3:1b', 'gemma2:2b', 'qwen3:0.6b', 'qwen3:14b']  # fallback models

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze CSV files and get trade recommendations using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest file with default model
  python scripts/latest_file_analyzer.py

  # Use latest file with specific model
  python scripts/latest_file_analyzer.py --model gemma2:2b

  # Use specific file
  python scripts/latest_file_analyzer.py --file data/XAUUSD_PERIOD_M15_200.csv

  # Use specific file and model
  python scripts/latest_file_analyzer.py --file data/XAUUSD_PERIOD_M1_200.csv --model qwen3:0.6b

  # List available files and models
  python scripts/latest_file_analyzer.py --list

  # Change API URL
  python scripts/latest_file_analyzer.py --api-url http://localhost:8080
        """
    )

    parser.add_argument('--file', '-f', type=str,
                       help='Specific CSV file to analyze (optional: uses latest if not specified)')
    parser.add_argument('--model', '-m', type=str, default='gemma3:1b',
                       help='LLM model to use (default: gemma3:1b)')
    parser.add_argument('--api-url', '-u', type=str, default='http://192.168.0.107:8080',
                       help='API base URL (default: http://192.168.0.107:8080)')
    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Data directory path (default: ./data)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available files and models')
    parser.add_argument('--no-api', action='store_true',
                       help='Skip API processing (analyze file only)')

    args = parser.parse_args()

    # Get available models
    available_models = get_available_models(args.api_url)

    if args.list:
        print("🚀 File Analysis Tool - Available Options")
        print("=" * 60)

        print("\n📁 Available CSV Files:")
        files_info = list_csv_files(args.data_dir)

        print(f"\n🤖 Available Models:")
        for model in available_models:
            print(f"   - {model}")
            if model == args.model:
                print("     ^ (default selected)")

        print(f"\n🔗 API URL: {args.api_url}")
        return True

    # Validate model (skip validation for remote APIs that might have different models)
    if available_models and args.model not in available_models:
        print(f"⚠️  Model '{args.model}' might not be available. Available models: {available_models}")
        print(f"🚀 Trying '{args.model}' anyway (remote API might have different models)...")
    elif args.model not in available_models:
        print(f"🚀 Trying '{args.model}' (could not verify available models)...")

    print("🚀 File Analysis & Processing")
    print("=" * 60)
    print(f"🤖 Model: {args.model}")
    print(f"🔗 API URL: {args.api_url}")
    print(f"📁 Data Directory: {args.data_dir}")

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

    # Step 2: Analyze the file
    file_info = analyze_csv_file(target_file)

    if not file_info:
        print("\n❌ File analysis failed")
        return False

    # Step 3: Process with API (unless skipped)
    if args.no_api:
        success = True
        print(f"\n⏭️ API processing skipped (--no-api flag)")
    else:
        success = process_with_api(file_info, args.api_url, args.model)

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"File: {file_info['filename']}")
    print(f"Timeframe: {file_info['timeframe']}")
    print(f"Format: {file_info['format']}")
    print(f"Rows: {file_info['rows']:,}")
    print(f"Model: {args.model}")
    print(f"API Processing: {'✅ Success' if success else '❌ Failed'}")

    if success and not args.no_api:
        print("\n🎉 File processed successfully!")
        print("📊 Data added to knowledge base and trade advice obtained.")
    elif success:
        print("\n✅ File analysis completed (API processing skipped).")
    else:
        print("\n⚠️ Processing completed with some issues.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)