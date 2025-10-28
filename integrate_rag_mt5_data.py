#!/usr/bin/env python3
"""
RAG MT5 Data Integration Script
Bridges MT5 RAG trading system with Financial Assistant knowledge feeding
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGMT5Integrator:
    def __init__(self, base_url: str = "http://localhost:8080", password: str = "admin123",
                 export_path: str = "./data", log_file: str = "./processed_rag_files.log"):
        self.base_url = base_url
        self.password = password
        self.token = None
        self.session = requests.Session()

        # MT5 export directories (configurable)
        self.mt5_export_path = export_path
        self.processed_files_log = log_file

        # Validate export path exists
        if not os.path.exists(export_path):
            logger.warning(f"Export path does not exist: {export_path}")
            logger.info(f"Creating directory: {export_path}")
            os.makedirs(export_path, exist_ok=True)

        # Login and get token
        self.authenticate()

    def authenticate(self) -> bool:
        """Authenticate with the Financial Assistant API"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"password": self.password}
            )

            if response.status_code == 200:
                self.token = response.json().get("session_token")
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                logger.info("✅ Authentication successful")
                return True
            else:
                logger.error(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False

    def load_processed_files(self) -> set:
        """Load list of already processed files"""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def mark_file_processed(self, filename: str):
        """Mark a file as processed"""
        with open(self.processed_files_log, 'a') as f:
            f.write(f"{filename}\n")

    def find_rag_files(self) -> List[str]:
        """Find RAG MT5 export files"""
        if not os.path.exists(self.mt5_export_path):
            logger.warning(f"MT5 export directory not found: {self.mt5_export_path}")
            return []

        processed = self.load_processed_files()
        rag_files = []

        # Look for RAG training data files
        for file in os.listdir(self.mt5_export_path):
            if file.startswith("RAG_") and file.endswith(".csv") and file not in processed:
                rag_files.append(os.path.join(self.mt5_export_path, file))

        return rag_files

    def parse_rag_csv(self, filepath: str) -> List[Dict]:
        """Parse RAG MT5 CSV file and convert to knowledge entries"""
        try:
            # Try different CSV parsing approaches to handle formatting issues
            logger.info(f"📖 Attempting to parse {os.path.basename(filepath)}...")

            # First try standard parsing
            try:
                df = pd.read_csv(filepath)
                logger.info(f"✅ Standard parsing successful: {len(df)} rows")
            except Exception as e:
                logger.warning(f"⚠️ Standard parsing failed: {e}")

                # Try with different encodings
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None

                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"✅ {encoding} parsing successful: {len(df)} rows")
                        break
                    except Exception as enc_e:
                        logger.warning(f"⚠️ {encoding} parsing failed: {enc_e}")
                        continue

                if df is None:
                    # Try with different separators and robust parsing
                    try:
                        df = pd.read_csv(filepath,
                                       sep=None,
                                       engine='python',
                                       skipinitialspace=True,
                                       quoting=1,  # QUOTE_ALL
                                       on_bad_lines='warn')
                        logger.info(f"✅ Robust parsing successful: {len(df)} rows")
                    except Exception as robust_e:
                        logger.error(f"❌ All parsing methods failed for {filepath}")
                        logger.error(f"Last error: {robust_e}")

                        # Try to examine the file structure
                        self.diagnose_csv_file(filepath)
                        return []

            if len(df) == 0:
                logger.warning(f"⚠️ No data rows found in {filepath}")
                return []

            logger.info(f"📊 Successfully loaded {len(df)} rows from {os.path.basename(filepath)}")

            # Check if required columns exist
            required_columns = ['Timestamp', 'RSI', 'MACD', 'Trend', 'Pattern', 'Confidence', 'Session']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"⚠️ Missing columns in {filepath}: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")

                # Try to work with available columns
                available_columns = df.columns.tolist()
                logger.info(f"📋 Working with available columns: {available_columns}")

            knowledge_entries = []
            symbol = "UNKNOWN"

            # Extract symbol from filename
            filename = os.path.basename(filepath)
            if "XAUUSD" in filename:
                symbol = "XAUUSD"
            elif "EURUSD" in filename:
                symbol = "EURUSD"
            elif "BTCUSD" in filename:
                symbol = "BTCUSD"
            else:
                # Try to extract from first row if symbol column exists
                if 'Symbol' in df.columns and len(df) > 0:
                    symbol = str(df.iloc[0]['Symbol']).upper()

            successful_rows = 0
            failed_rows = 0

            for index, row in df.iterrows():
                try:
                    # Extract key metrics with fallbacks
                    timestamp = str(row.get('Timestamp', f'Row_{index}'))
                    rsi = self.safe_float(row.get('RSI', 50))
                    macd = self.safe_float(row.get('MACD', 0))
                    trend = str(row.get('Trend', 'UNKNOWN')).upper()
                    pattern = str(row.get('Pattern', 'UNKNOWN')).upper()
                    confidence = self.safe_float(row.get('Confidence', 50))
                    session = str(row.get('Session', 'UNKNOWN')).upper()

                    # Skip rows with critical missing data
                    if timestamp == 'nan' or timestamp == 'Row_0':
                        failed_rows += 1
                        continue

                    # Create comprehensive knowledge entry
                    content = f"""
Trading Analysis for {symbol} at {timestamp}:

Technical Indicators:
- RSI: {rsi:.2f} (Overbought/Oversold: {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
- MACD: {macd:.6f}
- Trend: {trend}
- Pattern: {pattern}
- Session: {session}
- Confidence: {confidence:.1f}%

Market Context:
This analysis represents a {trend.lower()} market condition during {session.lower()} trading session.
The {pattern} pattern suggests {self.get_pattern_interpretation(pattern)}.

Data Source: {filename}
"""

                    knowledge_entry = {
                        "topic": f"{symbol} Technical Analysis {timestamp}",
                        "content": content.strip(),
                        "category": "trading_analysis",
                        "confidence": min(confidence / 100, 1.0),
                        "tags": [symbol.lower(), "technical", "rsi", "macd", trend.lower(), pattern.lower(), session.lower()],
                        "source": f"RAG_MT5_{filename}",
                        "priority": 7 if confidence > 70 else 5
                    }

                    knowledge_entries.append(knowledge_entry)
                    successful_rows += 1

                except Exception as row_error:
                    failed_rows += 1
                    if failed_rows <= 5:  # Only log first 5 errors to avoid spam
                        logger.warning(f"⚠️ Error processing row {index}: {row_error}")

            logger.info(f"📈 Processing summary for {filename}: {successful_rows} successful, {failed_rows} failed")

            return knowledge_entries

        except Exception as e:
            logger.error(f"❌ Critical error parsing {filepath}: {e}")
            return []

    def safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value == '' or value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def diagnose_csv_file(self, filepath: str):
        """Diagnose CSV file structure issues"""
        try:
            logger.info(f"🔍 Diagnosing CSV file: {filepath}")

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines
                lines = []
                for i, line in enumerate(f):
                    if i < 20:  # Read first 20 lines
                        lines.append(line.strip())
                    else:
                        break

            logger.info(f"📄 First 20 lines of {os.path.basename(filepath)}:")
            for i, line in enumerate(lines):
                logger.info(f"  Line {i+1:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
                logger.info(f"          Fields: {len(line.split(','))}")

            # Check file encoding
            try:
                with open(filepath, 'rb') as f:
                    raw_data = f.read(1000)
                logger.info(f"🔤 File encoding sample: {raw_data[:100]}")
            except Exception as e:
                logger.warning(f"⚠️ Could not read raw data: {e}")

        except Exception as e:
            logger.error(f"❌ Error diagnosing file: {e}")

    def get_pattern_interpretation(self, pattern: str) -> str:
        """Get interpretation of candlestick pattern"""
        interpretations = {
            "STANDARD_CANDLE": "normal market activity",
            "DOJI": "indecision in the market",
            "HAMMER": "potential bullish reversal",
            "SHOOTING_STAR": "potential bearish reversal",
            "ENGULFING_BULLISH": "strong bullish reversal signal",
            "ENGULFING_BEARISH": "strong bearish reversal signal"
        }
        return interpretations.get(pattern, "specific market condition")

    def feed_knowledge_bulk(self, entries: List[Dict]) -> bool:
        """Feed knowledge entries in bulk"""
        try:
            # Split into chunks of 10 to avoid request size limits
            chunk_size = 10
            success_count = 0

            for i in range(0, len(entries), chunk_size):
                chunk = entries[i:i + chunk_size]

                response = self.session.post(
                    f"{self.base_url}/api/knowledge/bulk",
                    json={"knowledge_entries": chunk}
                )

                if response.status_code == 200:
                    result = response.json()
                    success_count += result.get('success_count', 0)
                    logger.info(f"✅ Chunk {i//chunk_size + 1}: {result.get('success_count', 0)}/{len(chunk)} entries added")
                else:
                    logger.error(f"❌ Chunk {i//chunk_size + 1} failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")

            logger.info(f"📈 Total knowledge entries added: {success_count}/{len(entries)}")
            return success_count > 0

        except Exception as e:
            logger.error(f"❌ Error feeding knowledge: {e}")
            return False

    def create_trading_lesson(self, entries: List[Dict], symbol: str) -> bool:
        """Create a structured trading lesson from the data"""
        try:
            # Analyze patterns in the data
            bullish_patterns = sum(1 for e in entries if 'bullish' in str(e.get('tags', [])))
            bearish_patterns = sum(1 for e in entries if 'bearish' in str(e.get('tags', [])))
            total_entries = len(entries)

            lesson_content = f"""
# {symbol} Trading Analysis Lesson

## Overview
This lesson is based on analysis of {total_entries} data points from RAG MT5 system.

## Pattern Analysis
- Bullish patterns detected: {bullish_patterns} ({bullish_patterns/total_entries*100:.1f}%)
- Bearish patterns detected: {bearish_patterns} ({bearish_patterns/total_entries*100:.1f}%)
- Neutral/Other patterns: {total_entries - bullish_patterns - bearish_patterns}

## Key Insights
1. **Market Behavior**: Analysis shows various market conditions across different trading sessions
2. **Technical Indicators**: RSI and MACD provide valuable insights into market momentum
3. **Pattern Recognition**: Candlestick patterns help identify potential reversals

## Trading Strategy Considerations
- Use RSI levels to identify overbought/oversold conditions
- Combine MACD signals with pattern recognition for confirmation
- Consider trading session context for better timing

## Risk Management
- Always confirm signals with multiple indicators
- Be aware of session-specific market behaviors
- Monitor confidence levels in analysis

*This lesson was automatically generated from RAG MT5 trading system data.*
"""

            lesson_data = {
                "title": f"{symbol} Technical Analysis Lesson",
                "content": lesson_content.strip(),
                "category": "trading_education",
                "difficulty": "intermediate",
                "prerequisites": ["Basic understanding of technical analysis", "RSI and MACD knowledge"],
                "learning_objectives": [
                    f"Understand {symbol} market patterns",
                    "Apply technical indicators effectively",
                    "Recognize trading opportunities"
                ],
                "tags": [symbol.lower(), "technical_analysis", "trading_education"],
                "source": "RAG_MT5_Auto_Lesson"
            }

            response = self.session.post(
                f"{self.base_url}/api/lessons/add",
                json=lesson_data
            )

            if response.status_code == 200:
                logger.info(f"✅ Trading lesson for {symbol} created successfully")
                return True
            else:
                logger.error(f"❌ Failed to create lesson: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error creating lesson: {e}")
            return False

    def process_all_files(self) -> Dict:
        """Process all RAG MT5 files and feed to knowledge base"""
        logger.info("🚀 Starting RAG MT5 data integration...")

        # Find RAG files
        rag_files = self.find_rag_files()

        if not rag_files:
            logger.info("ℹ️ No new RAG files to process")
            return {"processed": 0, "total_entries": 0, "files": []}

        results = {"processed": 0, "total_entries": 0, "files": []}

        for filepath in rag_files:
            logger.info(f"📁 Processing {os.path.basename(filepath)}...")

            # Parse CSV file
            knowledge_entries = self.parse_rag_csv(filepath)

            if not knowledge_entries:
                logger.warning(f"⚠️ No entries extracted from {filepath}")
                continue

            # Feed to knowledge base
            if self.feed_knowledge_bulk(knowledge_entries):
                # Create lesson if we have enough data
                symbol = knowledge_entries[0].get('tags', ['unknown'])[0].upper()
                if len(knowledge_entries) >= 5:  # Only create lesson with sufficient data
                    self.create_trading_lesson(knowledge_entries, symbol)

                # Mark as processed
                self.mark_file_processed(os.path.basename(filepath))
                results["processed"] += 1
                results["total_entries"] += len(knowledge_entries)
                results["files"].append(os.path.basename(filepath))

                logger.info(f"✅ Successfully processed {os.path.basename(filepath)}")
            else:
                logger.error(f"❌ Failed to process {filepath}")

        return results

    def monitor_and_process(self, interval_minutes: int = 5):
        """Continuously monitor for new RAG files and process them"""
        logger.info(f"🔄 Starting continuous monitoring (every {interval_minutes} minutes)...")

        while True:
            try:
                results = self.process_all_files()

                if results["processed"] > 0:
                    logger.info(f"📊 Summary: {results['processed']} files, {results['total_entries']} entries added")
                    logger.info(f"📁 Files: {', '.join(results['files'])}")

                # Wait before next check
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def validate_directory(path: str) -> str:
    """Validate and normalize directory path"""
    if not os.path.exists(path):
        logger.warning(f"Directory does not exist: {path}")
        response = input(f"Create directory '{path}'? (y/n): ").lower().strip()
        if response == 'y':
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ Created directory: {path}")
        else:
            logger.error("❌ Directory required. Exiting.")
            sys.exit(1)

    if not os.path.isdir(path):
        logger.error(f"❌ Path is not a directory: {path}")
        sys.exit(1)

    # Get absolute path
    return os.path.abspath(path)

def list_csv_files(directory: str) -> List[str]:
    """List all CSV files in directory"""
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            csv_files.append(file)
    return csv_files

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG MT5 Data Integration - Feed trading data to Financial Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default ./data directory
  python3 integrate_rag_mt5_data.py

  # Specify custom directory
  python3 integrate_rag_mt5_data.py --export-path /path/to/mt5/exports

  # Monitor custom directory
  python3 integrate_rag_mt5_data.py --mode monitor --export-path /home/user/trading/data

  # Use remote server
  python3 integrate_rag_mt5_data.py --base-url http://ai.vn.aliases.me --export-path ./trading_data
        """
    )

    parser.add_argument("--mode", choices=["once", "monitor"], default="once",
                       help="Run once or continuously monitor (default: once)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Monitoring interval in minutes for monitor mode (default: 5)")
    parser.add_argument("--base-url", default="http://localhost:8080",
                       help="Base URL for Financial Assistant API (default: http://localhost:8080)")
    parser.add_argument("--export-path",
                       help="Path to MT5 RAG export files (default: ./data)")
    parser.add_argument("--log-file", default="./processed_rag_files.log",
                       help="Log file for tracking processed files (default: ./processed_rag_files.log)")
    parser.add_argument("--list-files", action="store_true",
                       help="List available CSV files in directory and exit")
    parser.add_argument("--password", default="admin123",
                       help="API password (default: admin123)")

    args = parser.parse_args()

    # Handle directory input
    if args.export_path:
        export_path = validate_directory(args.export_path)
    else:
        # Interactive mode if no directory specified
        print("🔍 RAG MT5 Data Integration")
        print("=" * 40)

        default_path = "./data"
        user_input = input(f"Enter directory path for RAG CSV files [{default_path}]: ").strip()

        if not user_input:
            export_path = default_path
        else:
            export_path = user_input

        export_path = validate_directory(export_path)

    # List files if requested
    if args.list_files:
        csv_files = list_csv_files(export_path)
        print(f"\n📁 CSV files in '{export_path}':")
        if csv_files:
            for i, file in enumerate(csv_files, 1):
                file_path = os.path.join(export_path, file)
                file_size = os.path.getsize(file_path)
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
                print(f"  {i:2d}. {file:<30} ({file_size:,} bytes, {file_date})")
        else:
            print("  No CSV files found.")
        return

    # Show directory info
    csv_files = list_csv_files(export_path)
    rag_files = [f for f in csv_files if f.startswith('RAG_')]

    print(f"\n📊 Directory: {export_path}")
    print(f"   Total CSV files: {len(csv_files)}")
    print(f"   RAG files found: {len(rag_files)}")

    if rag_files:
        print(f"   Latest RAG file: {max(rag_files)}")
    else:
        print("   ⚠️  No RAG_*.csv files found")

    # Create integrator with specified parameters
    try:
        integrator = RAGMT5Integrator(
            base_url=args.base_url,
            password=args.password,
            export_path=export_path,
            log_file=args.log_file
        )

        print(f"   API Server: {args.base_url}")
        print(f"   Log file: {args.log_file}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize integrator: {e}")
        sys.exit(1)

    if args.mode == "once":
        # Process once and exit
        print(f"\n🚀 Processing RAG files...")
        results = integrator.process_all_files()

        print(f"\n📈 Results:")
        print(f"   Files processed: {results['processed']}")
        print(f"   Knowledge entries: {results['total_entries']}")
        if results['files']:
            print(f"   Files: {', '.join(results['files'])}")

        logger.info(f"🏁 Processing complete: {results}")
    else:
        # Continuous monitoring
        print(f"\n🔄 Starting continuous monitoring (every {args.interval} minutes)...")
        print("   Press Ctrl+C to stop")
        integrator.monitor_and_process(args.interval)

if __name__ == "__main__":
    main()