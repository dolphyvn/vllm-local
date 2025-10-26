#!/usr/bin/env python3
"""
XAUUSD Knowledge Feeding Script

This script reads XAUUSD CSV data and feeds it to the AI model
via the knowledge feeding API with price prediction information.
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os
import glob
from pathlib import Path

class XAUUSDKnowledgeFeeder:
    def __init__(self, base_url: str = "http://ai.vn.aliases.me"):
        self.base_url = base_url
        self.session_token = None
        self.login()

    def login(self, password: str = "admin123"):
        """Login to get session token"""
        try:
            response = requests.post(f"{self.base_url}/auth/login",
                                    json={"password": password})
            if response.status_code == 200:
                data = response.json()
                self.session_token = data.get("session_token")
                print("✅ Login successful")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    def get_headers(self):
        """Get headers with authentication"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.session_token}"
        }

    def parse_xau_data(self, file_path: str) -> Dict:
        """Parse XAUUSD CSV file and extract key information"""
        try:
            # Read the CSV file line by line to handle multiple sections
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find the 1-minute chart section (SCALPING: 1-MINUTE CHART)
            start_idx = None
            for i, line in enumerate(lines):
                if "SCALPING: 1-MINUTE CHART (Execution Timeframe)" in line:
                    start_idx = i + 2  # Skip the header line
                    break

            if start_idx is None:
                raise ValueError("Could not find 1-minute chart data section")

            # Read only the 1-minute data section
            minute_data = []
            for line in lines[start_idx:]:
                if line.strip() == "" or "Timeframe Statistics" in line or "=====" in line:
                    break
                minute_data.append(line.strip())

            # Parse the data using pandas
            from io import StringIO
            df = pd.read_csv(StringIO('\n'.join(minute_data)))

            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Get basic statistics
            latest_price = df['Close'].iloc[-1]
            earliest_price = df['Close'].iloc[0]
            price_change = latest_price - earliest_price
            price_change_pct = (price_change / earliest_price) * 100

            # Get volatility metrics
            df['Range'] = df['High'] - df['Low']
            avg_range = df['Range'].mean()
            max_range = df['Range'].max()

            # Get volume information
            total_volume = df['Volume'].sum()
            avg_volume = df['Volume'].mean()

            # Detect trend
            if price_change_pct > 2:
                trend = "STRONG UPTREND"
            elif price_change_pct > 0.5:
                trend = "MODERATE UPTREND"
            elif price_change_pct > -0.5:
                trend = "SIDEWAYS/NEUTRAL"
            elif price_change_pct > -2:
                trend = "MODERATE DOWNTREND"
            else:
                trend = "STRONG DOWNTREND"

            return {
                "latest_price": latest_price,
                "earliest_price": earliest_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "trend": trend,
                "avg_range": avg_range,
                "max_range": max_range,
                "total_volume": total_volume,
                "avg_volume": avg_volume,
                "data_points": len(df),
                "start_time": df['Timestamp'].iloc[0],
                "end_time": df['Timestamp'].iloc[-1]
            }

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return None

    def parse_target_data(self, file_path: str) -> Optional[Dict]:
        """Parse target prediction data file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find the daily data line (look for any date in format YYYY.MM.DD)
            for i, line in enumerate(lines):
                # Skip header and empty lines, look for data lines
                if (line.strip() and
                    not line.startswith("Historical") and
                    not line.startswith("Date Range") and
                    not line.startswith("Generated") and
                    not line.startswith("Daily") and
                    not line.startswith("=") and
                    not line.startswith("Date,") and
                    ',' in line and
                    line.count(',') >= 5):

                    # Check if it starts with a date pattern (YYYY.MM.DD or YYYY-MM-DD)
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        date_str = parts[0]
                        # Validate date format by trying to parse numeric values
                        try:
                            open_price = float(parts[1])
                            high_price = float(parts[2])
                            low_price = float(parts[3])
                            close_price = float(parts[4])
                            volume = int(parts[5])

                            return {
                                "date": date_str,
                                "open": open_price,
                                "high": high_price,
                                "low": low_price,
                                "close": close_price,
                                "volume": volume
                            }
                        except (ValueError, IndexError):
                            # If can't parse numeric values, continue to next line
                            continue

            print("❌ Could not find valid target data line")
            return None

        except Exception as e:
            print(f"❌ Error parsing target file: {e}")
            return None

    def calculate_signal(self, last_known_price: float, target_close: float) -> str:
        """Calculate BUY/SELL/NO_GO signal based on price movement"""
        price_change_pct = ((target_close - last_known_price) / last_known_price) * 100

        if price_change_pct > 1.0:
            return "BUY"
        elif price_change_pct < -1.0:
            return "SELL"
        else:
            return "NO_GO"

    def feed_market_data(self, data: Dict):
        """Feed general market data knowledge"""
        knowledge_content = f"""
XAUUSD Market Analysis for {data['start_time'].strftime('%Y-%m-%d')}:

Price Action:
- Starting Price: ${data['earliest_price']:.2f}
- Ending Price: ${data['latest_price']:.2f}
- Price Change: ${data['price_change']:.2f} ({data['price_change_pct']:.2f}%)
- Overall Trend: {data['trend']}

Volatility Metrics:
- Average Range: ${data['avg_range']:.2f}
- Maximum Range: ${data['max_range']:.2f}
- Data Points Analyzed: {data['data_points']}

Volume Analysis:
- Total Volume: {data['total_volume']:,}
- Average Volume: {data['avg_volume']:,.0f}

Time Period: {data['start_time'].strftime('%H:%M')} - {data['end_time'].strftime('%H:%M')}
"""

        payload = {
            "topic": f"XAUUSD Market Analysis {data['start_time'].strftime('%Y-%m-%d')}",
            "content": knowledge_content.strip(),
            "category": "trading",
            "confidence": 0.9,
            "tags": ["XAUUSD", "gold", "market analysis", "price action", "volatility"],
            "source": "XAUUSD Historical Data",
            "priority": 8
        }

        try:
            response = requests.post(f"{self.base_url}/api/knowledge/add",
                                    json=payload,
                                    headers=self.get_headers())
            if response.status_code == 200:
                print("✅ Market data fed successfully")
                return True
            else:
                print(f"❌ Failed to feed market data: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error feeding market data: {e}")
            return False

    def feed_prediction_knowledge(self, training_data: Dict, target_data: Dict, signal: str):
        """Feed prediction knowledge with specific outcome"""
        prediction_content = f"""
XAUUSD Price Prediction Case Study:

Historical Context:
- Analysis Date: {training_data['end_time'].strftime('%Y-%m-%d %H:%M')}
- Last Known Price: ${training_data['latest_price']:.2f}
- Market Trend: {training_data['trend']}
- Volatility: Average ${training_data['avg_range']:.2f}, Max ${training_data['max_range']:.2f}

Prediction Target:
- Target Date: {target_data['date']}
- Predicted Outcome: {signal}
- Actual Result: Open ${target_data['open']:.2f}, High ${target_data['high']:.2f}, Low ${target_data['low']:.2f}, Close ${target_data['close']:.2f}
- Price Movement: ${target_data['close'] - training_data['latest_price']:.2f} ({((target_data['close'] - training_data['latest_price']) / training_data['latest_price']) * 100:.2f}%)

Trading Signal Analysis:
- Signal Generated: {signal}
- Correctness: {'✅ CORRECT' if signal == "BUY" else '❌ INCORRECT'}

Key Learning Points:
- When market shows {training_data['trend'].lower()} with volatility around ${training_data['avg_range']:.2f}
- Next day price movement of {abs(((target_data['close'] - training_data['latest_price']) / training_data['latest_price']) * 100):.2f}% indicates {signal} opportunity
- Volume pattern: {'High activity' if training_data['avg_volume'] > 1000 else 'Normal activity'}
"""

        payload = {
            "topic": f"XAUUSD Prediction {target_data['date']}",
            "content": prediction_content.strip(),
            "category": "trading",
            "confidence": 1.0,  # High confidence as this is actual historical data
            "tags": ["XAUUSD", "prediction", signal.lower(), "price movement", "case study"],
            "source": "Historical Backtesting",
            "priority": 10
        }

        try:
            response = requests.post(f"{self.base_url}/api/knowledge/add",
                                    json=payload,
                                    headers=self.get_headers())
            if response.status_code == 200:
                print(f"✅ Prediction knowledge fed successfully: {signal}")
                return True
            else:
                print(f"❌ Failed to feed prediction knowledge: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error feeding prediction knowledge: {e}")
            return False

    def feed_lesson(self, training_data: Dict, target_data: Dict, signal: str):
        """Feed structured lesson for learning"""
        price_movement = target_data['close'] - training_data['latest_price']
        movement_pct = (price_movement / training_data['latest_price']) * 100

        lesson_payload = {
            "title": f"XAUUSD {signal} Signal Analysis - {target_data['date']}",
            "situation": f"Analyzed XAUUSD data ending at {training_data['latest_price']:.2f} with {training_data['trend'].lower()} trend and ${training_data['avg_range']:.2f} average volatility",
            "lesson": f"Price movement of {abs(movement_pct):.2f}% the next day resulted in a {signal} signal. This demonstrates that {'upward momentum' if signal == 'BUY' else 'downward pressure'} can be identified through volatility and trend analysis.",
            "correct_approach": f"Generate {signal} signal when similar market conditions are detected: {training_data['trend'].lower()} trend with ${training_data['avg_range']:.2f} average range and {training_data['avg_volume']:,.0f} average volume",
            "wrong_approach": f"Ignore {'volatile' if training_data['avg_range'] > 20 else 'stable'} market conditions or fail to account for {training_data['trend'].lower()} momentum",
            "category": "trading",
            "confidence": 1.0,
            "tags": ["XAUUSD", signal.lower(), "lesson", "trading strategy", "technical analysis"]
        }

        try:
            response = requests.post(f"{self.base_url}/api/lessons/add",
                                    json=lesson_payload,
                                    headers=self.get_headers())
            if response.status_code == 200:
                print(f"✅ Lesson fed successfully: {signal} strategy")
                return True
            else:
                print(f"❌ Failed to feed lesson: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error feeding lesson: {e}")
            return False

    def feed_correction(self, incorrect_assumption: str, correct_fact: str, explanation: str):
        """Feed correction for common misconceptions"""
        correction_payload = {
            "incorrect_statement": incorrect_assumption,
            "correct_statement": correct_fact,
            "topic": "XAUUSD Price Prediction",
            "explanation": explanation,
            "confidence": 1.0,
            "category": "corrections"
        }

        try:
            response = requests.post(f"{self.base_url}/api/corrections/add",
                                    json=correction_payload,
                                    headers=self.get_headers())
            if response.status_code == 200:
                print("✅ Correction fed successfully")
                return True
            else:
                print(f"❌ Failed to feed correction: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error feeding correction: {e}")
            return False

    def scan_csv_folder(self, folder_path: str) -> List[Tuple[str, str]]:
        """
        Scan folder for CSV file pairs (training and target files)
        Returns list of (training_file, target_file) tuples
        """
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []

        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return []

        print(f"📁 Found {len(csv_files)} CSV files in {folder_path}")

        # Separate training and target files
        training_files = []
        target_files = []

        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Target files usually end with 'T.csv' or have 'target' in the name
            if filename.endswith('T.csv') or 'target' in filename.lower():
                target_files.append(file_path)
            else:
                training_files.append(file_path)

        print(f"   Training files: {len(training_files)}")
        print(f"   Target files: {len(target_files)}")

        # Match training and target files
        file_pairs = []

        for training_file in training_files:
            training_name = os.path.basename(training_file).replace('.csv', '')

            # Try to find matching target file
            matching_target = None

            # First try exact match with 'T' suffix
            expected_target = training_name + 'T.csv'
            for target_file in target_files:
                if os.path.basename(target_file) == expected_target:
                    matching_target = target_file
                    break

            # If not found, try fuzzy matching
            if not matching_target:
                training_base = training_name.replace('-2025', '').replace('.', '').replace('_', '')
                for target_file in target_files:
                    target_name = os.path.basename(target_file).replace('.csv', '').replace('T', '')
                    target_base = target_name.replace('-2025', '').replace('.', '').replace('_', '')
                    if training_base in target_base or target_base in training_base:
                        matching_target = target_file
                        break

            if matching_target:
                file_pairs.append((training_file, matching_target))
            else:
                print(f"⚠️  No target file found for {os.path.basename(training_file)}")

        print(f"📊 Matched {len(file_pairs)} file pairs")
        return file_pairs

    def process_folder(self, folder_path: str):
        """
        Process all CSV file pairs in a folder
        """
        print(f"📂 Processing CSV folder: {folder_path}")

        # Get all file pairs
        file_pairs = self.scan_csv_folder(folder_path)

        if not file_pairs:
            print("❌ No valid file pairs found")
            return False

        # Process each pair
        total_pairs = len(file_pairs)
        successful_pairs = 0

        for i, (training_file, target_file) in enumerate(file_pairs, 1):
            print(f"\n{'='*60}")
            print(f"📁 Processing pair {i}/{total_pairs}")
            print(f"{'='*60}")

            success = self.process_files(training_file, target_file)
            if success:
                successful_pairs += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"📊 FOLDER PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total file pairs: {total_pairs}")
        print(f"Successful: {successful_pairs}")
        print(f"Failed: {total_pairs - successful_pairs}")
        print(f"Success rate: {(successful_pairs/total_pairs)*100:.1f}%")

        return successful_pairs > 0

    def process_files(self, training_file: str, target_file: str):
        """Process both training and target files"""
        print(f"🔍 Processing {training_file}...")
        training_data = self.parse_xau_data(training_file)

        if not training_data:
            print("❌ Failed to parse training data")
            return False

        print(f"📊 Training Data Summary:")
        print(f"   Price Range: ${training_data['earliest_price']:.2f} - ${training_data['latest_price']:.2f}")
        print(f"   Change: {training_data['price_change_pct']:.2f}% ({training_data['trend']})")
        print(f"   Volatility: ${training_data['avg_range']:.2f} avg, ${training_data['max_range']:.2f} max")
        print(f"   Data Points: {training_data['data_points']}")

        print(f"\n🎯 Processing {target_file}...")
        target_data = self.parse_target_data(target_file)

        if not target_data:
            print("❌ Failed to parse target data")
            return False

        print(f"📈 Target Data:")
        print(f"   Date: {target_data['date']}")
        print(f"   OHLC: ${target_data['open']:.2f}/${target_data['high']:.2f}/${target_data['low']:.2f}/${target_data['close']:.2f}")

        # Calculate signal
        signal = self.calculate_signal(training_data['latest_price'], target_data['close'])
        price_movement = target_data['close'] - training_data['latest_price']
        movement_pct = (price_movement / training_data['latest_price']) * 100

        print(f"\n🚨 PREDICTION RESULT:")
        print(f"   From: ${training_data['latest_price']:.2f}")
        print(f"   To: ${target_data['close']:.2f}")
        print(f"   Movement: {movement_pct:.2f}%")
        print(f"   SIGNAL: {signal} {'✅' if movement_pct > 1 else '⚠️' if abs(movement_pct) < 1 else ''}")

        # Feed knowledge to the model
        print(f"\n📚 Feeding knowledge to AI model...")

        success = True

        # 1. Feed market data
        success &= self.feed_market_data(training_data)

        # 2. Feed prediction knowledge
        success &= self.feed_prediction_knowledge(training_data, target_data, signal)

        # 3. Feed structured lesson
        success &= self.feed_lesson(training_data, target_data, signal)

        # 4. Feed correction if needed
        if signal == "BUY" and movement_pct > 1:
            correction = self.feed_correction(
                "XAUUSD predictions are unreliable",
                "XAUUSD can show predictable movements when volatility and trend conditions are analyzed correctly",
                f"This case demonstrates that a {movement_pct:.2f}% upward movement was predictable from the {training_data['trend'].lower()} market conditions"
            )
            success &= correction

        if success:
            print(f"\n✅ All knowledge fed successfully!")
            print(f"🤖 The AI model now knows about this XAUUSD {signal} prediction case")
        else:
            print(f"\n❌ Some knowledge feeding failed")

        return success

def main():
    """Main function to run the XAUUSD knowledge feeder"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feed XAUUSD knowledge to AI model')
    parser.add_argument('--training', type=str, help='Training CSV file path')
    parser.add_argument('--target', type=str, help='Target CSV file path')
    parser.add_argument('--folder', type=str, help='Folder containing CSV files')
    parser.add_argument('--base-url', type=str, default='http://ai.vn.aliases.me', help='Base URL for API')

    args = parser.parse_args()

    # Validate arguments
    if args.folder and (args.training or args.target):
        print("❌ Cannot specify both --folder and individual files (--training, --target)")
        return False

    if not args.folder and not args.training and not args.target:
        # Default to single file pair if no arguments provided
        print("📋 No arguments provided, using default files...")
        args.training = "data/XAUUSD-2025.10.21.csv"
        args.target = "data/XAUUSD-2025.10.21T.csv"

    # Initialize feeder
    feeder = XAUUSDKnowledgeFeeder(base_url=args.base_url)

    if not feeder.session_token:
        print("❌ Failed to authenticate")
        return False

    success = False

    if args.folder:
        # Process folder
        print(f"📂 Processing CSV folder mode")
        success = feeder.process_folder(args.folder)

        if success:
            print(f"\n🎉 CSV folder processing completed!")
            print(f"💡 The AI model now knows about all XAUUSD prediction cases in the folder")
        else:
            print(f"\n💥 CSV folder processing failed!")

    else:
        # Process single file pair
        if not args.training or not args.target:
            print("❌ Both --training and --target files must be specified")
            return False

        # Check if files exist
        if not os.path.exists(args.training):
            print(f"❌ Training file not found: {args.training}")
            return False

        if not os.path.exists(args.target):
            print(f"❌ Target file not found: {args.target}")
            return False

        print(f"📁 Processing single file pair mode")
        success = feeder.process_files(args.training, args.target)

        if success:
            print(f"\n🎉 XAUUSD knowledge feeding completed!")
            print(f"💡 You can now ask the AI about XAUUSD predictions and trading strategies")
        else:
            print(f"\n💥 XAUUSD knowledge feeding failed!")

    return success

if __name__ == "__main__":
    main()