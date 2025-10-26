#!/usr/bin/env python3
"""
XAUUSD Analysis Demonstration

This script demonstrates the XAUUSD data analysis and shows exactly
what knowledge would be fed to the AI model for learning.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import os

class XAUUSDAnalyzer:
    def __init__(self):
        self.training_file = "data/XAUUSD-2025.10.21.csv"
        self.target_file = "data/XAUUSD-2025.10.21T.csv"

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
                "end_time": df['Timestamp'].iloc[-1],
                "df": df  # Keep dataframe for detailed analysis
            }

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return None

    def parse_target_data(self, file_path: str) -> Optional[Dict]:
        """Parse target prediction data file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find the daily data line
            for i, line in enumerate(lines):
                if line.startswith("2025.10.23,"):
                    data_parts = line.strip().split(',')
                    return {
                        "date": "2025.10.23",
                        "open": float(data_parts[1]),
                        "high": float(data_parts[2]),
                        "low": float(data_parts[3]),
                        "close": float(data_parts[4]),
                        "volume": int(data_parts[5])
                    }

            print("❌ Could not find target data line")
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

    def generate_knowledge_content(self, training_data: Dict, target_data: Dict, signal: str):
        """Generate the content that would be fed to the AI model"""

        # Market Data Knowledge
        market_knowledge = f"""
XAUUSD Market Analysis for {training_data['start_time'].strftime('%Y-%m-%d')}:

Price Action:
- Starting Price: ${training_data['earliest_price']:.2f}
- Ending Price: ${training_data['latest_price']:.2f}
- Price Change: ${training_data['price_change']:.2f} ({training_data['price_change_pct']:.2f}%)
- Overall Trend: {training_data['trend']}

Volatility Metrics:
- Average Range: ${training_data['avg_range']:.2f}
- Maximum Range: ${training_data['max_range']:.2f}
- Data Points Analyzed: {training_data['data_points']}

Volume Analysis:
- Total Volume: {training_data['total_volume']:,}
- Average Volume: {training_data['avg_volume']:,.0f}

Time Period: {training_data['start_time'].strftime('%H:%M')} - {training_data['end_time'].strftime('%H:%M')}
"""

        # Prediction Knowledge
        prediction_knowledge = f"""
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

        # Structured Lesson
        lesson_content = f"""
Title: XAUUSD {signal} Signal Analysis - {target_data['date']}

Situation: Analyzed XAUUSD data ending at {training_data['latest_price']:.2f} with {training_data['trend'].lower()} trend and ${training_data['avg_range']:.2f} average volatility

Lesson: Price movement of {abs(((target_data['close'] - training_data['latest_price']) / training_data['latest_price']) * 100):.2f}% the next day resulted in a {signal} signal. This demonstrates that {'upward momentum' if signal == 'BUY' else 'downward pressure'} can be identified through volatility and trend analysis.

Correct Approach: Generate {signal} signal when similar market conditions are detected: {training_data['trend'].lower()} trend with ${training_data['avg_range']:.2f} average range and {training_data['avg_volume']:,.0f} average volume

Wrong Approach: Ignore {'volatile' if training_data['avg_range'] > 20 else 'stable'} market conditions or fail to account for {training_data['trend'].lower()} momentum
"""

        return {
            "market_knowledge": market_knowledge.strip(),
            "prediction_knowledge": prediction_knowledge.strip(),
            "lesson_content": lesson_content.strip()
        }

    def analyze_and_display(self):
        """Run complete analysis and display results"""
        print("🔍 XAUUSD KNOWLEDGE FEEDING DEMONSTRATION")
        print("=" * 60)

        # Check if files exist
        if not os.path.exists(self.training_file):
            print(f"❌ Training file not found: {self.training_file}")
            return False

        if not os.path.exists(self.target_file):
            print(f"❌ Target file not found: {self.target_file}")
            return False

        print(f"📁 Processing files:")
        print(f"   Training: {self.training_file}")
        print(f"   Target: {self.target_file}")
        print()

        # Parse training data
        print("🔍 Analyzing training data...")
        training_data = self.parse_xau_data(self.training_file)

        if not training_data:
            print("❌ Failed to parse training data")
            return False

        print(f"📊 Training Data Summary:")
        print(f"   Price Range: ${training_data['earliest_price']:.2f} → ${training_data['latest_price']:.2f}")
        print(f"   Change: {training_data['price_change_pct']:.2f}% ({training_data['trend']})")
        print(f"   Volatility: ${training_data['avg_range']:.2f} avg, ${training_data['max_range']:.2f} max")
        print(f"   Volume: {training_data['total_volume']:,} total, {training_data['avg_volume']:,.0f} avg")
        print(f"   Data Points: {training_data['data_points']}")
        print(f"   Time Period: {training_data['start_time'].strftime('%H:%M')} - {training_data['end_time'].strftime('%H:%M')}")
        print()

        # Parse target data
        print("🎯 Analyzing target data...")
        target_data = self.parse_target_data(self.target_file)

        if not target_data:
            print("❌ Failed to parse target data")
            return False

        print(f"📈 Target Data:")
        print(f"   Date: {target_data['date']}")
        print(f"   OHLC: ${target_data['open']:.2f}/${target_data['high']:.2f}/${target_data['low']:.2f}/${target_data['close']:.2f}")
        print()

        # Calculate signal
        signal = self.calculate_signal(training_data['latest_price'], target_data['close'])
        price_movement = target_data['close'] - training_data['latest_price']
        movement_pct = (price_movement / training_data['latest_price']) * 100

        print("🚨 PREDICTION ANALYSIS:")
        print("=" * 30)
        print(f"   From Price: ${training_data['latest_price']:.2f}")
        print(f"   To Price:   ${target_data['close']:.2f}")
        print(f"   Movement:   {movement_pct:+.2f}%")
        print(f"   SIGNAL:     {signal} {'✅' if movement_pct > 1 else '⚠️' if abs(movement_pct) < 1 else ''}")
        print()

        # Generate knowledge content
        print("📚 KNOWLEDGE CONTENT TO BE FED:")
        print("=" * 40)

        knowledge_content = self.generate_knowledge_content(training_data, target_data, signal)

        print("📋 1. MARKET DATA KNOWLEDGE:")
        print("-" * 35)
        print(knowledge_content["market_knowledge"])
        print()

        print("🎯 2. PREDICTION KNOWLEDGE:")
        print("-" * 35)
        print(knowledge_content["prediction_knowledge"])
        print()

        print("📖 3. STRUCTURED LESSON:")
        print("-" * 35)
        print(knowledge_content["lesson_content"])
        print()

        print("🏷️ TAGS FOR SEARCHABILITY:")
        print("   XAUUSD, gold, trading, prediction, " + signal.lower() +
              ", price action, volatility, market analysis, technical analysis")
        print()

        print("📊 API PAYLOAD EXAMPLE:")
        print("-" * 25)
        print("POST /api/knowledge/add")
        print("Content-Type: application/json")
        print("Authorization: Bearer <session_token>")
        print()
        print("{")
        print('  "topic": "XAUUSD Prediction 2025.10.23",')
        print('  "content": "' + knowledge_content["prediction_knowledge"][:100] + '...",')
        print('  "category": "trading",')
        print('  "confidence": 1.0,')
        print('  "tags": ["XAUUSD", "prediction", "' + signal.lower() + '"],')
        print('  "source": "Historical Backtesting",')
        print('  "priority": 10')
        print("}")
        print()

        print("✅ DEMONSTRATION COMPLETE!")
        print("🤖 The AI model would learn this XAUUSD prediction pattern")
        print("   and be able to apply it to future market analysis.")

        return True

def main():
    """Main function to run the demonstration"""
    analyzer = XAUUSDAnalyzer()
    success = analyzer.analyze_and_display()

    if success:
        print()
        print("💡 NEXT STEPS:")
        print("1. Deploy the updated API endpoints to your remote server")
        print("2. Run the feed_xau_data.py script to actually feed this knowledge")
        print("3. Test the AI's understanding by asking about XAUUSD predictions")
        print("4. Add more historical data to improve the model's learning")
    else:
        print("❌ Demonstration failed!")

if __name__ == "__main__":
    main()