#!/usr/bin/env python3

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from scripts.technical_analysis_engine import AdvancedTechnicalAnalyzer
from scripts.comprehensive_feature_analyzer import ComprehensiveFeatureAnalyzer

def test_single_timeframe():
    """Test processing a single timeframe"""
    symbol = "XAUUSD"
    csv_file = f"./data/{symbol}_PERIOD_M30_0.csv"

    print(f"🔄 Testing single timeframe: M30")

    try:
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin1')

        print(f"✅ Successfully read {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")

        # Process datetime
        if 'DateTime' in df.columns:
            df['datetime'] = pd.to_datetime(df['DateTime'])
        else:
            print("❌ No DateTime column found")
            return

        # Map column names
        column_mapping = {
            'open': ['Open'],
            'high': ['High'],
            'low': ['Low'],
            'close': ['Close'],
            'volume': ['Volume']
        }

        for standard_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns:
                    df[standard_col] = df[col]
                    break

        # Validate columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return

        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Take only last 200 rows for testing
        df = df.tail(200).copy()
        print(f"📊 Using last 200 rows")

        # Test basic analysis
        analyzer = ComprehensiveFeatureAnalyzer()

        print("🔍 Testing price features...")
        price_features = analyzer.calculate_price_features(df)
        print(f"✅ Price features: {len(price_features)} categories")

        print("🔍 Testing intraday session analysis...")
        intraday_features = analyzer.calculate_intraday_session_analysis(df)
        print(f"✅ Intraday features: {len(intraday_features)} categories")

        print("🔍 Testing technical analysis engine...")
        tech_analyzer = AdvancedTechnicalAnalyzer()

        print("🔍 Testing gap analysis...")
        gap_analysis = tech_analyzer.detect_gaps(df)
        print(f"✅ Gap analysis: {gap_analysis.get('total_gaps', 0)} gaps found")

        print("🔍 Testing order blocks...")
        order_blocks = tech_analyzer.detect_order_blocks(df)
        print(f"✅ Order blocks: {len(order_blocks.get('order_blocks', []))} blocks found")

        print("🔍 Testing Fibonacci...")
        fibonacci = tech_analyzer.calculate_fibonacci_retracements(df)
        print(f"✅ Fibonacci: {len(fibonacci)} levels calculated")

        print("🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_timeframe()
    if success:
        print("\n✅ Single timeframe test successful!")
        print("🚀 You can now run the full comprehensive analyzer")
    else:
        print("\n❌ Single timeframe test failed")