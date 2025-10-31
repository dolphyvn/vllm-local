#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from comprehensive_feature_analyzer import ComprehensiveFeatureAnalyzer

def quick_test():
    """Quick test with one timeframe"""
    print("🧪 Quick test: Comprehensive Feature Analyzer")

    analyzer = ComprehensiveFeatureAnalyzer()

    # Test with just M15 timeframe
    csv_file = "./data/XAUUSD_PERIOD_M15_0.csv"

    try:
        # Test data reading and processing
        import pandas as pd
        try:
            # Read with different encodings and skip BOM if present
            with open(csv_file, 'rb') as f:
                content = f.read()
                if content.startswith(b'\xef\xbb\xbf'):
                    content = content[3:]  # Remove UTF-8 BOM

            from io import StringIO
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        except Exception as e:
            print(f"⚠️  Encoding issue: {e}")
            try:
                df = pd.read_csv(csv_file, encoding='latin1')
            except:
                df = pd.read_csv(csv_file)
        print(f"✅ Successfully read {len(df)} rows")

        # Map columns to expected names
        if 'DateTime' in df.columns:
            df['datetime'] = pd.to_datetime(df['DateTime'])

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

        # Take last 1000 rows for testing
        df = df.tail(1000).copy()

        # Debug: Print available columns
        print(f"📋 Available columns after mapping: {list(df.columns)}")

        # Test a few feature calculations
        print("🔍 Testing price features...")
        price_features = analyzer.calculate_price_features(df)
        print(f"✅ Price features calculated: {len(price_features)} categories")

        print("🔍 Testing volatility indicators...")
        vol_features = analyzer.calculate_volatility_indicators(df)
        print(f"✅ Volatility features calculated: {len(vol_features)} categories")

        print("🔍 Testing time analysis...")
        time_features = analyzer.calculate_time_analysis(df)
        print(f"✅ Time features calculated: {len(time_features)} categories")

        print("🎉 Quick test successful!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ All fixes working correctly!")
        print("🚀 Ready to run full analysis")
    else:
        print("\n❌ Some issues remain")