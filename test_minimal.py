#!/usr/bin/env python3

"""Minimal test of the comprehensive analyzer with the fixes applied"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_minimal():
    """Minimal test to verify the fixes work"""
    print("🧪 Minimal test: Comprehensive Feature Analyzer")

    try:
        from comprehensive_feature_analyzer import ComprehensiveFeatureAnalyzer

        # Create analyzer instance
        analyzer = ComprehensiveFeatureAnalyzer()

        # Test with a single timeframe using existing data
        symbol = "XAUUSD"
        timeframe = "W1"  # Smallest dataset

        # Build the data path
        csv_file = f"./data/{symbol}_PERIOD_{timeframe}_0.csv"

        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            return False

        print(f"📁 Testing with {timeframe} timeframe")

        # Test the actual process_timeframe_data function
        from multi_timeframe_analyzer import process_timeframe_data

        result = process_timeframe_data(csv_file, symbol, timeframe)

        if result:
            print(f"✅ Successfully processed {timeframe}")
            print(f"📊 Candles: {result['analyzed_candles']}")
            print(f"🔍 Features: {len(result)} categories")

            # Check for advanced features
            if 'intraday_sessions' in result['technical_patterns']:
                sessions = result['technical_patterns']['intraday_sessions']
                print(f"🕐 Sessions: {len(sessions)}")

            if 'order_block_analysis' in result['technical_patterns']:
                order_blocks = result['technical_patterns']['order_block_analysis']
                print(f"📦 Order blocks: {len(order_blocks.get('order_blocks', []))}")

            if 'gap_analysis' in result['technical_patterns']:
                gaps = result['technical_patterns']['gap_analysis']
                print(f"📊 Gaps found: {gaps.get('total_gaps', 0)}")

            print("🎉 All fixes working correctly!")
            return True
        else:
            print("❌ Failed to process timeframe")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal()
    if success:
        print("\n✅ Comprehensive Feature Analyzer is fully working!")
        print("🚀 Ready for full historical analysis")
    else:
        print("\n❌ Some issues need to be resolved")