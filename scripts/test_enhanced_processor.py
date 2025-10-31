#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced trading data processor
with market profile, VWAP, session analysis, and auction theory
"""

import os
import sys
import json
from data_processor import TradingDataProcessor

def test_enhanced_processor():
    """Test the enhanced processor with sample data"""

    # Find a CSV file to test with
    data_dir = "./data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'PERIOD' in f]

    if not csv_files:
        print("No CSV files found in ./data directory")
        print("Please place some MT5 CSV files with 'PERIOD' in the filename")
        return

    # Use the first CSV file found
    csv_file = os.path.join(data_dir, csv_files[0])
    print(f"Testing with: {csv_file}")

    try:
        # Initialize the enhanced processor
        processor = TradingDataProcessor(csv_file)

        # Load and process data
        processor.load_csv()
        processor.calculate_indicators()

        # Test pattern detection
        patterns = processor.detect_patterns()

        if patterns:
            print(f"\n{'='*80}")
            print("ENHANCED TRADING PATTERN ANALYSIS DEMONSTRATION")
            print(f"{'='*80}")

            # Show the first enhanced pattern document
            sample_pattern = patterns[0]
            print("\nSample Enhanced Pattern Document:")
            print("-" * 80)
            print(sample_pattern['text'][:2000] + "..." if len(sample_pattern['text']) > 2000 else sample_pattern['text'])

            # Show enhanced metadata
            print("\nEnhanced Metadata:")
            print("-" * 40)
            metadata = sample_pattern['metadata']

            # Group metadata by category
            categories = {
                'Basic Info': ['pattern', 'date', 'time', 'session', 'session_major'],
                'Technical': ['rsi', 'trend', 'volume_ratio', 'bb_position'],
                'VWAP Analysis': ['vwap_deviation', 'daily_vwap'],
                'Market Profile': ['poc_price', 'value_area_width', 'in_value_area'],
                'Correlations': ['price_correlation', 'volume_correlation'],
                'Microstructure': ['buying_pressure', 'liquidity_ratio', 'price_efficiency'],
                'Auction Theory': ['auction_balance', 'auction_position'],
                'Advanced Metrics': ['trade_quality_score', 'trade_conviction', 'market_efficiency_score']
            }

            for category, keys in categories.items():
                print(f"\n{category}:")
                for key in keys:
                    if key in metadata:
                        value = metadata[key]
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")

            # Summary statistics
            print(f"\n{'='*60}")
            print("PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total patterns found: {len(patterns)}")
            print(f"Timeframe detected: {processor.timeframe_label}")
            print(f"Total data points: {len(processor.df)}")

            # Show some enhanced features summary
            if 'price_correlation' in processor.df.columns:
                avg_correlation = processor.df['price_correlation'].mean()
                print(f"Average interday correlation: {avg_correlation:.2f}")

            if 'poc_price' in processor.df.columns:
                poc_count = processor.df['poc_price'].notna().sum()
                print(f"Market profile data points: {poc_count}")

            if 'daily_vwap' in processor.df.columns:
                vwap_count = processor.df['daily_vwap'].notna().sum()
                print(f"VWAP data points: {vwap_count}")

            print(f"\n✅ Enhanced processor test completed successfully!")

            # Save a sample output
            output_file = csv_file.replace('.csv', '_enhanced_sample.json')
            with open(output_file, 'w') as f:
                json.dump({
                    'sample_pattern': patterns[0],
                    'metadata_summary': {
                        'total_patterns': len(patterns),
                        'enhanced_features': [
                            'Market Profile Analysis',
                            'Session-specific VWAP',
                            'Interday Correlations',
                            'Auction Theory Metrics',
                            'Market Microstructure',
                            'Enhanced Volume Analysis',
                            'Advanced Momentum Indicators'
                        ]
                    }
                }, f, indent=2)

            print(f"Sample output saved to: {output_file}")

        else:
            print("No patterns detected in the data")

    except Exception as e:
        print(f"❌ Error testing enhanced processor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_processor()