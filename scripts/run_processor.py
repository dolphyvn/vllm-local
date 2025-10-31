#!/usr/bin/env python3
"""
Runner script for trading data processor
Automatically processes all CSV files in the data directory
"""

import os
import sys
import glob
from data_processor import TradingDataProcessor

def process_all_files(data_dir="./data", output_dir="./data/processed"):
    """Process all CSV files in the data directory"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Filter for MT5 period files
    period_files = [f for f in csv_files if "PERIOD_" in f]

    print(f"Found {len(period_files)} MT5 period files to process")
    print("="*60)

    results = {}

    for csv_file in period_files:
        filename = os.path.basename(csv_file)
        print(f"\nProcessing: {filename}")
        print("-"*60)

        try:
            # Process file
            processor = TradingDataProcessor(csv_file)
            result = processor.process_all()

            # Save output
            output_file = os.path.join(output_dir, filename.replace('.csv', '_processed.json'))

            import json
            with open(output_file, 'w') as f:
                json.dump({
                    'patterns': result['patterns'],
                    'levels': result['levels']
                }, f, indent=2)

            print(f"\nOutput saved to: {output_file}")

            results[filename] = {
                'patterns': len(result['patterns']),
                'levels': len(result['levels']),
                'status': 'success',
                'output_file': output_file
            }

        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            results[filename] = {
                'status': 'error',
                'error': str(e)
            }
            continue

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)

    for filename, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {filename}: {result['patterns']} patterns, {result['levels']} levels")
        else:
            print(f"❌ {filename}: {result['error']}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process trading CSV files into RAG format')
    parser.add_argument('--data-dir', default='./data', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='./data/processed', help='Directory for output files')
    parser.add_argument('--single', help='Process single file instead of all files')

    args = parser.parse_args()

    if args.single:
        # Process single file
        print(f"Processing single file: {args.single}")
        processor = TradingDataProcessor(args.single)
        result = processor.process_all()

        # Save output
        import json
        output_file = args.single.replace('.csv', '_processed.json')
        with open(output_file, 'w') as f:
            json.dump({
                'patterns': result['patterns'],
                'levels': result['levels']
            }, f, indent=2)

        print(f"\nOutput saved to: {output_file}")
    else:
        # Process all files
        process_all_files(args.data_dir, args.output_dir)
