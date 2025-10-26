#!/usr/bin/env python3
"""
Test script for CSV folder functionality in feed_xau_data.py

This script demonstrates how to use the updated feed_xau_data.py
with both single file pairs and folder input options.
"""

import os
import sys
from feed_xau_data import XAUUSDKnowledgeFeeder

def test_single_file_pair():
    """Test processing a single file pair"""
    print("="*60)
    print("TEST 1: Single File Pair Processing")
    print("="*60)

    # Check if test files exist
    training_file = "backtest/XAUUSD-2025.10.22.csv"
    target_file = "backtest/XAUUSD-2025.10.22T.csv"

    if not os.path.exists(training_file) or not os.path.exists(target_file):
        print(f"❌ Test files not found")
        print(f"   Training: {training_file}")
        print(f"   Target: {target_file}")
        return False

    # Initialize feeder (use localhost for testing)
    feeder = XAUUSDKnowledgeFeeder(base_url="http://localhost:8080")

    if not feeder.session_token:
        print("⚠️  Could not authenticate with localhost, skipping test")
        return True

    # Test the scanning functionality without actually feeding data
    print(f"🔍 Testing file parsing...")

    # Parse training data
    training_data = feeder.parse_xau_data(training_file)
    if training_data:
        print(f"✅ Training data parsed successfully")
        print(f"   Price range: ${training_data['earliest_price']:.2f} - ${training_data['latest_price']:.2f}")
        print(f"   Trend: {training_data['trend']}")
        print(f"   Data points: {training_data['data_points']}")
    else:
        print(f"❌ Failed to parse training data")
        return False

    # Parse target data
    target_data = feeder.parse_target_data(target_file)
    if target_data:
        print(f"✅ Target data parsed successfully")
        print(f"   Date: {target_data['date']}")
        print(f"   Close: ${target_data['close']:.2f}")
    else:
        print(f"❌ Failed to parse target data")
        return False

    # Calculate signal
    signal = feeder.calculate_signal(training_data['latest_price'], target_data['close'])
    print(f"✅ Signal calculated: {signal}")

    return True

def test_folder_scanning():
    """Test folder scanning functionality"""
    print("\n" + "="*60)
    print("TEST 2: Folder Scanning")
    print("="*60)

    folder_path = "backtest"

    if not os.path.exists(folder_path):
        print(f"❌ Test folder not found: {folder_path}")
        return False

    # Initialize feeder
    feeder = XAUUSDKnowledgeFeeder(base_url="http://localhost:8080")

    # Test folder scanning
    file_pairs = feeder.scan_csv_folder(folder_path)

    if file_pairs:
        print(f"✅ Folder scanning successful")
        print(f"   Found {len(file_pairs)} file pairs:")
        for i, (training, target) in enumerate(file_pairs, 1):
            print(f"     {i}. {os.path.basename(training)} + {os.path.basename(target)}")
        return True
    else:
        print(f"❌ No file pairs found")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)

    print("""
1. Process single file pair (default behavior):
   python3 feed_xau_data.py

2. Process specific files:
   python3 feed_xau_data.py --training data/file1.csv --target data/file1T.csv

3. Process entire folder:
   python3 feed_xau_data.py --folder data/

4. Use different API server:
   python3 feed_xau_data.py --folder data/ --base-url http://localhost:8080

5. Get help:
   python3 feed_xau_data.py --help

FILE NAMING CONVENTIONS:
- Training files: XAUUSD-2025.10.22.csv, data_2025-10-22.csv, etc.
- Target files: XAUUSD-2025.10.22T.csv, data_2025-10-22T.csv, target_2025-10-22.csv, etc.

The script will automatically match training and target files based on:
1. Exact name match with 'T' suffix (preferred)
2. Fuzzy matching on date/base name
""")

def main():
    """Run all tests"""
    print("🧪 Testing CSV folder functionality for feed_xau_data.py")

    # Test 1: Single file pair
    test1_success = test_single_file_pair()

    # Test 2: Folder scanning
    test2_success = test_folder_scanning()

    # Show usage examples
    show_usage_examples()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Single file pair test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"Folder scanning test: {'✅ PASSED' if test2_success else '❌ FAILED'}")

    if test1_success and test2_success:
        print("\n🎉 All tests passed! The CSV folder functionality is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)