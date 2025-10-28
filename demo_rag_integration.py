#!/usr/bin/env python3
"""
RAG MT5 Integration Demo
Shows how to use the integration system
"""

import os
import sys
import pandas as pd
from datetime import datetime
import requests
import json

def create_sample_rag_data():
    """Create sample RAG data for testing"""
    print("📊 Creating sample RAG MT5 data...")

    # Sample data structure from your RAG system
    sample_data = {
        'Timestamp': ['2025.10.28 15:00', '2025.10.28 15:05', '2025.10.28 15:10'],
        'Open': [2693.50, 2693.77, 2694.21],
        'High': [2694.21, 2695.80, 2696.50],
        'Low': [2690.15, 2691.20, 2692.80],
        'Close': [2693.77, 2694.61, 2695.30],
        'Volume': [1523, 1680, 1450],
        'RSI': [65.2, 67.8, 70.5],
        'MACD': [0.0234, 0.0256, 0.0278],
        'MACD_Signal': [0.0189, 0.0195, 0.0201],
        'MACD_Hist': [0.0045, 0.0061, 0.0077],
        'EMA_Short': [2686.50, 2687.20, 2687.90],
        'EMA_Medium': [2680.30, 2680.80, 2681.30],
        'EMA_Long': [2665.80, 2666.20, 2666.60],
        'ATR': [4.23, 4.35, 4.40],
        'BB_Upper': [2695.80, 2696.90, 2698.00],
        'BB_Middle': [2688.50, 2689.20, 2689.90],
        'BB_Lower': [2681.20, 2681.50, 2681.80],
        'Volume_Avg': [1580, 1590, 1600],
        'Trend': ['BULLISH', 'BULLISH', 'BULLISH'],
        'Session': ['US_SESSION', 'US_SESSION', 'US_SESSION'],
        'Day_of_Week': ['MONDAY', 'MONDAY', 'MONDAY'],
        'Hour': [15, 15, 15],
        'Pattern': ['STANDARD_CANDLE', 'STANDARD_CANDLE', 'STANDARD_CANDLE'],
        'Signal': ['BULLISH', 'BULLISH', 'BULLISH'],
        'Confidence': [50.0, 65.0, 75.0],
        'POC_Distance': [12.3, 10.5, 8.7],
        'VA_Position': ['WITHIN_VA', 'WITHIN_VA', 'ABOVE_VA'],
        'Price_Level': ['AROUND_POC', 'ABOVE_POC', 'ABOVE_POC']
    }

    df = pd.DataFrame(sample_data)

    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Save sample RAG file
    filename = f"./data/RAG_XAUUSD_{datetime.now().strftime('%Y.%m.%d')}.csv"
    df.to_csv(filename, index=False)

    print(f"✅ Sample RAG data created: {filename}")
    print(f"📊 Created {len(df)} sample data points")
    return filename

def test_integration(export_path: str = "./data"):
    """Test the integration system"""
    print(f"\n🧪 Testing RAG MT5 Integration with directory: {export_path}")

    # Import and test integrator
    try:
        from integrate_rag_mt5_data import RAGMT5Integrator

        print("\n🔧 Initializing integrator...")
        integrator = RAGMT5Integrator(export_path=export_path)

        print("\n📊 Testing file processing...")
        results = integrator.process_all_files()

        print(f"\n📈 Integration Results:")
        print(f"  - Files processed: {results.get('processed', 0)}")
        print(f"  - Total entries: {results.get('total_entries', 0)}")
        print(f"  - Files: {results.get('files', [])}")

        return results.get('processed', 0) > 0

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_knowledge_retrieval():
    """Test if the AI can retrieve the fed knowledge"""
    print("\n🔍 Testing knowledge retrieval...")

    try:
        # Test authentication
        session = requests.Session()
        response = session.post(
            "http://localhost:8080/auth/login",
            json={"password": "admin123"}
        )

        if response.status_code != 200:
            print("❌ Authentication failed")
            return False

        token = response.json().get("session_token")
        session.headers.update({"Authorization": f"Bearer {token}"})

        # Test knowledge stats
        response = session.get("http://localhost:8080/api/knowledge/stats")

        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Knowledge Base Stats:")
            print(f"  - Total entries: {stats.get('total_entries', 0)}")
            print(f"  - Trading analysis: {stats.get('by_category', {}).get('trading_analysis', 0)}")
            print(f"  - Lessons: {stats.get('lessons_count', 0)}")

            # Test chat with fed knowledge
            chat_response = session.post(
                "http://localhost:8080/chat",
                json={"message": "What do you know about XAUUSD technical analysis?"}
            )

            if chat_response.status_code == 200:
                ai_response = chat_response.json().get('response', '')
                print(f"\n🤖 AI Response Test:")
                print(f"  - Response length: {len(ai_response)} characters")
                print(f"  - Contains trading knowledge: {'yes' if 'RSI' in ai_response or 'MACD' in ai_response else 'no'}")

                return True
            else:
                print(f"❌ Chat test failed: {chat_response.status_code}")
                return False
        else:
            print(f"❌ Stats check failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Knowledge retrieval test failed: {e}")
        return False

def main():
    """Main demonstration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG MT5 Integration Demo - Test integration with custom directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default ./data directory
  python3 demo_rag_integration.py

  # Test with custom directory
  python3 demo_rag_integration.py --export-path /path/to/trading/data

  # Create sample data first
  python3 demo_rag_integration.py --create-sample
        """
    )

    parser.add_argument("--export-path", default="./data",
                       help="Path to RAG CSV files (default: ./data)")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data and exit")

    args = parser.parse_args()

    # Create directory if needed
    if not os.path.exists(args.export_path):
        print(f"📁 Creating directory: {args.export_path}")
        os.makedirs(args.export_path, exist_ok=True)

    print("🚀 RAG MT5 Integration Demo")
    print("=" * 50)
    print(f"📁 Directory: {os.path.abspath(args.export_path)}")

    # List existing files
    csv_files = [f for f in os.listdir(args.export_path) if f.endswith('.csv')]
    rag_files = [f for f in csv_files if f.startswith('RAG_')]

    print(f"📊 Files found:")
    print(f"   Total CSV files: {len(csv_files)}")
    print(f"   RAG files: {len(rag_files)}")

    if rag_files:
        print(f"   Latest RAG file: {max(rag_files)}")

    # Create sample data if requested
    if args.create_sample:
        sample_file = create_sample_rag_data()
        print(f"\n✅ Sample data created: {sample_file}")
        print("   Now run the integration test.")
        return

    # If no RAG files exist, offer to create sample
    if not rag_files and not csv_files:
        print(f"\n⚠️ No CSV files found in {args.export_path}")
        response = input("Create sample RAG data for testing? (y/n): ").lower().strip()
        if response == 'y':
            create_sample_rag_data()
        else:
            print("❌ No data to test. Exiting.")
            return

    # Test integration
    integration_success = test_integration(args.export_path)

    # Test knowledge retrieval
    retrieval_success = test_knowledge_retrieval()

    # Summary
    print("\n📋 Demo Summary:")
    print(f"  - Integration test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"  - Knowledge retrieval: {'✅ PASSED' if retrieval_success else '❌ FAILED'}")

    if integration_success and retrieval_success:
        print("\n🎉 All tests passed! Your RAG MT5 integration is working.")
        print(f"\n📝 Next Steps:")
        print(f"  1. Place your real RAG MT5 CSV files in {args.export_path}")
        print(f"  2. Run: python3 integrate_rag_mt5_data.py --export-path {args.export_path}")
        print(f"  3. Or monitor: python3 integrate_rag_mt5_data.py --mode monitor --export-path {args.export_path}")
        print(f"  4. Or schedule: python3 schedule_rag_feeding.py --export-path {args.export_path}")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        print("\n💡 Tips:")
        print("  - Ensure Financial Assistant server is running")
        print("  - Check if directory contains RAG_*.csv files")
        print("  - Verify API credentials are correct")

if __name__ == "__main__":
    main()