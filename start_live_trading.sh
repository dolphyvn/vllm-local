#!/bin/bash
# Quick Start Script for Live Trading Analysis System

echo "🚀 Live Trading Analysis System - Quick Start"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if required directories exist
echo "📁 Checking directories..."
mkdir -p data/live
mkdir -p config
mkdir -p logs

# Check if source data exists
if [ ! -f "data/XAUUSD_PERIOD_M15_0.csv" ]; then
    echo "❌ Source data file not found: data/XAUUSD_PERIOD_M15_0.csv"
    echo "Please ensure you have the historical CSV data file."
    exit 1
fi

# Check if processed data exists (for RAG)
if [ ! -f "data/XAUUSD_PERIOD_M15_0_processed.json" ]; then
    echo "⚠️ Processed data not found. Checking if enhanced processor is running..."

    # Check if processor is running
    if pgrep -f "data_processor.py.*XAUUSD_PERIOD_M15_0.csv" > /dev/null; then
        echo "✅ Enhanced processor is running. Please wait for it to complete."
        echo "Once completed, run: python scripts/feed_to_rag_direct.py --file data/XAUUSD_PERIOD_M15_0_processed.json"
    else
        echo "❌ Processed data not found and processor is not running."
        echo "Please run the enhanced processor first:"
        echo "python scripts/data_processor.py data/XAUUSD_PERIOD_M15_0.csv"
    fi
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
python3 -m pip install pandas numpy ta requests chromadb > /dev/null 2>&1

# Test the system
echo "🧪 Testing system..."
python3 scripts/setup_live_trading.py test

if [ $? -eq 0 ]; then
    echo "✅ System test passed!"

    echo ""
    echo "🎯 Ready to start live trading system!"
    echo "Choose your option:"
    echo "1) Start complete system (recommended)"
    echo "2) Start live data feeder only"
    echo "3) Start alert system only"
    echo "4) View configuration"
    echo "5) Exit"
    echo ""

    read -p "Enter your choice (1-5): " choice

    case $choice in
        1)
            echo "🚀 Starting complete live trading system..."
            echo "Press Ctrl+C to stop all components"
            echo ""
            python3 scripts/setup_live_trading.py start --port 8080
            ;;
        2)
            echo "📡 Starting live data feeder..."
            python3 scripts/live_data_feeder.py --source data/XAUUSD_PERIOD_M15_0.csv --interval 60
            ;;
        3)
            echo "🔔 Starting alert system..."
            python3 scripts/live_alert_system.py --live-data data/live/XAUUSD_M15_LIVE.csv
            ;;
        4)
            echo "📋 Current configuration:"
            python3 scripts/setup_live_trading.py config
            ;;
        5)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "❌ System test failed. Please check the error messages above."
    echo ""
    echo "Common solutions:"
    echo "1. Ensure RAG system is running with processed data"
    echo "2. Check that all dependencies are installed"
    echo "3. Verify source data file exists and is readable"
    echo ""
    echo "For detailed setup instructions, see: scripts/LIVE_TRADING_GUIDE.md"
fi