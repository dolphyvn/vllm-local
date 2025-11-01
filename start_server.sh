#!/bin/bash

# Live Trading System Server Startup Script
# Usage: ./start_server.sh

echo "🚀 Starting vLLM-Local Trading System Server..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if port 8080 is already in use
if lsof -i :8080 | grep -q "LISTEN"; then
    echo "⚠️  Port 8080 is already in use:"
    lsof -i :8080
    echo ""
    echo "Options:"
    echo "1. Kill the process using port 8080:"
    echo "   sudo lsof -ti :8080 | xargs sudo kill -9"
    echo "2. Use a different port:"
    echo "   export PORT=8081"
    echo "   ./start_server.sh"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1)
echo "🐍 Python version: $python_version"

# Check for key dependencies
echo "📦 Checking dependencies..."

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "❌ FastAPI not found. Installing..."
    pip3 install fastapi
fi

if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "❌ Uvicorn not found. Installing..."
    pip3 install uvicorn
fi

if ! python3 -c "import ollama" 2>/dev/null; then
    echo "⚠️  Ollama not found. Please install if you want to use local LLM:"
    echo "   pip3 install ollama"
fi

# Set environment variables
export HOST=0.0.0.0
export PORT=8080
echo "🌐 Server will run on: http://$HOST:$PORT"

# Check if ChromaDB works
echo "🗄️  Testing ChromaDB connection..."
python3 -c "
import sys
sys.path.append('scripts')
try:
    from chroma_live_analyzer import ChromaLiveAnalyzer
    analyzer = ChromaLiveAnalyzer()
    stats = analyzer.get_collection_stats()
    print(f'✅ ChromaDB connected: {stats[\"total_analyses\"]} analyses stored')
except Exception as e:
    print(f'⚠️  ChromaDB warning: {e}')
"

echo ""
echo "🚀 Starting server..."
echo "📝 Logs will appear below. Press Ctrl+C to stop the server."
echo "🌐 Chat UI: http://localhost:$PORT"
echo "📊 API: http://localhost:$PORT/docs"
echo ""

# Start the server
python3 main.py