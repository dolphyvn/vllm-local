#!/bin/bash

# 🚀 MT5 Trading System Startup Script
# Quick resume script for the trading analysis system

echo "🔄 Starting MT5 Trading Analysis System..."
echo "========================================"

# Navigate to project directory
cd /opt/works/personal/vllm-local

# Check if port 8080 is in use
if lsof -i:8080 > /dev/null 2>&1; then
    echo "⚠️  Port 8080 is already in use. Killing existing process..."
    lsof -ti:8080 | xargs kill -9
    sleep 2
fi

# Start FastAPI server
echo "🌐 Starting FastAPI server on port 8080..."
uvicorn main:app --reload --host 0.0.0.0 --port 8080 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 5

# Test server health
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Server started successfully!"
else
    echo "❌ Server failed to start. Check logs above."
    exit 1
fi

# Test upload functionality
echo "🧪 Testing upload functionality..."
if python scripts/test_multipart_upload.py > /dev/null 2>&1; then
    echo "✅ Upload endpoints working!"
else
    echo "⚠️  Upload test failed. Check configuration."
fi

# Display URLs
echo ""
echo "🎯 System is ready! Use these URLs:"
echo "=================================="
echo "🏠 Main Web UI:        http://localhost:8080/"
echo "📚 API Documentation:  http://localhost:8080/docs"
echo "📊 Upload Status:      http://localhost:8080/upload/status"
echo "🏥 Health Check:       http://localhost:8080/health"
echo ""
echo "📸 For MT5 EA Integration:"
echo "   Upload Endpoint:     POST http://localhost:8080/upload"
echo "   Fallback Endpoint:   POST http://localhost:8080/upload/simple"
echo ""
echo "📝 To stop the server: kill $SERVER_PID"
echo "📖 Full documentation: WORKFLOW_RESUME_GUIDE.md"
echo ""

# Keep script running or exit based on parameter
if [ "$1" == "--keep-running" ]; then
    echo "🔄 System running. Press Ctrl+C to stop."
    wait $SERVER_PID
else
    echo "✅ System started in background (PID: $SERVER_PID)"
    echo "💡 Use './start_system.sh --keep-running' to run in foreground"
fi