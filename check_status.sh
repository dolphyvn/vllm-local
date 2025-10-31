#!/bin/bash

# 📊 System Status Check Script
# Quick overview of current system status

echo "🔍 MT5 Trading System Status Check"
echo "================================="

# Check if server is running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ FastAPI Server: RUNNING (http://localhost:8080)"
else
    echo "❌ FastAPI Server: NOT RUNNING"
    echo "   Start with: ./start_system.sh"
    exit 1
fi

# Check upload status
echo ""
echo "📤 Upload Status:"
UPLOAD_STATUS=$(curl -s "http://localhost:8080/upload/status" 2>/dev/null)
if [ $? -eq 0 ]; then
    TOTAL_FILES=$(echo "$UPLOAD_STATUS" | jq -r '.total_mt5_files // 0')
    LIVE_FILES=$(echo "$UPLOAD_STATUS" | jq -r '.total_live_files // 0')
    echo "   Total MT5 Files: $TOTAL_FILES"
    echo "   Live Files: $LIVE_FILES"
else
    echo "   ⚠️  Could not fetch upload status"
fi

# Check recent uploads
echo ""
echo "📁 Recent Data Files:"
if [ -d "data" ]; then
    echo "   Main data directory: $(ls -1 data/*.csv 2>/dev/null | wc -l) files"
    echo "   Live data directory: $(ls -1 data/live/*.csv 2>/dev/null | wc -l) files"

    # Show most recent files
    echo "   Most recent uploads:"
    ls -lt data/*.csv 2>/dev/null | head -3 | while read line; do
        echo "     $line"
    done
else
    echo "   ❌ Data directory not found"
fi

# Check RAG knowledge base
echo ""
echo "🧠 Knowledge Base:"
KNOWLEDGE_STATS=$(curl -s "http://localhost:8080/api/knowledge/stats" 2>/dev/null)
if [ $? -eq 0 ]; then
    TOTAL_KNOWLEDGE=$(echo "$KNOWLEDGE_STATS" | jq -r '.total_entries // 0')
    echo "   Total Knowledge Entries: $TOTAL_KNOWLEDGE"
else
    echo "   ⚠️  Could not fetch knowledge stats"
fi

# System resources
echo ""
echo "💻 System Resources:"
echo "   Memory Usage: $(ps aux | grep uvicorn | grep -v grep | awk '{print $4}' | head -1)%"
echo "   CPU Usage: $(ps aux | grep uvicorn | grep -v grep | awk '{print $3}' | head -1)%"

# Quick URLs reminder
echo ""
echo "🌐 Quick Access URLs:"
echo "   Main UI:        http://localhost:8080/"
echo "   API Docs:       http://localhost:8080/docs"
echo "   Upload Status:  http://localhost:8080/upload/status"

echo ""
echo "✅ Status check complete!"
echo "📖 Full documentation: WORKFLOW_RESUME_GUIDE.md"