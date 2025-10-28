# RAG MT5 Data Integration Guide

## 🎯 Overview

This guide shows how to integrate your RAG MT5 trading system data with your Financial Assistant workflow. The integration automatically converts MT5 trading analysis into knowledge that your AI can learn from.

## 📁 Files Created

### **Integration Scripts:**
- `integrate_rag_mt5_data.py` - Main integration script
- `schedule_rag_feeding.py` - Automated scheduler
- `demo_rag_integration.py` - Demonstration and testing
- `RAG_MT5_Integration_Guide.md` - This guide

### **Existing Data:**
- `data/XAUUSD-2025.10.21.csv` - Your existing RAG training data
- `data/XAUUSD-2025.10.21T.csv` - Target prediction data

## 🚀 Quick Start

### **1. Start Your Financial Assistant**
```bash
# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Or use your run script
./run.sh start
```

### **2. Test the Integration**
```bash
# Run the demo to test everything works
python3 demo_rag_integration.py

# Test with custom directory
python3 demo_rag_integration.py --export-path /path/to/your/rag/data
```

### **3. Process Existing RAG Data**
```bash
# One-time processing of default ./data directory
python3 integrate_rag_mt5_data.py

# Process specific directory
python3 integrate_rag_mt5_data.py --export-path /path/to/mt5/exports

# Interactive mode - will prompt for directory
python3 integrate_rag_mt5_data.py

# List files in directory
python3 integrate_rag_mt5_data.py --export-path ./data --list-files
```

### **4. Set Up Continuous Learning**
```bash
# Monitor default directory
python3 integrate_rag_mt5_data.py --mode monitor --interval 5

# Monitor custom directory
python3 integrate_rag_mt5_data.py --mode monitor --export-path /path/to/trading/data

# Use scheduler with custom directory
python3 schedule_rag_feeding.py --export-path /home/user/mt5_exports

# Monitor remote server
python3 integrate_rag_mt5_data.py --base-url http://ai.vn.aliases.me --export-path ./trading_data
```

## 📊 How It Works

### **Data Flow:**
```
MT5 RAG System → CSV Export → Integration Script → Knowledge API → Financial Assistant
```

### **Knowledge Creation:**
Your RAG data with 25+ features gets converted into:
- **Technical Analysis Entries**: RSI, MACD, trend analysis
- **Pattern Recognition**: Candlestick patterns and interpretations
- **Market Context**: Session information and volatility
- **Trading Lessons**: Structured learning from historical data

### **Sample Knowledge Entry:**
```json
{
  "topic": "XAUUSD Technical Analysis 2025.10.28 15:00",
  "content": "Trading Analysis for XAUUSD: RSI: 65.2, MACD: 0.0234, Trend: BULLISH...",
  "category": "trading_analysis",
  "confidence": 0.65,
  "tags": ["xauusd", "technical", "rsi", "macd", "bullish"],
  "source": "RAG_MT5_RAG_XAUUSD_2025.10.28.csv"
}
```

## 🔧 Configuration

### **Integration Settings:**
```python
# In integrate_rag_mt5_data.py
integrator = RAGMT5Integrator(
    base_url="http://localhost:8080",  # Your Financial Assistant URL
    password="admin123",               # Your API password
    export_path="./data",              # Where your RAG CSV files are stored
    log_file="./processed_files.log"   # Track processed files
)
```

### **Directory Input Options:**

#### **Command Line:**
```bash
# Specify directory directly
python3 integrate_rag_mt5_data.py --export-path /path/to/your/rag/files

# Interactive mode (prompts for directory)
python3 integrate_rag_mt5_data.py

# List files in directory
python3 integrate_rag_mt5_data.py --list-files --export-path ./data
```

#### **Programmatic:**
```python
from integrate_rag_mt5_data import RAGMT5Integrator

integrator = RAGMT5Integrator(
    export_path="/home/user/mt5_exports",
    log_file="./custom_processed.log"
)
results = integrator.process_all_files()
```

#### **Scheduler with Custom Directory:**
```bash
# Monitor specific directory
python3 schedule_rag_feeding.py --export-path /path/to/mt5/exports

# Full custom setup
python3 schedule_rag_feeding.py \
  --base-url http://ai.vn.aliases.me \
  --export-path /home/user/trading/data \
  --log-file ./trading_processed.log
```

## 📈 Expected Results

### **Knowledge Base Growth:**
- **1 RAG file** (~400 rows) → ~400 knowledge entries
- **1 trading lesson** per symbol with sufficient data
- **Automatic tagging** for easy retrieval
- **Confidence scoring** based on RAG analysis

### **AI Capabilities After Integration:**
- Answer technical analysis questions
- Recognize trading patterns
- Explain RSI/MACD signals
- Provide market context
- Reference historical examples

## 🧪 Testing Your Integration

### **1. Test Data Processing:**
```bash
python3 demo_rag_integration.py
```

### **2. Check Knowledge Stats:**
```bash
# Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}' | jq -r '.session_token')

# Check knowledge statistics
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/knowledge/stats
```

### **3. Test AI Responses:**
```bash
# Test with fed knowledge
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What do you know about XAUUSD RSI analysis?"}'
```

## 📁 File Management

### **Processed Files:**
- Files are tracked in `processed_rag_files.log`
- Already processed files are skipped
- Manual processing possible by removing from log

### **Knowledge Categories:**
- `trading_analysis` - Individual data points
- `trading_education` - Structured lessons
- `corrections` - Error corrections (if needed)

## 🔄 Continuous Learning Workflow

### **Automated Schedule:**
1. **Every 30 minutes**: Check for new RAG files
2. **Major trading sessions**: Process at session opens
3. **Weekly analysis**: Review knowledge base growth

### **Manual Integration:**
```bash
# Process specific file
python3 -c "
from integrate_rag_mt5_data import RAGMT5Integrator
integrator = RAGMT5Integrator()
entries = integrator.parse_rag_csv('./data/RAG_XAUUSD_2025.10.28.csv')
integrator.feed_knowledge_bulk(entries)
"
```

## 🎯 Best Practices

### **RAG Data Quality:**
- Ensure CSV files have required columns
- Check confidence scores are reasonable
- Verify timestamp consistency

### **Knowledge Management:**
- Monitor knowledge base size
- Review AI responses for accuracy
- Add corrections if needed

### **Performance:**
- Use bulk uploads for efficiency
- Schedule during off-hours if needed
- Monitor server resources

## 🚨 Troubleshooting

### **Common Issues:**

#### **1. Connection Error:**
```bash
# Check server is running
curl http://localhost:8080/health

# Start server if needed
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

#### **2. Authentication Failed:**
```bash
# Verify password
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}'
```

#### **3. No New Files Found:**
```bash
# Check data directory
ls -la data/

# Check processed log
cat processed_rag_files.log

# Clear log to reprocess (if needed)
rm processed_rag_files.log
```

#### **4. Knowledge Not Retrieved:**
```bash
# Check knowledge stats
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/knowledge/stats

# Check ChromaDB
ls -la chroma_db/
```

## 📊 Monitoring

### **Knowledge Growth:**
```bash
# Monitor statistics
watch -n 30 'curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/knowledge/stats | jq'
```

### **Integration Logs:**
```bash
# Monitor integration logs
tail -f fastapi.log | grep -i "rag\|knowledge\|mt5"
```

## 🎉 Success Indicators

### **When Integration is Working:**
- ✅ RAG CSV files are processed successfully
- ✅ Knowledge entries appear in stats
- ✅ AI responds with trading analysis
- ✅ Historical patterns are referenced
- ✅ Lessons are created automatically

### **Expected AI Responses:**
```
User: "What does an RSI of 65 mean?"
AI: "Based on the XAUUSD data I've learned, an RSI of 65 indicates
the market is approaching overbought territory but still has room
for upward movement. In our historical analysis, RSI levels between
60-70 during US sessions often preceded bullish continuation patterns..."

User: "Tell me about XAUUSD patterns"
AI: "From the 400+ data points I've analyzed, XAUUSD shows strong
session-based patterns. During US sessions, bullish patterns with
RSI above 60 often lead to continuation moves. The data shows
specific volatility characteristics that help predict reversals..."
```

## 📞 Support

### **For Issues:**
1. Check server logs: `tail -f fastapi.log`
2. Verify data format: Check CSV columns
3. Test authentication: Manual login test
4. Check API endpoints: `/docs` for API documentation

### **Next Steps:**
1. Deploy to production server
2. Set up monitoring
3. Expand to other symbols (EURUSD, BTCUSD)
4. Add more sophisticated analysis
5. Integrate real-time predictions

---

**Your RAG MT5 integration is now ready to enhance your Financial Assistant with real trading knowledge!**