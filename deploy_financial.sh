#!/bin/bash

# Deployment script for financial embedding system
# Run this on your production server

echo "🚀 Deploying Financial Embedding System"

# Step 1: Backup current config
echo "📋 Backing up current config.json..."
cp config.json config.backup.$(date +%Y%m%d_%H%M%S).json

# Step 2: Update configuration
echo "⚙️ Updating configuration with financial embeddings..."
cat > config.json << 'EOF'
{
  "default_model": "0xroyce/plutus:latest",
  "system_prompt": "You are a professional trading analyst and financial expert. Provide accurate, data-driven analysis for trading decisions. Always consider technical indicators, risk management, and market context. If you're uncertain about market movements, acknowledge the uncertainty rather than making definitive predictions.",
  "auth_settings": {
    "password": "admin123",
    "session_timeout_minutes": 480,
    "cookie_secret": "your-secret-key-change-this-in-production"
  },
  "financial_embedding": {
    "enabled": true,
    "model_type": "finance-technical-v1",
    "dimensions": 768,
    "max_sequence_length": 512,
    "normalize_embeddings": true
  },
  "api_settings": {
    "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_timeout": 300,
    "max_tokens": 2048,
    "temperature": 0.3,
    "memory_context_entries": 5,
    "financial_context_enabled": true
  },
  "memory_settings": {
    "collection_name": "financial_trading_memory",
    "persist_directory": "./chroma_db",
    "max_memory_age_days": 90,
    "backup_enabled": true,
    "financial_memory_enabled": true
  },
  "trading_settings": {
    "default_timeframes": ["1m", "5m", "15m", "1h", "1d"],
    "supported_indicators": ["RSI", "MACD", "BB", "SMA", "EMA", "VWAP", "ATR", "Stochastic", "ADX"],
    "risk_management": {
      "max_position_size": 0.05,
      "stop_loss_percentage": 0.02,
      "take_profit_percentage": 0.06
    }
  },
  "ollama_settings": {
    "recommended_models": [
      "llama3.2",
      "llama3.1:8b",
      "qwen2.5:7b",
      "mistral:7b",
      "phi3:mini",
      "0xroyce/plutus:latest"
    ],
    "financial_models": [
      "llama3.2",
      "llama3.1:8b",
      "qwen2.5:7b",
      "0xroyce/plutus:latest"
    ]
  }
}
EOF

echo "✅ Configuration updated"

# Step 3: Install dependencies
echo "📦 Installing financial embedding dependencies..."
pip3 install sentence-transformers torch numpy

echo "✅ Dependencies installed"

# Step 4: Create deployment guide
echo "📋 Creating deployment guide..."
cat > deployment_guide.md << 'EOF'
# Financial Embedding Deployment Guide

## Changes Made:
1. ✅ Updated config.json with financial embedding settings
2. ✅ Added finance-technical-v1 embedding model
3. ✅ Enhanced system prompt for trading analysis
4. ✅ Added trading-specific settings

## Next Steps:
1. Copy the new Python files to your server:
   - financial_embedding_config.py
   - financial_memory_manager.py

2. Update main.py to use FinancialMemoryManager:
   ```python
   from financial_memory_manager import FinancialMemoryManager

   # Replace this line:
   memory_manager = MemoryManager()

   # With this:
   memory_manager = FinancialMemoryManager(embedding_model="finance-technical-v1")
   ```

3. Restart your application:
   ```bash
   # Stop current process
   pkill -f "python3 main.py"

   # Start with new config
   python3 main.py
   ```

## Benefits:
- 🎯 Better understanding of technical indicators (RSI, MACD, etc.)
- 📊 Improved trading pattern recognition
- 🔍 More relevant context retrieval for trading queries
- 📈 Enhanced market scenario matching

EOF

echo "✅ Deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Copy financial_embedding_config.py and financial_memory_manager.py to /opt/vllm-local/"
echo "2. Update main.py to use FinancialMemoryManager"
echo "3. Restart your application"
echo ""
echo "📖 See deployment_guide.md for detailed instructions"