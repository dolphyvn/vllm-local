# 🤖 Local Financial Assistant

A powerful, private financial analysis and trading strategy assistant that runs entirely locally using Ollama. Get ChatGPT-like financial insights with complete privacy and no subscription fees.

## ✨ Features

### 🧠 **AI-Powered Financial Analysis**
- **Market Analysis**: Real-time analysis of forex, gold, and commodity markets
- **Trading Strategies**: AI-powered suggestions for entry/exit points
- **Risk Management**: Professional risk assessment and position sizing
- **Technical Analysis**: Chart pattern recognition and indicator interpretation
- **Economic Insights**: Impact analysis of economic events on markets

### 🌐 **Modern Web Interface**
- **ChatGPT-like Experience**: Real-time streaming responses
- **Professional Design**: Clean, modern interface with dark/light themes
- **Mobile Responsive**: Works perfectly on desktop and mobile devices
- **Interactive Memory**: Store and retrieve trading insights and rules
- **System Monitoring**: Real-time status of models and services

### 🔒 **Complete Privacy & Control**
- **100% Local**: All processing happens on your machine
- **No Data Sharing**: No API calls to external services
- **No Subscription Fees**: Free and unlimited usage
- **Customizable**: Fine-tune models for your specific needs

### ⚡ **High Performance**
- **Real-Time Streaming**: Watch responses appear token by token
- **Optimized Models**: Fast inference with Ollama
- **Efficient Memory**: Smart context management with ChromaDB
- **Resource Friendly**: Runs on standard consumer hardware

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Ollama installed (see Ollama setup below)
- At least 8GB RAM (16GB recommended)
- Modern web browser

### Step 1: Install Ollama

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
Download from https://ollama.com/download

### Step 2: Pull a Model
```bash
# Recommended for financial analysis
ollama pull llama3.2

# Alternative options
ollama pull llama3.1:8b    # For complex analysis
ollama pull mistral:7b     # Fast responses
ollama pull phi3:mini      # Lightweight option
```

### Step 3: Start the Assistant
```bash
# Clone or download this repository
git clone https://github.com/yourusername/vllm-local.git
cd vllm-local

# Install Python dependencies
pip install -r requirements.txt

# Start all services (Ollama + Web Interface)
./run.sh start
```

### Step 4: Access the Web Interface
Open your browser to: **http://localhost:8080**

## 🛠️ Detailed Setup

### Installation Options

#### Option A: Automated Setup (Recommended)
```bash
./run.sh start
```
This automatically:
- ✅ Checks Ollama installation
- ✅ Starts Ollama service
- ✅ Pulls required models
- ✅ Starts the web interface
- ✅ Verifies everything works

#### Option B: Manual Setup
```bash
# 1. Start Ollama (in one terminal)
ollama serve

# 2. Pull a model (in another terminal)
ollama pull llama3.2

# 3. Start the web interface (in third terminal)
python -m uvicorn main:app --reload --port 8080

# 4. Open browser to http://localhost:8080
```

#### Option C: Docker Setup
```bash
# Pull Ollama image
docker pull ollama/ollama

# Run Ollama container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull model through container
docker exec -it ollama ollama pull llama3.2

# Start the assistant
./run.sh fastapi-only
```

### Configuration

The system is configured through `config.json`:

```json
{
  "default_model": "llama3.2",
  "system_prompt": "You are a professional financial trading assistant...",
  "risk_profile": "moderate",
  "default_pair": "XAUUSD",
  "api_settings": {
    "ollama_base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "ollama_settings": {
    "recommended_models": [
      "llama3.2",
      "llama3.1:8b",
      "qwen2.5:7b",
      "mistral:7b",
      "phi3:mini"
    ]
  }
}
```

## 📊 Usage Examples

### Web Interface

1. **Basic Chat**: Just type your financial questions in the chat interface
2. **Memory Management**: Store important trading rules and insights
3. **System Monitoring**: Check model status and system health
4. **Theme Toggle**: Switch between light and dark modes

### API Usage

#### Chat Analysis
```python
import requests

response = requests.post(
    "http://localhost:8080/chat",
    json={
        "message": "Analyze current gold market trends",
        "model": "llama3.2",
        "memory_context": 3
    }
)
result = response.json()["response"]
print(result)
```

#### Streaming Responses
```python
import requests

response = requests.post(
    "http://localhost:8080/chat/stream",
    json={"message": "What are key risk management principles?"},
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = line[6:]  # Remove 'data: ' prefix
        if data == b'[DONE]':
            break
        try:
            chunk = json.loads(data)
            token = chunk['choices'][0]['delta']['content']
            print(token, end='', flush=True)
        except:
            continue
```

#### Store Trading Rules
```python
requests.post(
    "http://localhost:8080/memorize",
    json={
        "key": "risk_management",
        "value": "Never risk more than 2% per trade",
        "category": "risk"
    }
)
```

### Interactive Testing
```bash
# Run interactive test client
python test_client.py interactive

# Available commands:
# - 'quit' to exit
# - 'health' to check system status
# - 'models' to list available models
# - 'stream' to toggle streaming mode
# - 'memorize' to store information
```

## 🎯 Model Selection Guide

### Recommended Models for Financial Analysis

| Model | Size | Best For | Performance |
|-------|------|----------|-------------|
| **llama3.2** | 4.7B | **General financial analysis** | ⭐⭐⭐⭐⭐ |
| **llama3.1:8b** | 4.7B | Complex market analysis | ⭐⭐⭐⭐⭐ |
| **qwen2.5:7b** | 4.7B | Structured, detailed responses | ⭐⭐⭐⭐ |
| **mistral:7b** | 4.1B | Fast, real-time responses | ⭐⭐⭐⭐ |
| **phi3:mini** | 2.2B | Quick insights, low resource | ⭐⭐⭐ |

### Hardware Requirements

| Model | Min RAM | Recommended RAM | CPU Only | GPU Recommended |
|-------|---------|------------------|----------|----------------|
| phi3:mini | 4GB | 8GB | ✅ Good | ✅ Good |
| mistral:7b | 8GB | 16GB | ✅ OK | ✅ Good |
| llama3.2 | 8GB | 16GB | ⚠️ Slow | ✅ Good |
| llama3.1:8b | 8GB | 16GB | ⚠️ Slow | ✅ Better |

## 🔧 Advanced Configuration

### Custom Model Prompts
Edit `config.json` to customize the system prompt:

```json
{
  "system_prompt": "You are an expert quantitative analyst specializing in algorithmic trading strategies. Provide detailed technical analysis with specific entry/exit points, stop-loss levels, and risk-reward ratios."
}
```

### Memory Management
The system automatically stores conversations in ChromaDB for context retrieval. Customize settings in `config.json`:

```json
{
  "memory_settings": {
    "collection_name": "financial_memory",
    "persist_directory": "./chroma_db",
    "max_memory_age_days": 30
  }
}
```

### Model Fine-tuning
Create custom models with Modelfiles:

```dockerfile
FROM llama3.2
SYSTEM You are a professional financial analyst with 20 years of experience in forex and commodities trading.
PARAMETER temperature 0.1
PARAMETER stop "<|impossible>"
```

```bash
# Build custom model
ollama create fin-analyst -f Modelfile

# Update config.json to use custom model
```

## 📱 Features Overview

### 💬 Chat Interface
- **Real-time Streaming**: Watch responses appear instantly
- **Context Memory**: Previous conversations inform responses
- **Professional Layout**: Clean, distraction-free interface
- **Responsive Design**: Works on all device sizes

### 🧠 Memory System
- **Smart Storage**: Automatically saves important conversations
- **Quick Retrieval**: Relevant context is pulled from memory
- **Categorization**: Organize memories by trading, strategy, risk, etc.
- **Persistent Storage**: Memory survives across sessions

### 📊 System Monitoring
- **Model Status**: Real-time model availability checks
- **Memory Health**: ChromaDB connection status
- **API Connectivity**: Service health monitoring
- **Resource Usage**: System performance indicators

### 🎨 User Experience
- **Dark/Light Themes**: Toggle between themes
- **Keyboard Shortcuts**: Enter to send, Shift+Enter for new lines
- **Character Counting**: Track message length
- **Error Handling**: Clear error messages and recovery options

## 🔍 Testing and Troubleshooting

### Run Tests
```bash
# Comprehensive API tests
./run.sh test

# Interactive testing mode
./run.sh interactive

# Health check
curl http://localhost:8080/health
```

### Common Issues

#### "Ollama is not installed"
```bash
# Install Ollama
brew install ollama  # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux
```

#### "Model not found"
```bash
# Pull the required model
ollama pull llama3.2

# Check available models
ollama list
```

#### "Service not running"
```bash
# Check service status
./run.sh status

# Restart services
./run.sh restart
```

#### "Slow responses"
- Try a smaller model (`phi3:mini`, `mistral:7b`)
- Check system resource usage
- Close other applications
- Consider GPU acceleration

### Performance Optimization

#### CPU-only optimization
```json
{
  "api_settings": {
    "max_tokens": 1024,  // Reduce response length
    "temperature": 0.5   // Lower for faster responses
  }
}
```

#### Memory optimization
- Use smaller models
- Reduce context window size
- Clear old memories periodically

## 📚 Documentation

- **[README_WEB_UI.md](./README_WEB_UI.md)** - Web interface detailed guide
- **[README_STREAMING.md](./README_STREAMING.md)** - Streaming functionality
- **[README_OLLAMA.md](./README_OLLAMA.md)** - Comprehensive Ollama setup guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/vllm-local.git
cd vllm-local

# Install dependencies
pip install -r requirements.txt

# Start development server
./run.sh start

# Run tests
python test_client.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** - For making local LLM deployment accessible
- **FastAPI** - For the excellent Python web framework
- **ChromaDB** - For the efficient vector database
- **Financial Community** - For the domain expertise and inspiration

## 🎉 Ready to Start!

Your Local Financial Assistant is now ready to provide professional financial analysis and trading insights - all running locally on your machine with complete privacy and control.

🚀 **Start exploring AI-powered financial analysis today!**

---

**Need Help?**
- Check the detailed documentation in the `README_*.md` files
- Run `./run.sh test` for troubleshooting
- Open an issue on GitHub for support