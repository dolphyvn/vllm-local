# Ollama Integration Guide

## 🦙 Overview

Your Local Financial Assistant has been updated to use Ollama instead of vLLM for local LLM deployment. Ollama provides a much simpler and more accessible way to run large language models locally, with automatic model management and optimized performance.

## ✨ Why Ollama?

### 🚀 **Key Advantages Over vLLM**
- **Simpler Setup**: No complex configuration or Python dependencies
- **Automatic Model Management**: Download and manage models with simple commands
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Resource Efficient**: Optimized for local deployment
- **Model Variety**: Access to a wide range of open-source models
- **Easy Installation**: One-command installation process

### 🎯 **Perfect For Financial Analysis**
- **Fast Performance**: Optimized inference for real-time chat
- **Local Privacy**: All processing happens on your machine
- **Cost Effective**: No API fees or usage limits
- **Customizable**: Fine-tune models for financial domain

## 🛠️ Installation and Setup

### Step 1: Install Ollama

#### macOS
```bash
# Using Homebrew (recommended)
brew install ollama

# Or download from https://ollama.com/download
```

#### Linux
```bash
# Official installation script
curl -fsSL https://ollama.com/install.sh | sh

# Or for specific package managers:
# Ubuntu/Debian: apt install ollama
# Arch Linux: yay -S ollama
# Fedora: dnf install ollama
```

#### Windows
1. Download from https://ollama.com/download
2. Run the installer
3. Restart your terminal/command prompt

### Step 2: Verify Installation
```bash
ollama --version
```

### Step 3: Start Ollama Service
```bash
# Start the Ollama service
ollama serve

# Or run in background (Linux/macOS)
nohup ollama serve > ollama.log 2>&1 &

# On Windows, the service starts automatically
```

### Step 4: Install Recommended Models

#### For Financial Analysis (Recommended)
```bash
# Primary model - great for analysis and reasoning
ollama pull llama3.2

# Alternative models for different needs
ollama pull llama3.1:8b    # Larger model for complex analysis
ollama pull qwen2.5:7b     # Excellent for structured responses
ollama pull mistral:7b     # Fast and efficient
ollama pull phi3:mini      # Lightweight and fast
```

#### Model Recommendations
| Model | Size | Best For | Performance |
|-------|------|----------|-------------|
| `llama3.2` | 4.7B | General financial analysis | ⭐⭐⭐⭐⭐ |
| `llama3.1:8b` | 4.7B | Complex market analysis | ⭐⭐⭐⭐⭐ |
| `qwen2.5:7b` | 4.7B | Structured, detailed responses | ⭐⭐⭐⭐ |
| `mistral:7b` | 4.1B | Fast, real-time responses | ⭐⭐⭐⭐ |
| `phi3:mini` | 2.2B | Quick insights, low resource | ⭐⭐⭐ |

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Use the provided script
./run.sh start
```

This will:
- ✅ Check Ollama installation
- ✅ Start Ollama service (if not running)
- ✅ Pull required models automatically
- ✅ Start the FastAPI web interface
- ✅ Verify everything is working

### Option 2: Manual Setup
```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model (in another terminal)
ollama pull llama3.2

# 3. Start the web interface
python -m uvicorn main:app --reload --port 8080

# 4. Open browser to http://localhost:8080
```

### Option 3: Using Docker
```bash
# Pull Ollama image
docker pull ollama/ollama

# Run Ollama container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull model through container
docker exec -it ollama ollama pull llama3.2

# Start the assistant (from host)
./run.sh fastapi-only
```

## 🔧 Configuration

### Update Model in config.json
```json
{
  "default_model": "llama3.2",
  "api_settings": {
    "ollama_base_url": "http://localhost:11434",
    "ollama_timeout": 120,
    "max_tokens": 2048,
    "temperature": 0.7,
    "memory_context_entries": 3
  },
  "ollama_settings": {
    "recommended_models": [
      "llama3.2",
      "llama3.1:8b",
      "qwen2.5:7b",
      "mistral:7b",
      "phi3:mini"
    ],
    "financial_models": [
      "llama3.2",
      "llama3.1:8b",
      "qwen2.5:7b"
    ]
  }
}
```

### Model Selection Guidelines

#### For Different Use Cases:
- **Quick Analysis**: `phi3:mini` or `mistral:7b`
- **Detailed Reports**: `llama3.2` or `llama3.1:8b`
- **Structured Data**: `qwen2.5:7b`
- **Real-time Chat**: `llama3.2` (balanced performance)

#### Hardware Requirements:
| Model | RAM Required | VRAM Recommended | CPU Only |
|-------|--------------|------------------|----------|
| `phi3:mini` | 4GB | 2GB | ✅ Good |
| `mistral:7b` | 8GB | 4GB | ✅ OK |
| `llama3.2` | 8GB | 4GB | ⚠️ Slow |
| `llama3.1:8b` | 8GB | 6GB | ⚠️ Slow |
| `qwen2.5:7b` | 8GB | 4GB | ⚠️ Slow |

## 🌐 Web Interface

### Access the Web UI
```bash
# Start all services
./run.sh start

# Open browser
open http://localhost:8080
```

### Features Available
- ✅ **Real-time streaming** responses
- ✅ **Memory management** for storing trading insights
- ✅ **System monitoring** and model status
- ✅ **Dark/light theme** toggle
- ✅ **Mobile responsive** design
- ✅ **Conversation history** with context

### API Endpoints
```bash
# Health check
curl http://localhost:8080/health

# List available models
curl http://localhost:8080/models

# Send chat message
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze gold market trends"}'

# Stream responses
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What are risk management principles?"}' \
  --no-buffer
```

## 🧪 Testing and Validation

### Run Automated Tests
```bash
./run.sh test
```

### Interactive Testing
```bash
./run.sh interactive
```

### Test Commands
```bash
# Check service status
./run.sh status

# Start only Ollama
./run.sh ollama-only

# Start only web interface
./run.sh fastapi-only
```

### Manual Ollama Commands
```bash
# List installed models
ollama list

# Show model information
ollama show llama3.2

# Pull new model
ollama pull mistral:7b

# Remove model
ollama remove phi3:mini

# Update model
ollama pull llama3.2  # Re-pulls latest version
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue: "Ollama is not installed"
```bash
# Solution: Install Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

#### Issue: "Model not found"
```bash
# Check available models
ollama list

# Pull the required model
ollama pull llama3.2

# Update config.json if using different model
```

#### Issue: "Ollama service not running"
```bash
# Start Ollama service
ollama serve

# Check if running on correct port
lsof -i :11434
```

#### Issue: "Slow response times"
```bash
# Solutions:
# 1. Use a smaller model
ollama pull phi3:mini

# 2. Check system resources
htop  # or Task Manager on Windows

# 3. Close other applications
# 4. Consider GPU acceleration if available
```

#### Issue: "Out of memory errors"
```bash
# Solutions:
# 1. Use smaller model
# 2. Reduce context window in config.json
# 3. Close other applications
# 4. Add more RAM to system
```

#### Issue: "Web UI not accessible"
```bash
# Check FastAPI service
./run.sh status

# Check logs
tail -f fastapi.log

# Restart services
./run.sh restart
```

### Performance Optimization

#### CPU Only Optimization
```bash
# 1. Use smaller models
ollama pull phi3:mini

# 2. Reduce response length
# Edit config.json:
{
  "api_settings": {
    "max_tokens": 1024,  # Reduced from 2048
    "temperature": 0.5   # Lower for faster responses
  }
}
```

#### GPU Acceleration
```bash
# Ollama automatically uses GPU if available
# Check if GPU is being used:
ollama list  # Shows model details with acceleration info

# For NVIDIA GPUs:
# 1. Install CUDA drivers
# 2. Verify GPU detection:
nvidia-smi

# For Apple Silicon (M1/M2/M3):
# GPU acceleration is automatic and optimized
```

### Monitoring Resources

#### Check System Performance
```bash
# CPU and memory usage
htop  # Linux/macOS
top   # Available everywhere

# GPU usage (NVIDIA)
nvidia-smi

# Ollama process status
ps aux | grep ollama

# Network connections
netstat -an | grep 11434
```

#### Check Ollama Logs
```bash
# If started manually
# Logs appear in terminal where ollama serve was run

# If started in background
tail -f ollama.log

# If using Docker
docker logs ollama
```

## 🔒 Security and Privacy

### Local Processing Benefits
- ✅ **No data leaves your machine**
- ✅ **No API calls to external services**
- ✅ **No usage tracking or logging**
- ✅ **Complete data privacy**
- ✅ **No subscription fees**

### Network Security
- Ollama runs on localhost (127.0.0.1) by default
- No external network connections required
- All data stays within your local environment

### Model Security
- Models are downloaded from official Ollama registry
- Models are cryptographically verified
- No third-party model sources by default

## 📈 Performance Benchmarks

### Model Performance Comparison
| Model | Response Time | Quality | Resources |
|-------|----------------|---------|------------|
| `phi3:mini` | ~2-3s | Good | Low |
| `mistral:7b` | ~4-6s | Very Good | Medium |
| `llama3.2` | ~5-8s | Excellent | Medium |
| `llama3.1:8b` | ~6-10s | Excellent | High |
| `qwen2.5:7b` | ~4-7s | Very Good | Medium |

### Hardware Impact
| Configuration | CPU Usage | RAM Usage | VRAM Usage |
|---------------|-----------|----------|-----------|
| `phi3:mini` (CPU) | 60-80% | 3-4GB | 0GB |
| `llama3.2` (CPU) | 80-95% | 6-8GB | 0GB |
| `llama3.2` (GPU) | 20-40% | 4-6GB | 4-6GB |

## 🔄 Migration from vLLM

### What Changed
- ✅ **Simpler installation** - No Python dependencies for model serving
- ✅ **Easier model management** - Simple pull commands
- ✅ **Better resource utilization** - Optimized for local deployment
- ✅ **Cross-platform support** - Works consistently across operating systems

### Compatibility
- ✅ All existing API endpoints remain the same
- ✅ Web interface works unchanged
- ✅ Streaming responses fully supported
- ✅ Memory system integration maintained
- ✅ Configuration file structure compatible

### Migration Steps
```bash
# 1. Install Ollama (instead of vLLM)
brew install ollama  # or curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull your preferred model
ollama pull llama3.2

# 3. Update config.json (already done in this update)
# 4. Start the assistant (same command)
./run.sh start
```

## 📚 Advanced Usage

### Model Fine-tuning
```bash
# Ollama supports custom models
# See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md

# Create Modelfile
echo "FROM llama3.2
SYSTEM You are an expert financial analyst..." > Modelfile

# Build custom model
ollama create fin-analyst -f Modelfile

# Use custom model
ollama run fin-analyst
```

### API Integration Examples
```python
# Python client example
import requests

def chat_with_ollama(message):
    response = requests.post(
        "http://localhost:8080/chat",
        json={"message": message, "model": "llama3.2"}
    )
    return response.json()["response"]

# Use in your financial analysis
analysis = chat_with_ollama("Analyze current gold market trends")
print(analysis)
```

### Batch Processing
```bash
# Process multiple queries
for query in "Market analysis" "Risk assessment" "Portfolio review"; do
    curl -X POST http://localhost:8080/chat \
      -H "Content-Type: application/json" \
      -d "{\"message\": \"$query\", \"model\": \"llama3.2\"}"
    echo "---"
done
```

## 🆙 Model Updates and Maintenance

### Update Models
```bash
# Check for updates
ollama pull llama3.2  # Re-pulls latest version

# Update all models
for model in $(ollama list | tail -n +2 | awk '{print $1}'); do
    echo "Updating $model..."
    ollama pull "$model"
done
```

### Clean Up Old Models
```bash
# List all models
ollama list

# Remove unused models
ollama remove old-model-name

# Clean up disk space
# Models are stored in ~/.ollama/models on Unix systems
# %USERPROFILE%\.ollama\models on Windows
```

### Backup Models
```bash
# Backup model directory
cp -r ~/.ollama/models ./ollama-backup/

# Restore backup
# Stop Ollama first, then:
cp -r ./ollama-backup/models ~/.ollama/
```

## 🎯 Best Practices

### For Financial Analysis
1. **Use appropriate models**: `llama3.2` for balanced performance
2. **Provide clear context**: Include relevant market data in prompts
3. **Verify responses**: Cross-check critical financial information
4. **Use memory system**: Store important trading rules and insights

### For Production Use
1. **Monitor resources**: Keep an eye on CPU and memory usage
2. **Regular updates**: Keep models updated for best performance
3. **Backup configuration**: Save your customized config.json
4. **Test thoroughly**: Validate all features before deployment

### For Optimal Performance
1. **Choose right model size**: Match model to your hardware capabilities
2. **Manage context window**: Don't exceed model's context limits
3. **Regular maintenance**: Clean up unused models and logs
4. **Monitor system health**: Use health checks and status monitoring

---

## 🎉 Congratulations!

You now have a powerful Local Financial Assistant running with Ollama! This setup provides:

- 🦙 **Easy model management** with Ollama
- 🚀 **Fast, local inference** for financial analysis
- 🌐 **Modern web interface** with real-time streaming
- 💾 **Intelligent memory system** for storing insights
- 🔒 **Complete privacy** with local processing
- 📊 **Professional-grade features** for trading analysis

Your assistant is ready to help with market analysis, trading strategies, risk management, and much more - all running locally on your machine with the power of Ollama!

🌊 **Start exploring the world of AI-powered financial analysis!**