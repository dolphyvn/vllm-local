# Development Workflow Guide

This document outlines the complete workflow for developing, deploying, and maintaining the Financial Assistant application.

## 🔄 Development Workflow

### **Local Development Setup**

#### **Prerequisites:**
- Python 3.9+
- Git
- Ollama (for local testing)
- Code editor (VS Code recommended)

#### **Setup Commands:**
```bash
# Clone repository
git clone https://github.com/dolphyvn/vllm-local.git
cd vllm-local

# Install dependencies
pip install -r requirements.txt

# Start local Ollama (for testing)
ollama serve

# Start development server
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Access application
# Local: http://localhost:8080
# Production: http://ai.vn.aliases.me
```

### **Code Development Cycle**

1. **Make Changes:**
   ```bash
   # Edit files
   vim main.py
   # or use IDE
   ```

2. **Test Locally:**
   ```bash
   # Test API endpoints
   curl http://localhost:8080/health

   # Test chat functionality
   curl -X POST http://localhost:8080/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is RSI?"}'
   ```

3. **Commit Changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push to Remote:**
   ```bash
   git push origin main
   ```

5. **Deploy to Production:**
   ```bash
   ssh root@bo-X99-F8D
   cd /opt/vllm-local
   git pull origin main
   # Server auto-reloads with --reload flag
   ```

---

## 🚀 Deployment Workflow

### **Production Server Setup**

#### **Server Configuration:**
```bash
# SSH into server
ssh root@bo-X99-F8D

# Navigate to project
cd /opt/vllm-local

# Check Git status
git status

# Pull latest changes
git pull origin main

# Ensure Ollama is running
systemctl status ollama
# or start manually:
# ollama serve &
```

#### **Service Management:**
```bash
# Check running processes
ps aux | grep uvicorn

# Start server with auto-reload
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload &

# Monitor logs
tail -f fastapi.log

# Stop server (if needed)
pkill -f uvicorn
```

#### **Health Monitoring:**
```bash
# Check application health
curl http://ai.vn.aliases.me/health

# Check server resources
htop
df -h
```

---

## 📚 Knowledge Management Workflow

### **Manual Knowledge Addition (UI)**
1. **Access:** `http://ai.vn.aliases.me/`
2. **Login:** Use password `admin123`
3. **Add Memory:** Use Memory Management section in sidebar
4. **Store Corrections:** Correct the AI in chat (automatic detection)

### **Programmatic Knowledge Feeding (API)**

#### **Authentication Setup:**
```bash
# Get session token
TOKEN=$(curl -s -X POST http://ai.vn.aliases.me/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}' | jq -r '.session_token')

# Verify token
curl -H "Authorization: Bearer $TOKEN" \
  http://ai.vn.aliases.me/auth/status
```

#### **Single Knowledge Entry:**
```bash
curl -X POST http://ai.vn.aliases.me/api/knowledge/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "topic": "Support Level",
    "content": "Price level where buying pressure overcomes selling pressure",
    "category": "trading",
    "confidence": 0.9,
    "tags": ["technical", "support", "price"],
    "source": "Trading Manual",
    "priority": 9
  }'
```

#### **Bulk Knowledge Upload:**
```python
import requests

# Authenticate
session = requests.post("http://ai.vn.aliases.me/auth/login",
                       json={"password": "admin123"})
token = session.json()["session_token"]

# Prepare knowledge entries
trading_knowledge = [
    {
        "topic": "Support Level",
        "content": "Price level where buying pressure overcomes selling pressure",
        "category": "trading",
        "confidence": 0.9,
        "tags": ["support", "technical", "price"]
    },
    {
        "topic": "Resistance Level",
        "content": "Price level where selling pressure overcomes buying pressure",
        "category": "trading",
        "confidence": 0.9,
        "tags": ["resistance", "technical", "price"]
    }
]

# Bulk upload
response = requests.post(
    "http://ai.vn.aliases.me/api/knowledge/bulk",
    json={"knowledge_entries": trading_knowledge},
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
)

print(f"Upload result: {response.json()}")
```

#### **Correction Entry:**
```bash
curl -X POST http://ai.vn.aliases.me/api/corrections/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "incorrect_statement": "TPO stands for Thyroid-stimulating hormone",
    "correct_statement": "TPO stands for Time Price Opportunity",
    "topic": "TPO",
    "explanation": "In trading contexts, TPO refers to time and price analysis of opportunities",
    "confidence": 1.0,
    "category": "corrections"
  }'
```

#### **Statistics Monitoring:**
```bash
# Get knowledge statistics
curl -H "Authorization: Bearer $TOKEN" \
  http://ai.vn.aliases.me/api/knowledge/stats

# Monitor in real-time
watch -n 5 'curl -s -H "Authorization: Bearer $TOKEN" http://ai.vn.aliases.me/api/knowledge/stats | jq'
```

---

## 🔧 Configuration Management

### **Configuration File Structure:**
```json
{
  "default_model": "gemma3:1b",
  "system_prompt": "You are a helpful AI assistant...",
  "auth_settings": {
    "password": "admin123",
    "session_timeout_minutes": 480,
    "cookie_secret": "your-secret-key-change-this-in-production"
  },
  "api_settings": {
    "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_timeout": 300,
    "max_tokens": 2048,
    "temperature": 0.7
  },
  "memory_settings": {
    "collection_name": "financial_memory",
    "persist_directory": "./chroma_db",
    "max_memory_age_days": 30
  }
}
```

### **Model Update Workflow:**
```bash
# 1. Update configuration
vim config.json
# Change "default_model": "llama3.2:8b"

# 2. Pull new model (if needed)
ollama pull llama3.2:8b

# 3. Update server
git add config.json
git commit -m "Update default model to llama3.2:8b"
git push origin main

# 4. Deploy to production
ssh root@bo-X99-F8D "cd /opt/vllm-local && git pull origin main"

# 5. Server auto-reloads with new model
```

### **Ollama URL Updates:**
```bash
# If Ollama moves to different server/port:
vim config.json
# Change "ollama_base_url": "http://new-server:11434"

# Test connectivity
curl -s http://new-server:11434/api/tags

# Update and deploy
git add config.json
git commit -m "Update Ollama base URL"
git push origin main
```

---

## 🔍 Debugging Workflow

### **Common Issues and Solutions**

#### **1. Application Not Loading:**
```bash
# Check server logs
tail -f fastapi.log

# Check if server is running
ps aux | grep uvicorn

# Check port availability
lsof -i :8080

# Restart server if needed
pkill -f uvicorn
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

#### **2. Authentication Issues:**
```bash
# Check auth logs
grep -i auth fastapi.log

# Test login endpoint
curl -X POST http://ai.vn.aliases.me/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password":"admin123"}'

# Clear local storage (browser cache issue)
# In browser console:
localStorage.clear()
```

#### **3. Knowledge Not Retrieved:**
```bash
# Check memory logs
grep -i "retrieved.*context" fastapi.log

# Check ChromaDB health
curl http://ai.vn.aliases.me/health

# Test knowledge stats
curl -H "Authorization: Bearer $TOKEN" \
  http://ai.vn.aliases.me/api/knowledge/stats

# Verify ChromaDB data
ls -la chroma_db/
```

#### **4. LLM Connection Issues:**
```bash
# Test Ollama connectivity
curl -s http://127.0.0.1:11434/api/tags

# Check available models
ollama list

# Test model directly
ollama run gemma3:1b "Hello"

# Check Ollama logs
docker logs ollama  # if running in Docker
journalctl -u ollama  # if running as service
```

### **Log Analysis:**
```bash
# Real-time log monitoring
tail -f fastapi.log

# Search for specific errors
grep -i "error" fastapi.log

# Memory-related logs
grep -i "memory\|rag\|lesson" fastapi.log

# Authentication logs
grep -i "auth\|login\|session" fastapi.log

# Performance metrics
grep -i "generated.*response.*length" fastapi.log
```

---

## 🧪 Testing Workflow

### **Manual Testing:**
1. **UI Testing:**
   - Navigate to `http://ai.vn.aliases.me/`
   - Test login/logout functionality
   - Test chat responses
   - Test memory management features

2. **API Testing:**
   - Test all API endpoints
   - Verify error handling
   - Check authentication
   - Validate response formats

3. **Integration Testing:**
   - Test knowledge feeding → Chat responses
   - Verify RAG retrieval
   - Test correction detection
   - Validate semantic search

### **Automated Testing:**
```bash
# Test health endpoint
curl -f http://ai.vn.aliases.me/health

# Test authentication flow
python test_auth_flow.py

# Test knowledge feeding
python test_knowledge_feeding.py

# Test RAG functionality
python test_rag_system.py
```

### **Performance Testing:**
```bash
# Load testing
ab -n 100 -c 10 http://ai.vn.aliases.me/health

# Memory usage monitoring
htop

# Response time testing
time curl http://ai.vn.aliases.me/health
```

---

## 📊 Monitoring and Maintenance

### **Daily Monitoring:**
```bash
# Check application health
curl -s http://ai.vn.aliases.me/health

# Monitor resource usage
htop
df -h

# Check recent logs
tail -50 fastapi.log | grep -E "(ERROR|WARN|✅|❌)"

# Check knowledge base size
curl -s -H "Authorization: Bearer $TOKEN" \
  http://ai.vn.aliases.me/api/knowledge/stats
```

### **Weekly Maintenance:**
```bash
# Clear old logs (keep last 7 days)
find ./logs -name "*.log" -mtime +7 -delete

# Check database size
du -sh chroma_db/

# Update dependencies (if needed)
pip install -r requirements.txt --upgrade

# Backup configuration and data
cp config.json config.json.backup
tar -czf backup_$(date +%Y%m%d).tar.gz chroma_db/
```

### **Monthly Maintenance:**
```bash
# Review knowledge base quality
curl -s -H "Authorization: Bearer $TOKEN" \
  http://ai.vn.aliases.me/api/knowledge/stats | jq .

# Check for unused knowledge
# (Requires custom script)

# Update system packages
apt update && apt upgrade -y

# Review and rotate logs
logrotate /etc/logrotate.d/vllm-local
```

This workflow provides a comprehensive guide for developing, deploying, and maintaining the Financial Assistant application with all its RAG and knowledge feeding capabilities.