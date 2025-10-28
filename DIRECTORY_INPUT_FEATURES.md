# Directory Input Features - RAG MT5 Integration

## 🎯 New Features Added

All integration scripts now accept flexible directory input through multiple methods:

### **1. Command Line Arguments**

#### **Main Integration Script:**
```bash
# Basic usage with directory
python3 integrate_rag_mt5_data.py --export-path /path/to/your/data

# All options with directory
python3 integrate_rag_mt5_data.py \
  --export-path /home/user/trading/exports \
  --mode monitor \
  --interval 10 \
  --base-url http://your-server.com \
  --log-file ./custom_processed.log

# List files only
python3 integrate_rag_mt5_data.py --list-files --export-path ./data
```

#### **Scheduler Script:**
```bash
# Monitor custom directory
python3 schedule_rag_feeding.py --export-path /path/to/mt5/exports

# Full customization
python3 schedule_rag_feeding.py \
  --export-path /home/user/trading/data \
  --base-url http://ai.vn.aliases.me \
  --log-file ./trading_processed.log
```

#### **Demo Script:**
```bash
# Test with custom directory
python3 demo_rag_integration.py --export-path /path/to/trading/data

# Create sample data in custom directory
python3 demo_rag_integration.py --create-sample --export-path ./test_data
```

### **2. Interactive Mode**

When no directory is specified, the system will prompt:

```bash
$ python3 integrate_rag_mt5_data.py
🔍 RAG MT5 Data Integration
========================================
Enter directory path for RAG CSV files [./data]: /home/user/my_trading_data
```

### **3. Directory Validation**

The system automatically:

- ✅ **Validates directory exists**
- ✅ **Creates directory if it doesn't exist** (with user confirmation in interactive mode)
- ✅ **Normalizes to absolute path**
- ✅ **Lists CSV files with details** (size, modification date)
- ✅ **Identifies RAG files** (files starting with `RAG_`)

### **4. File Discovery**

The system shows comprehensive directory information:

```
📊 Directory: /opt/works/personal/vllm-local/data
   Total CSV files: 3
   RAG files found: 1
   Latest RAG file: RAG_XAUUSD_2025.10.28.csv
   API Server: http://localhost:8080
   Log file: ./processed_rag_files.log
```

### **5. File Listing Feature**

List all CSV files in a directory with detailed information:

```bash
$ python3 integrate_rag_mt5_data.py --list-files --export-path ./data

📁 CSV files in '/opt/works/personal/vllm-local/data':
   1. RAG_XAUUSD_2025.10.28.csv      (872 bytes, 2025-10-28 13:17)
   2. XAUUSD-2025.10.21.csv          (86,591 bytes, 2025-10-26 18:02)
   3. XAUUSD-2025.10.21T.csv         (421 bytes, 2025-10-26 18:03)
```

## 🛠️ Technical Improvements

### **Enhanced Class Constructors**

All classes now accept directory parameters:

```python
# RAGMT5Integrator
integrator = RAGMT5Integrator(
    base_url="http://localhost:8080",
    password="admin123",
    export_path="./custom_data",  # 🆕 Custom directory
    log_file="./custom_processed.log"  # 🆕 Custom log file
)

# RAGFeedingScheduler
scheduler = RAGFeedingScheduler(
    base_url="http://localhost:8080",
    password="admin123",
    export_path="./custom_data",  # 🆕 Custom directory
    log_file="./custom_processed.log"  # 🆕 Custom log file
)
```

### **Directory Validation Functions**

```python
def validate_directory(path: str) -> str:
    """Validate and normalize directory path"""
    if not os.path.exists(path):
        # Interactive confirmation or auto-create
        if user_confirms():
            os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def list_csv_files(directory: str) -> List[str]:
    """List all CSV files in directory"""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]
```

### **Enhanced Error Handling**

- ✅ **Graceful handling of missing directories**
- ✅ **User-friendly error messages**
- ✅ **Automatic directory creation options**
- ✅ **Path normalization and validation**

## 📋 Use Case Examples

### **Multiple Trading Symbols**

```bash
# Monitor different symbol directories
python3 integrate_rag_mt5_data.py --mode monitor --export-path ./data/xauusd
python3 integrate_rag_mt5_data.py --mode monitor --export-path ./data/eurusd
python3 integrate_rag_mt5_data.py --mode monitor --export-path ./data/btcusd
```

### **Production vs Development**

```bash
# Development with local data
python3 integrate_rag_mt5_data.py --export-path ./dev_data

# Production with live data
python3 integrate_rag_mt5_data.py \
  --export-path /production/mt5/exports \
  --base-url http://ai.vn.aliases.me
```

### **Testing and Validation**

```bash
# Create sample data for testing
python3 demo_rag_integration.py --create-sample --export-path ./test_data

# Test integration with test data
python3 demo_rag_integration.py --export-path ./test_data
```

### **Log File Management**

```bash
# Use custom log files for different environments
python3 integrate_rag_mt5_data.py \
  --export-path ./prod_data \
  --log-file ./prod_processed.log

python3 integrate_rag_mt5_data.py \
  --export-path ./test_data \
  --log-file ./test_processed.log
```

## 🔧 Configuration Flexibility

### **All Parameters Now Configurable:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--export-path` | `./data` | Directory containing RAG CSV files |
| `--log-file` | `./processed_rag_files.log` | File tracking processed entries |
| `--base-url` | `http://localhost:8080` | Financial Assistant API URL |
| `--password` | `admin123` | API authentication password |
| `--mode` | `once` | `once` or `monitor` |
| `--interval` | `5` | Monitoring interval in minutes |

### **Environment-Specific Configurations**

```bash
# Local development
python3 integrate_rag_mt5_data.py \
  --export-path ./dev_data \
  --base-url http://localhost:8080

# Staging environment
python3 integrate_rag_mt5_data.py \
  --export-path ./staging_data \
  --base-url http://staging.example.com

# Production environment
python3 integrate_rag_mt5_data.py \
  --export-path /prod/mt5/exports \
  --base-url http://ai.vn.aliases.me \
  --log-file /var/log/rag_integration.log
```

## 🎉 Benefits

### **For Users:**
- 🎯 **Flexible directory management** - Use any folder structure
- 🔍 **Easy file discovery** - See what files are available
- 📁 **Organized data management** - Separate logs for different environments
- 🚀 **Quick setup** - No hardcoded paths

### **For Developers:**
- 🔧 **Configurable integration** - Easy to adapt to different setups
- 🧪 **Better testing** - Use test directories without affecting production
- 📊 **Environment separation** - Different configs for dev/staging/prod
- 🛠️ **Enhanced debugging** - Clear file listing and validation

---

**All your RAG MT5 integration scripts now support flexible directory input!** 🎉