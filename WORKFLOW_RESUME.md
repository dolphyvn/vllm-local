# Project Workflow & Architecture Resume

**Last Updated**: 2025-10-28
**Project**: Local Financial Assistant with MRAG System
**URL**: http://ai.vn.aliases.me
**Git Branch**: main (latest commit: a51dcea)

## 🏗️ Current Architecture

### Technology Stack
- **Backend**: Python 3.12 + FastAPI
- **Frontend**: Vanilla JavaScript + CSS + HTML5
- **AI Model**: vLLM with Ollama (Gemma3:4b, Phi-3 Mini)
- **Database**: ChromaDB for vector storage and semantic search
- **Authentication**: Session-based with JWT tokens
- **File Storage**: Local disk storage with base64 encoding

### Core Components

#### 1. **Backend System** (`main.py`)
- **Authentication Manager**: Session-based auth with automatic re-authentication
- **Memory Management**: ChromaDB-based vector storage with semantic search
- **RAG System**: Memory-Augmented Retrieval-Augmented Generation
- **File Upload System**: Multi-format file processing (images, text, documents)
- **Streaming Chat**: Real-time AI responses with Server-Sent Events
- **Knowledge Feeding**: Programmatic API for batch training data ingestion

#### 2. **Frontend Application** (`static/js/app.js`)
- **FinancialAssistantApp**: Main application class
- **File Upload System**: Drag-and-drop, preview, multi-file support
- **Memory Management**: Store/retrieve trading rules and insights
- **Theme System**: Light/dark mode with persistent storage
- **Model Selection**: Interactive dropdown for model switching
- **Real-time Chat**: Streaming responses with typing indicators

#### 3. **Enhanced Systems**
- **RAG Enhancer** (`rag_enhancer.py`): Automatic lesson extraction from corrections
- **Memory System** (`memory.py`): Enhanced context retrieval with personal info continuity
- **Feed XAU Data** (`feed_xau_data.py`): CSV folder processing for trading data

## 🚀 Current Features

### ✅ Implemented Features

1. **Authentication & Security**
   - Session-based authentication with automatic token renewal
   - 401 error handling with seamless re-authentication
   - Secure file upload with authorization

2. **Chat System**
   - Streaming responses from vLLM models
   - Real-time typing indicators
   - Message history and context management
   - Cross-session memory continuity

3. **File Upload System**
   - Multi-format support (images, text, documents)
   - Drag-and-drop interface with preview
   - Base64 encoding for images
   - File content analysis by AI
   - Error handling and validation

4. **Memory Management**
   - Semantic search with ChromaDB
   - Personal information continuity across sessions
   - Category-based memory organization (trading, strategy, risk)
   - Automatic lesson extraction from corrections

5. **Model Management**
   - Interactive model selection dropdown
   - Real-time model switching
   - Model status monitoring
   - Ollama client integration

6. **Knowledge Feeding API**
   - Programmatic batch data ingestion
   - CSV folder processing capabilities
   - XAUUSD trading analysis workflow
   - RESTful API endpoints

### 🔄 Recent Improvements (Last 3 Commits)

1. **Fixed content_type KeyError** (commit: a51dcea)
   - Fixed missing content_type field in frontend
   - Added robust error handling for file data access
   - Implemented safe field access with fallbacks

2. **Enhanced File Upload Functionality** (commit: cf9dc6c)
   - Fixed send button to enable when files are attached without text
   - Enhanced AI file analysis with explicit instructions
   - Improved file context processing

3. **File Upload System** (commit: ad347d3)
   - Comprehensive file and image upload functionality
   - Multi-file support with preview
   - Secure file processing and storage

## 📁 Key File Structure

```
vllm-local/
├── main.py                 # Main FastAPI application
├── memory.py              # ChromaDB memory management
├── rag_enhancer.py        # Enhanced RAG with lesson extraction
├── feed_xau_data.py       # XAUUSD data processing
├── templates/
│   └── index.html         # Main web interface
├── static/
│   ├── js/app.js          # Frontend application
│   └── css/style.css      # Styling and themes
├── uploads/               # File upload storage
├── chroma_db/            # Vector database storage
└── WORKFLOW_RESUME.md     # This documentation
```

## 🔧 Development Workflow

### Current Development State
- **Working Environment**: http://ai.vn.aliases.me
- **Git Status**: Clean (all changes committed and pushed)
- **Database**: ChromaDB with active memory storage
- **Models**: Gemma3:4b (active), Phi-3 Mini (available)

### Running the Application
```bash
# Start the application
python main.py

# The application runs on:
# - Web Interface: http://ai.vn.aliases.me
# - API Endpoints: http://ai.vn.aliases.me/api/
# - Health Check: http://ai.vn.aliases.me/health
```

### Testing File Upload
```bash
# Test upload endpoint (requires auth token)
curl -X POST -F "file=@test.txt" \
  -H "Authorization: Bearer <token>" \
  http://ai.vn.aliases.me/api/upload
```

## 🎯 Current Capabilities

### Chat & AI Features
- ✅ Streaming responses from local AI models
- ✅ Memory-augmented responses with context awareness
- ✅ File analysis (images, documents, text)
- ✅ Personal information continuity
- ✅ Automatic correction learning

### File System Features
- ✅ Multi-format file upload (images, PDFs, text, CSV)
- ✅ Drag-and-drop interface
- ✅ File preview and management
- ✅ Base64 image encoding
- ✅ Content extraction and analysis

### Memory & Knowledge
- ✅ Semantic search across conversations
- ✅ Category-based memory organization
- ✅ Automatic lesson extraction
- ✅ Cross-session continuity
- ✅ Knowledge feeding API

## 🚨 Known Issues & Considerations

### Resolved Issues
- ✅ Send button disabled when only files attached
- ✅ Images not being sent with text messages
- ✅ content_type KeyError in file processing
- ✅ Memory sharing across browser sessions
- ✅ Authentication token expiration handling

### Monitoring Points
- File upload size limits (currently handled by FastAPI defaults)
- ChromaDB storage growth (monitor disk usage)
- Model switching reliability (test with different models)
- Session timeout handling (currently automatic)

## 🔄 Next Development Steps

### Immediate Enhancements
1. **File Upload Improvements**
   - Add file size validation and user feedback
   - Implement file type restrictions
   - Add progress indicators for large files

2. **Memory System Enhancements**
   - Memory cleanup and archival
   - Memory export/import functionality
   - Advanced search filters

3. **UI/UX Improvements**
   - Better loading states
   - Error message improvements
   - Mobile responsiveness enhancements

### Future Features
1. **Advanced Analytics**
   - Conversation statistics
   - Memory usage analytics
   - File analysis reports

2. **Integration Features**
   - External API connections
   - Real-time market data integration
   - Trading strategy backtesting

## 📊 API Endpoints Summary

### Authentication
- `POST /auth/login` - User authentication
- `GET /auth/status` - Session validation

### Chat & Memory
- `POST /chat/stream` - Streaming chat with file support
- `POST /chat` - Non-streaming chat fallback
- `POST /memories` - Store memory
- `GET /memories` - Retrieve memories

### File Management
- `POST /api/upload` - File upload endpoint
- `GET /api/files/{file_id}` - File retrieval

### Knowledge Feeding
- `POST /knowledge/feed` - Batch data ingestion
- `GET /knowledge/status` - System status

### System
- `GET /health` - Health check
- `GET /models` - Available models
- `POST /models/switch` - Model switching

## 🛠️ Development Commands

### Git Workflow
```bash
# Check current status
git status

# View recent changes
git log --oneline -5

# Pull latest changes
git pull origin main

# Push changes
git push origin main
```

### Database Management
```bash
# ChromaDB is stored in ./chroma_db/
# Backup: Copy entire directory
# Reset: Delete directory and restart application
```

### File Storage
```bash
# Uploads stored in ./uploads/
# Monitor: ls -la uploads/
# Cleanup: Remove old files as needed
```

## 📝 Configuration Notes

### Environment Variables
- Application runs with default configuration
- Models served via Ollama integration
- Authentication handled internally

### Port Configuration
- Default port: 8000 (configurable in main.py)
- Production: Configured via reverse proxy

### Memory Settings
- ChromaDB persistence enabled
- Semantic search with embedding model
- Automatic memory enhancement enabled

---

## 🔄 Resume Instructions

When resuming work:

1. **Check Current Status**
   ```bash
   git status
   git log --oneline -3
   python main.py  # Start application
   ```

2. **Verify Functionality**
   - Test authentication at http://ai.vn.aliases.me
   - Check health endpoint: http://ai.vn.aliases.me/health
   - Test file upload with an image
   - Verify memory continuity

3. **Continue Development**
   - Review current TODO items in code
   - Check for any runtime errors in logs
   - Test recent features before adding new ones

4. **Deployment Ready**
   - All changes are committed and pushed
   - Application is running on production URL
   - Database and file storage are persistent

---

## 🎯 Recent Accomplishments Summary

### File Upload System (Latest Work)
1. **Fixed Send Button Logic**: Now enables when files are attached without text
2. **Enhanced AI File Analysis**: Added explicit instructions for proper file content analysis
3. **Resolved content_type KeyError**: Fixed file data structure issues with robust error handling
4. **Multi-format Support**: Images, documents, and text files all processed correctly

### Previous Major Features
1. **Memory-Augmented RAG System**: Cross-session continuity and personal info retention
2. **Knowledge Feeding API**: Programmatic batch data ingestion for training
3. **XAUUSD Trading Analysis**: CSV processing and market prediction capabilities
4. **Model Management**: Interactive model switching with real-time updates

---

**Project Status**: ✅ Production Ready
**Last Tested**: 2025-10-28
**Version**: 1.3.0 (File Upload Complete)
**Next Priority**: File size validation and progress indicators