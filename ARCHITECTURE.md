# System Architecture Documentation

**Last Updated**: 2025-10-28
**Project**: Local Financial Assistant with MRAG System
**URL**: http://ai.vn.aliases.me
**Version**: 1.3.0 (File Upload Complete)

## 📋 Project Overview

**Project Name:** Local Financial Assistant
**Architecture Type:** Memory-Augmented RAG with Programmatic Knowledge Feeding
**Backend:** FastAPI (Python 3.12)
**Frontend:** Vanilla JavaScript + CSS + HTML5
**Vector Database:** ChromaDB
**LLM Integration:** Ollama (Gemma3:4b, Phi-3 Mini)
**Authentication:** Session-based with Automatic Re-authentication
**File Storage**: Local disk with base64 encoding
**Deployment:** Remote Server (`ai.vn.aliases.me`)

---

## 🏗️ System Architecture

### **High-Level Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           External Data Sources                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  ┌─────────────┐         │
│  │   REST API  │  │   CSV Files  │  │  Web Scrapers   │  │  Databases  │         │
│  │   Uploads   │  │   Imports    │  │   Feeders       │  │   Exports  │         │
│  └─────────────┘  └──────────────┘  └─────────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Knowledge Feeding Layer                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Knowledge API Endpoints (REST)                                        │   │
│  │ • POST /api/knowledge/add      • POST /api/knowledge/bulk              │   │
│  │ • POST /api/lessons/add       • POST /api/corrections/add            │   │
│  │ • POST /api/definitions/add   • GET /api/knowledge/stats             │   │
│  │ • DELETE /api/knowledge/clear                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │
│                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Vector Database Layer (ChromaDB)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ • Semantic Search Vectors      • Lesson Storage                     │   │
│  │ • Conversation History         • Correction Tracking                │   │
│  │ • Category-based Organization  • Metadata Tags                     │   │
│  │ • Persistent Storage           • Semantic Similarity              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │
│                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG Processing Layer                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ RAGEnhancer Class                                                    │   │
│  │ • Enhanced Query Processing        • Correction Detection                │   │
│  │ • Lesson Extraction               • Semantic Search Enhancement        │   │
│  │ • Context Building                 • Prompt Engineering              │   │
│  │ • Memory Retrieval                • Lesson Integration              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │
│                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer (FastAPI)                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ FastAPI Application                                                    │   │
│  │ • REST API Endpoints               • WebSocket Streaming          │   │
│  │ • Authentication Middleware         • CORS Configuration            │   │
│  │ • Static File Serving              • Template Rendering           │   │
│  │ • Error Handling                   • Logging                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │
│                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LLM Integration Layer (Ollama)                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ OllamaClient Class                                                    │   │
│  │ • HTTP API Calls                    • Streaming Responses            │   │
│  │ • Model Management                  • Error Handling                │   │
│  │ • Timeout Configuration            • Model Availability Checks    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │
│                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             User Interface Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Web Frontend (HTML/JS/CSS)                                            │   │
│  │ • Authentication UI               • Chat Interface                     │   │
│  │ • Memory Management UI            • System Status Display           │   │
│  │ • Theme Toggle                    • Responsive Design               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### **1. Authentication System (`auth.py`)**
- **Type:** Session-based with Cookie + Token Fallback
- **Features:**
  - SHA-256 password hashing
  - Session token generation and validation
  - Cookie management with proper security
  - Authorization middleware for API protection
- **Endpoints:** `/auth/login`, `/auth/logout`, `/auth/status`

### **2. Memory Management (`memory.py`)**
- **Type:** ChromaDB-based Vector Database
- **Features:**
  - Semantic similarity search
  - Category-based organization
  - Persistent storage
  - Lesson and conversation memory
  - Lazy initialization
- **Collections:** `financial_memory`

### **3. Lesson Management (`lessons.py`)**
- **Type:** Structured Learning System
- **Features:**
  - CRUD operations for lessons
  - Statistical tracking
  - Category filtering
  - Confidence scoring
  - Metadata management

### **4. RAG Enhancement (`rag_enhancer.py`)**
- **Type:** Advanced RAG Processing
- **Features:**
  - Automatic correction detection
  - Lesson extraction from user feedback
  - Enhanced query processing
  - Semantic search improvement
  - Context building

### **5. Knowledge Feeding (`knowledge_feeder.py`)**
- **Type:** Pydantic Models for API
- **Features:**
  - Structured data validation
  - Multiple knowledge types
  - Bulk upload support
  - Category management
  - API response standardization

### **6. LLM Integration (OllamaClient)**
- **Type:** Ollama API Client
- **Features:**
  - HTTP and streaming support
  - Model management
  - Error handling
  - Timeout configuration
  - Model availability checks

---

## 📊 Data Flow Architecture

### **Normal Chat Flow:**
```
User Message → RAGEnhancer → ChromaDB Search → Context Building →
Enhanced Prompt → Ollama → Stream Response → Store Memory/Lesson
```

### **Knowledge Feeding Flow:**
```
External Source → REST API → Validation → ChromaDB Storage →
Vector Embedding → Semantic Index
```

### **Correction Detection Flow:**
```
User Correction → Pattern Matching → Lesson Extraction →
ChromaDB Storage → Future Query Enhancement
```

---

## 🗂️ Database Schema

### **ChromaDB Collections Structure:**

#### **Main Collection: `financial_memory`**
```json
{
  "documents": [
    "User: What is TPO short for?",
    "Assistant: TPO stands for Thyroid-stimulating hormone...",
    "User: no, you're wrong, TPO is short for Time Price Opportunity",
    "Knowledge: TPO = Time Price Opportunity (Correction)",
    "Definition: TPO = Time Price Opportunity"
  ],
  "metadatas": [
    {
      "category": "conversation",
      "timestamp": "2024-10-26T...",
      "confidence": 0.8
    },
    {
      "category": "corrections",
      "topic": "TPO",
      "confidence": 1.0,
      "source": "API Upload"
    }
  ],
  "embeddings": [
    [0.1, 0.2, 0.3, ...], // Vector embeddings
    [0.4, 0.5, 0.6, ...],
    // ... more vectors
  ]
}
```

---

## 🔐 Security Architecture

### **Authentication Flow:**
1. **Initial Request:** Check for session token in localStorage/cookie
2. **Missing Token:** Redirect to login page
3. **Login Process:** Password verification → Session creation → Token generation
4. **Session Storage:** Store token in localStorage and/or secure cookie
5. **API Requests:** Include `Authorization: Bearer <token>` header

### **Security Features:**
- **Password Hashing:** SHA-256 with salt
- **Session Management:** Configurable timeout (default 8 hours)
- **CORS Configuration:** Properly configured for web access
- **Input Validation:** Pydantic models for all API inputs
- **Error Handling:** No sensitive data leakage in error messages

---

## 🚀 Deployment Architecture

### **Server Configuration:**
- **Host:** `ai.vn.aliases.me`
- **Port:** 8080
- **Process Manager:** Uvicorn with auto-reload
- **Persistence:** Local file system for ChromaDB

### **File Structure:**
```
/opt/vllm-local/
├── main.py                    # FastAPI application
├── auth.py                   # Authentication logic
├── memory.py                 # ChromaDB memory management
├── lessons.py                 # Lesson management
├── rag_enhancer.py            # RAG processing
├── knowledge_feeder.py        # API data models
├── config.json                # Configuration file
├── templates/                 # HTML templates
│   ├── index.html             # Main chat interface
│   └── login.html            # Login page
├── static/                    # Static assets
│   ├── js/                   # JavaScript files
│   │   └── app.js           # Main application JS
│   └── css/                  # CSS styles
│       └── style.css        # Main stylesheet
├── chroma_db/                # ChromaDB persistence
└── logs/                    # Application logs
```

### **Environment Variables:**
- **OLLAMA_URL:** Ollama service URL
- **MODEL_NAME:** Default LLM model
- **AUTH_PASSWORD:** Authentication password
- **DB_PATH:** ChromaDB storage path

---

## 🔄 Configuration Management

### **Configuration Structure (`config.json`):**
```json
{
  "default_model": "gemma3:1b",
  "system_prompt": "You are a helpful AI assistant...",
  "auth_settings": {
    "password": "admin123",
    "session_timeout_minutes": 480,
    "cookie_secret": "your-secret-key"
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

### **Runtime Configuration:**
- **Model Loading:** Dynamic from config
- **Database Initialization:** Lazy loading with fallback
- **Service Health:** Continuous monitoring
- **Error Recovery:** Graceful degradation

---

## 📊 Monitoring and Logging

### **Logging Architecture:**
- **Framework:** Python logging module
- **Levels:** INFO, WARNING, ERROR, DEBUG
- **Outputs:** Console + File logs
- **Rotation:** Log file management

### **Health Check Endpoint (`/health`):**
```json
{
  "status": "ok",
  "model": "gemma3:1b",
  "memory_status": "ok",
  "timestamp": "2024-10-26T23:30:00",
  "lessons_available": true,
  "api_status": "healthy"
}
```

### **Knowledge Statistics (`/api/knowledge/stats`):**
```json
{
  "total_entries": 150,
  "entries_by_category": {
    "trading": 45,
    "corrections": 12,
    "definitions": 28,
    "risk_management": 18,
    "general": 47
  },
  "total_lessons": 150,
  "total_corrections": 12,
  "total_definitions": 28
}
```

---

## 🔌 API Endpoint Architecture

### **Authentication Endpoints:**
- `POST /auth/login` - User authentication
- `POST /auth/logout` - Session termination
- `GET /auth/status` - Authentication check

### **Chat Endpoints:**
- `POST /chat` - Standard chat with memory
- `POST /chat/stream` - Streaming chat responses

### **Knowledge Feeding Endpoints:**
- `POST /api/knowledge/add` - Single knowledge entry
- `POST /api/knowledge/bulk` - Bulk knowledge upload
- `POST /api/lessons/add` - Structured lesson addition
- `POST /api/corrections/add` - Correction entry
- `POST /api/definitions/add` - Definition entry
- `GET /api/knowledge/stats` - Knowledge statistics
- `DELETE /api/knowledge/clear` - Clear all knowledge

### **Management Endpoints:**
- `POST /memories` - Store memory entry
- `GET /memories` - Retrieve recent memories
- `POST /lessons` - Add lesson
- `GET /lessons` - Retrieve lessons
- `GET /health` - System health check
- `GET /models` - Available models

### **Static File Endpoints:**
- `/` - Main chat interface
- `/login` - Login page
- `/static/*` - Static assets

---

## 🎯 Workflow Summary

### **Development Workflow:**
1. **Local Development:** Code changes → Git push → Server pull → Auto-reload
2. **Knowledge Feeding:** External data → REST API → ChromaDB storage → Enhanced responses
3. **Model Updates:** Model change → Config update → Restart → Knowledge preserved

### **Data Persistence Strategy:**
- **Conversations:** Temporary, stored in ChromaDB for context
- **Lessons:** Permanent, high-value knowledge with semantic search
- **Corrections:** Permanent, critical fixes with high confidence
- **Definitions:** Permanent, term definitions with examples

### **Scalability Considerations:**
- **Vector Database:** ChromaDB scales to millions of vectors
- **API Rate Limiting:** Session-based authentication prevents abuse
- **Memory Management:** Configurable retention policies
- **Model Flexibility:** Easy model swapping without data loss

---

## 🚀 Future Enhancement Opportunities

### **Potential Improvements:**
1. **Multi-Modal Support:** Add image/document processing
2. **Advanced Analytics:** Knowledge usage analytics and insights
3. **Real-time Feeds:** Market data integration
4. **User Management:** Multiple user support with isolation
5. **API Rate Limiting:** Enhanced security and performance
6. **Automated Validation:** Knowledge quality checking
7. **Export/Import:** Knowledge base portability
8. **Multi-LLM Support:** Model routing based on query type

### **Plugin Architecture:**
- **Knowledge Source Plugins:** Database connectors, web scrapers
- **Processing Plugins:** Document processors, data transformers
- **Integration Plugins:** Third-party service integrations
- **Notification Plugins:** Alert systems, monitoring tools

This architecture provides a robust, scalable, and maintainable foundation for a knowledge-enhanced AI assistant with RAG capabilities.