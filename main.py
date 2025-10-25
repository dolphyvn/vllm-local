"""
main.py - FastAPI application for local financial AI assistant
Provides chat, memory management, and health endpoints for vLLM integration
"""

import json
import asyncio
import requests
import aiohttp
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
import os
import uuid
import queue
import threading

from memory import MemoryManager
from lessons import LessonManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local Financial Assistant",
    description="AI-powered financial analysis and trading strategy assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files and templates
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    templates = None

# Initialize memory managers
memory_manager = MemoryManager()
lesson_manager = LessonManager()

# Load configuration
def load_config():
    """Load system configuration from config.json"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found. Please create it first.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config.json: {e}")
        raise

config = load_config()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    model: str = config.get("default_model", "phi3")
    memory_context: int = 3
    stream: bool = False

class StreamChatRequest(BaseModel):
    message: str
    model: str = config.get("default_model", "phi3")
    memory_context: int = 3

class MemorizeRequest(BaseModel):
    key: str
    value: str
    category: Optional[str] = "general"

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str
    memory_used: bool = False

class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: str
    memory_status: str

class MemorizeResponse(BaseModel):
    success: bool
    message: str
    timestamp: str

class MemoryEntryResponse(BaseModel):
    key: str
    value: str
    category: str
    timestamp: str

# Lesson-related Pydantic models
class LessonRequest(BaseModel):
    title: str
    content: str
    category: str
    confidence: float = 0.7
    tags: List[str] = []
    source_conversation_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    lesson_id: str
    rating: int  # 1-5
    feedback_text: str = ""
    helpful: bool = True
    user_context: Dict[str, Any] = {}

class CorrectionRequest(BaseModel):
    original_response: str
    corrected_response: str
    correction_reason: str
    conversation_id: str

class LessonResponse(BaseModel):
    success: bool
    lesson_id: str
    message: str
    timestamp: str

class LessonsResponse(BaseModel):
    lessons: List[Dict[str, Any]]
    total_count: int
    timestamp: str

class LessonStatsResponse(BaseModel):
    total_lessons: int
    lessons_by_category: Dict[str, int]
    average_feedback_rating: float
    total_feedback_count: int
    average_effectiveness: float
    total_applications: int
    application_outcomes: Dict[str, int]
    most_effective_categories: List[Dict[str, Any]]

# Ollama API client
class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat completion request to Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the completion

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/api/chat"

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            ollama_messages.append({
                "role": role,
                "content": msg["content"]
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers, timeout=120) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get("message", {}).get("content", "")

        except aiohttp.ClientError as e:
            logger.error(f"Ollama API request failed: {e}")
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from Ollama: {e}")
            raise HTTPException(status_code=500, detail="Invalid response from model service")

    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """
        Send streaming chat completion request to Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the completion

        Yields:
            Token chunks as they arrive
        """
        url = f"{self.base_url}/api/chat"

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            ollama_messages.append({
                "role": role,
                "content": msg["content"]
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers, timeout=120) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        if line:
                            try:
                                line_str = line.decode('utf-8').strip()
                                if line_str:
                                    chunk = json.loads(line_str)
                                    if chunk.get("done"):
                                        break
                                    if "message" in chunk and "content" in chunk["message"]:
                                        content = chunk["message"]["content"]
                                        if content:
                                            yield content
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientError as e:
            logger.error(f"Ollama streaming request failed: {e}")
            yield f"ERROR: Streaming failed - {str(e)}"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"ERROR: {str(e)}"

    def check_model(self) -> bool:
        """
        Check if the specified model is available in Ollama

        Returns:
            True if model is available, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]
            return self.model.split(":")[0] in model_names

        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models in Ollama

        Returns:
            List of available model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            models = response.json().get("models", [])
            return [model.get("name", "") for model in models]

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

# Initialize Ollama client
ollama_client = OllamaClient(
    base_url=config.get("api_settings", {}).get("ollama_base_url", "http://localhost:11434"),
    model=config.get("default_model", "llama3.2")
)

def build_contextual_prompt(user_message: str, memory_context: int = 3, lesson_context: int = 2) -> List[Dict[str, str]]:
    """
    Build a prompt with system context, memory, and user message

    Args:
        user_message: The user's input message
        memory_context: Number of memory entries to include
        lesson_context: Number of lessons to include

    Returns:
        List of message dictionaries for the model
    """
    messages = []

    # System prompt from config
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    messages.append({"role": "system", "content": system_prompt})

    # Add user profile context
    user_profile = {
        "risk_profile": config.get("risk_profile", "moderate"),
        "default_pair": config.get("default_pair", "XAUUSD"),
        "language": config.get("language", "English")
    }
    profile_context = f"<user_profile>\nRisk Profile: {user_profile['risk_profile']}\nDefault Pair: {user_profile['default_pair']}\nLanguage: {user_profile['language']}\n</user_profile>"
    messages.append({"role": "system", "content": profile_context})

    # Retrieve relevant memory and lessons
    try:
        # Get combined context (conversations + lessons)
        context_data = memory_manager.get_combined_context(user_message, memory_context, lesson_context)

        # Add conversation memories
        if context_data["conversations"]:
            memory_context_str = "<recent_context>\n"
            for i, entry in enumerate(context_data["conversations"], 1):
                memory_context_str += f"Memory {i}: {entry}\n"
            memory_context_str += "</recent_context>"
            messages.append({"role": "system", "content": memory_context_str})

        # Add lesson memories
        if context_data["lessons"]:
            lesson_context_str = "<learned_lessons>\n"
            for i, lesson in enumerate(context_data["lessons"], 1):
                lesson_context_str += f"Lesson {i}: {lesson}\n"
            lesson_context_str += "</learned_lessons>\n"
            lesson_context_str += "IMPORTANT: Apply these lessons to improve your analysis and reasoning. Use them to avoid past mistakes and incorporate successful strategies.\n"
            messages.append({"role": "system", "content": lesson_context_str})

        logger.info(f"Retrieved context: {len(context_data['conversations'])} memories, {len(context_data['lessons'])} lessons")

    except Exception as e:
        logger.warning(f"Failed to retrieve context: {e}")

    # Add user message
    messages.append({"role": "user", "content": user_message})

    return messages

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes user messages and returns AI responses

    Args:
        request: ChatRequest containing message and parameters

    Returns:
        ChatResponse with AI response and metadata
    """
    logger.info(f"Received chat request: {request.message[:100]}...")

    try:
        # Build contextual prompt with memory
        messages = build_contextual_prompt(request.message, request.memory_context)

        # Get response from Ollama
        ai_response = await ollama_client.chat_completion(messages)

        # Store conversation in memory
        try:
            memory_manager.add_memory(request.message, ai_response)
            memory_used = True
        except Exception as e:
            logger.warning(f"Failed to store conversation in memory: {e}")
            memory_used = False

        # Return response
        response = ChatResponse(
            response=ai_response,
            model=request.model,
            timestamp=datetime.now().isoformat(),
            memory_used=memory_used
        )

        logger.info(f"Generated response of length: {len(ai_response)}")
        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: StreamChatRequest):
    """
    Streaming chat endpoint that processes user messages and returns AI responses in real-time

    Args:
        request: StreamChatRequest containing message and parameters

    Returns:
        StreamingResponse with token-by-token AI response
    """
    logger.info(f"Received streaming chat request: {request.message[:100]}...")

    async def generate_tokens():
        """Generate and stream tokens from vLLM"""
        accumulated_response = ""
        message_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()

        try:
            # Build contextual prompt with memory
            messages = build_contextual_prompt(request.message, request.memory_context)

            # Send initial chunk with metadata
            initial_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": start_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Stream response from Ollama
            async for token in ollama_client.chat_completion_stream(messages):
                accumulated_response += token

                chunk = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": start_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send final chunk
            final_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": start_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # vLLM doesn't provide token count
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            # Store complete conversation in memory
            try:
                memory_manager.add_memory(request.message, accumulated_response)
                logger.info(f"Stored streaming conversation in memory")
            except Exception as e:
                logger.warning(f"Failed to store streaming conversation in memory: {e}")

            logger.info(f"Generated streaming response of length: {len(accumulated_response)}")

        except Exception as e:
            logger.error(f"Streaming chat endpoint error: {e}")
            error_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": start_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\nERROR: {str(e)}"},
                    "finish_reason": "stop"
                }],
                "error": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/memorize", response_model=MemorizeResponse)
async def memorize_endpoint(request: MemorizeRequest):
    """
    Store important information in persistent memory

    Args:
        request: MemorizeRequest with key-value pair

    Returns:
        MemorizeResponse indicating success/failure
    """
    logger.info(f"Memorizing: {request.key}")

    try:
        # Format memory entry
        memory_entry = f"{request.key}: {request.value}"

        # Store in memory
        memory_manager.add_memory(
            user_input=f"Store memory: {request.key}",
            model_reply=request.value,
            metadata={"category": request.category, "type": "explicit_memory"}
        )

        return MemorizeResponse(
            success=True,
            message=f"Successfully stored memory for key: {request.key}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Memorize endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """
    Health check endpoint

    Returns:
        HealthResponse with system status
    """
    try:
        # Check memory system
        memory_status = "ok" if memory_manager.is_healthy() else "error"

        # Check Ollama model availability
        model_available = ollama_client.check_model()
        model_status = config.get("default_model", "llama3.2") if model_available else "Model not found"

        return HealthResponse(
            status="ok",
            model=model_status,
            timestamp=datetime.now().isoformat(),
            memory_status=memory_status
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=Dict[str, Any])
async def list_models_endpoint():
    """
    List available Ollama models

    Returns:
        Dictionary with available models and current model
    """
    try:
        available_models = ollama_client.list_models()
        current_model = config.get("default_model", "llama3.2")
        model_available = ollama_client.check_model()

        return {
            "current_model": current_model,
            "model_available": model_available,
            "available_models": available_models,
            "total_models": len(available_models)
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def ui_endpoint(request: Request):
    """
    Serve the main web UI

    Returns:
        HTML page with the web interface
    """
    if templates and os.path.exists("templates/index.html"):
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Fallback basic HTML if template not found
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Local Financial Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .api-endpoints { background: #f8f9fa; padding: 15px; border-radius: 5px; }
                .endpoint { margin: 5px 0; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Local Financial Assistant</h1>
                <div class="status">
                    ✅ API is running successfully!
                </div>
                <div class="api-endpoints">
                    <h3>Available Endpoints:</h3>
                    <div class="endpoint">POST /chat - Send chat messages</div>
                    <div class="endpoint">POST /memorize - Store information in memory</div>
                    <div class="endpoint">GET /health - Check system health</div>
                    <div class="endpoint">GET /docs - Interactive API documentation</div>
                </div>
                <p style="text-align: center; margin-top: 30px; color: #666;">
                    Web UI template not found. Please ensure the templates directory exists.
                </p>
            </div>
        </body>
        </html>
        """)

@app.get("/memories", response_model=List[MemoryEntryResponse])
async def get_memories_endpoint(n: int = 10):
    """
    Get recent memories for the web UI

    Args:
        n: Number of memories to return

    Returns:
        List of recent memory entries
    """
    try:
        memories = memory_manager.get_recent_memories(n)

        formatted_memories = []
        for memory in memories:
            # Extract key-value from explicit memories
            if "Store memory:" in memory["text"]:
                parts = memory["text"].split("Store memory:", 1)[1].strip()
                if ":" in parts:
                    key, value = parts.split(":", 1)
                    formatted_memories.append(MemoryEntryResponse(
                        key=key.strip(),
                        value=value.strip(),
                        category=memory["metadata"].get("category", "general"),
                        timestamp=memory["metadata"].get("timestamp", datetime.now().isoformat())
                    ))

        return formatted_memories

    except Exception as e:
        logger.error(f"Failed to get memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Local Financial Assistant API")
    logger.info(f"Configuration loaded: {config.get('default_model', 'llama3.2')} model")
    logger.info("Memory system initialized")

    # Check Ollama availability
    try:
        model_available = ollama_client.check_model()
        if model_available:
            logger.info(f"✅ Ollama model '{config.get('default_model', 'llama3.2')}' is available")
        else:
            available_models = ollama_client.list_models()
            if available_models:
                logger.warning(f"⚠️  Model '{config.get('default_model', 'llama3.2')}' not found. Available models: {', '.join(available_models)}")
                logger.info("Please update config.json with an available model or pull the required model using 'ollama pull <model_name>'")
            else:
                logger.error("❌ No Ollama models found. Please install Ollama and pull a model using 'ollama pull <model_name>'")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        logger.info("Please ensure Ollama is running on http://localhost:11434")

    # Check if UI files are available
    if os.path.exists("templates/index.html"):
        logger.info("Web UI template found - web interface available at http://localhost:8080")
    else:
        logger.warning("Web UI template not found - only API endpoints available")

    if os.path.exists("static"):
        logger.info("Static files directory found")
    else:
        logger.warning("Static files directory not found")

# ==================== LESSON MANAGEMENT ENDPOINTS ====================

@app.post("/lessons", response_model=LessonResponse)
async def add_lesson(request: LessonRequest):
    """
    Add a new lesson to the system
    """
    try:
        logger.info(f"Adding new lesson: {request.title}")

        # Add to structured database
        lesson_id = lesson_manager.add_lesson(
            title=request.title,
            content=request.content,
            category=request.category,
            confidence=request.confidence,
            source_conversation_id=request.source_conversation_id,
            tags=request.tags
        )

        # Add to semantic memory for retrieval
        memory_manager.add_lesson_memory(
            lesson_title=request.title,
            lesson_content=request.content,
            category=request.category,
            confidence=request.confidence,
            tags=request.tags,
            source_conversation=request.source_conversation_id
        )

        return LessonResponse(
            success=True,
            lesson_id=lesson_id,
            message=f"Lesson '{request.title}' added successfully",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to add lesson: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add lesson: {str(e)}")

@app.get("/lessons", response_model=LessonsResponse)
async def get_lessons(query: str = "", category: str = "", limit: int = 10):
    """
    Retrieve lessons based on query and optional category
    """
    try:
        logger.info(f"Retrieving lessons - Query: {query[:50]}, Category: {category}")

        # Get lessons from structured database
        db_lessons = lesson_manager.get_relevant_lessons(
            query_text=query,
            category=category if category else None,
            max_lessons=limit
        )

        # Get lessons from semantic memory
        semantic_lessons = memory_manager.search_lessons(
            query=query,
            category=category if category else None,
            n=limit
        )

        # Combine and deduplicate lessons
        all_lessons = []
        seen_lesson_ids = set()

        # Add database lessons
        for lesson in db_lessons:
            if lesson['id'] not in seen_lesson_ids:
                all_lessons.append(lesson)
                seen_lesson_ids.add(lesson['id'])

        # Add semantic lessons (converted from memory format)
        for lesson in semantic_lessons:
            metadata = lesson.get('metadata', {})
            lesson_id = f"semantic_{metadata.get('title', '')}_{metadata.get('timestamp', '')}"
            if lesson_id not in seen_lesson_ids:
                all_lessons.append({
                    'id': lesson_id,
                    'title': metadata.get('title', 'Unknown'),
                    'content': lesson['document'],
                    'category': metadata.get('category', 'general'),
                    'confidence': metadata.get('confidence', 0.5),
                    'created_at': metadata.get('timestamp', ''),
                    'source': 'semantic_memory'
                })
                seen_lesson_ids.add(lesson_id)

        return LessonsResponse(
            lessons=all_lessons[:limit],
            total_count=len(all_lessons),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to retrieve lessons: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve lessons: {str(e)}")

@app.post("/lessons/{lesson_id}/feedback", response_model=dict)
async def add_lesson_feedback(lesson_id: str, request: FeedbackRequest):
    """
    Add feedback for a lesson
    """
    try:
        logger.info(f"Adding feedback for lesson {lesson_id}: {request.rating}/5")

        feedback_id = lesson_manager.add_feedback(
            lesson_id=lesson_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            helpful=request.helpful,
            user_context=request.user_context
        )

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback added successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to add feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add feedback: {str(e)}")

@app.post("/corrections", response_model=dict)
async def add_correction(request: CorrectionRequest):
    """
    Add a correction with derived lesson
    """
    try:
        logger.info(f"Adding correction: {request.correction_reason[:50]}...")

        # Add correction to database
        correction_id = lesson_manager.add_correction(
            original_response=request.original_response,
            corrected_response=request.corrected_response,
            correction_reason=request.correction_reason,
            lesson_derived=request.correction_reason,  # Use reason as lesson for now
            conversation_id=request.conversation_id
        )

        # Extract and add as a lesson
        lesson_id = lesson_manager.add_lesson(
            title=f"Correction: {request.correction_reason[:50]}...",
            content=f"Original: {request.original_response}\n\nCorrected: {request.corrected_response}\n\nLesson: {request.correction_reason}",
            category="correction",
            confidence=0.8,  # High confidence as this is user-validated
            source_conversation_id=request.conversation_id,
            tags=["correction", "user-feedback", "improvement"]
        )

        # Add to semantic memory
        memory_manager.add_lesson_memory(
            lesson_title=f"Correction: {request.correction_reason[:50]}...",
            lesson_content=f"Lesson learned: {request.correction_reason}. Corrected response: {request.corrected_response}",
            category="correction",
            confidence=0.8,
            tags=["correction", "user-feedback", "improvement"],
            source_conversation=request.conversation_id
        )

        return {
            "success": True,
            "correction_id": correction_id,
            "lesson_id": lesson_id,
            "message": "Correction and derived lesson added successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to add correction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add correction: {str(e)}")

@app.get("/lessons/stats", response_model=LessonStatsResponse)
async def get_lesson_statistics():
    """
    Get comprehensive lesson statistics
    """
    try:
        stats = lesson_manager.get_lesson_statistics()
        return LessonStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get lesson statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/lessons/{lesson_id}/apply", response_model=dict)
async def record_lesson_application(lesson_id: str, conversation_id: str,
                                   application_context: str, outcome: str,
                                   effectiveness_rating: int = None):
    """
    Record when a lesson is applied and its effectiveness
    """
    try:
        logger.info(f"Recording lesson application: {lesson_id} -> {outcome}")

        application_id = lesson_manager.record_lesson_application(
            lesson_id=lesson_id,
            conversation_id=conversation_id,
            application_context=application_context,
            outcome=outcome,
            effectiveness_rating=effectiveness_rating
        )

        return {
            "success": True,
            "application_id": application_id,
            "message": f"Lesson application recorded with outcome: {outcome}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to record lesson application: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record application: {str(e)}")

# ==================== HEALTH CHECK WITH LESSONS ====================

@app.get("/health", response_model=dict)
async def health_check():
    """
    Comprehensive health check including lessons system
    """
    try:
        ollama_status = "ok"
        model_available = True

        # Check Ollama
        try:
            response = requests.get(f"{config.get('api_settings', {}).get('ollama_base_url', 'http://localhost:11434')}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if not models:
                    ollama_status = "no_models"
                    model_available = False
            else:
                ollama_status = "error"
                model_available = False
        except:
            ollama_status = "offline"
            model_available = False

        # Check memory systems
        memory_healthy = memory_manager.is_healthy()
        lessons_healthy = lesson_manager.is_healthy()

        # Get lesson stats if healthy
        lesson_stats = {}
        if lessons_healthy:
            lesson_stats = lesson_manager.get_lesson_statistics()

        return {
            "status": "healthy",
            "model": config.get("default_model", "phi3:latest"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_status": "ok" if memory_healthy else "error",
            "lessons_status": "ok" if lessons_healthy else "error",
            "lesson_stats": lesson_stats,
            "services": {
                "ollama": ollama_status,
                "model_available": model_available,
                "chroma": "ok" if memory_healthy else "error",
                "sqlite": "ok" if lessons_healthy else "error"
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Local Financial Assistant API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)