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
from rag_enhancer import RAGEnhancer
from auth import AuthManager, get_current_user
from knowledge_feeder import (
    KnowledgeEntry, BulkKnowledgeRequest, LessonEntry,
    CorrectionEntry, DefinitionEntry, ApiResponse, KnowledgeStats,
    KnowledgeCategory
)

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

# Initialize managers with config
memory_manager = MemoryManager()
lesson_manager = LessonManager()
rag_enhancer = RAGEnhancer(memory_manager)

# Initialize authentication manager
auth_settings = config.get("auth_settings", {})
auth_manager = AuthManager(
    password=auth_settings.get("password", "admin123"),
    session_timeout_minutes=auth_settings.get("session_timeout_minutes", 480),
    cookie_secret=auth_settings.get("cookie_secret", "default-secret")
)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    session_token: Optional[str] = None

class LogoutResponse(BaseModel):
    success: bool
    message: str

class AuthStatusResponse(BaseModel):
    authenticated: bool
    message: str

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
                "num_predict": kwargs.get("max_tokens", 4096)
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers, timeout=300) as response:
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


def build_enhanced_contextual_prompt(user_message: str, context_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build an enhanced prompt with improved RAG context

    Args:
        user_message: The user's input message
        context_data: Enhanced context data from RAG enhancer

    Returns:
        List of message dictionaries for the model
    """
    messages = []

    # Enhanced system prompt with RAG instructions
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    enhanced_system_prompt = f"""{system_prompt}

You have access to retrieved context from previous conversations and learned lessons. Use this information to provide more accurate and informed responses. Pay special attention to corrections and definitions that have been previously validated."""

    messages.append({"role": "system", "content": enhanced_system_prompt})

    # Add conversation memories with better formatting
    if context_data.get("conversations"):
        memory_context_str = "<previous_conversations>\n"
        for i, entry in enumerate(context_data["conversations"], 1):
            memory_context_str += f"Previous Conversation {i}: {entry}\n"
        memory_context_str += "</previous_conversations>\n"
        messages.append({"role": "system", "content": memory_context_str})

    # Add lesson memories with emphasis on corrections
    if context_data.get("lessons"):
        lesson_context_str = "<learned_lessons_and_corrections>\n"
        for i, lesson in enumerate(context_data["lessons"], 1):
            lesson_context_str += f"Lesson {i}: {lesson}\n"
        lesson_context_str += "</learned_lessons_and_corrections>\n"
        lesson_context_str += "CRITICAL: Pay close attention to these lessons, especially corrections. They contain validated information that should override your general knowledge. Use corrections to avoid repeating mistakes and provide accurate definitions.\n"
        messages.append({"role": "system", "content": lesson_context_str})

    # Enhanced instructions for personal information context
    if context_data.get("conversations"):
        # Check if any conversation contains personal information
        has_personal_info = any("name" in conv.lower() or "call me" in conv.lower() or "i am" in conv.lower() for conv in context_data["conversations"])
        if has_personal_info:
            personal_info_instruction = "\nATTENTION: Previous conversations contain personal information about the user (name, preferences, etc.). Use this information to provide personalized responses and remember details about the user across sessions.\n"
            messages.append({"role": "system", "content": personal_info_instruction})

        logger.info(f"Enhanced RAG applied: {len(context_data.get('conversations', []))} memories, {len(context_data.get('lessons', []))} lessons")

    # Add user message
    messages.append({"role": "user", "content": user_message})

    return messages

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint that processes user messages and returns AI responses

    Args:
        request: ChatRequest containing message and parameters
        http_request: FastAPI request object for authentication

    Returns:
        ChatResponse with AI response and metadata
    """
    # Check authentication
    get_current_user(auth_manager, http_request)
    logger.info(f"Received chat request: {request.message[:100]}...")

    try:
        # Enhanced RAG context retrieval
        context_data = rag_enhancer.enhance_query_with_rag(request.message, max_context=request.memory_context)

        # Build contextual prompt with enhanced memory
        messages = build_enhanced_contextual_prompt(request.message, context_data)

        # Get response from Ollama
        ai_response = await ollama_client.chat_completion(messages)

        # Store conversation in memory
        try:
            memory_manager.add_memory(request.message, ai_response)
            memory_used = True

            # Check if this is a correction and store as lesson
            try:
                lesson_stored = rag_enhancer.store_correction_as_lesson(request.message, ai_response)
                if lesson_stored:
                    logger.info("✅ Correction detected and stored as lesson in regular chat")
            except Exception as e:
                logger.warning(f"Failed to store correction as lesson in regular chat: {e}")

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
async def chat_stream_endpoint(request: StreamChatRequest, http_request: Request):
    """
    Streaming chat endpoint that processes user messages and returns AI responses in real-time

    Args:
        request: StreamChatRequest containing message and parameters
        http_request: FastAPI request object for authentication

    Returns:
        StreamingResponse with token-by-token AI response
    """
    # Check authentication
    get_current_user(auth_manager, http_request)
    logger.info(f"Received streaming chat request: {request.message[:100]}...")

    async def generate_tokens():
        """Generate and stream tokens from vLLM"""
        accumulated_response = ""
        message_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()

        try:
            # Enhanced RAG context retrieval
            context_data = rag_enhancer.enhance_query_with_rag(request.message, max_context=request.memory_context)

            # Build contextual prompt with enhanced memory
            messages = build_enhanced_contextual_prompt(request.message, context_data)

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

                # Check if this is a correction and store as lesson
                try:
                    lesson_stored = rag_enhancer.store_correction_as_lesson(request.message, accumulated_response)
                    if lesson_stored:
                        logger.info("✅ Correction detected and stored as lesson")
                except Exception as e:
                    logger.warning(f"Failed to store correction as lesson: {e}")

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

@app.post("/auth/login")
async def login_endpoint(request: LoginRequest):
    """
    Authenticate user and create session

    Args:
        request: LoginRequest with password
        response: FastAPI response for setting cookies

    Returns:
        LoginResponse with authentication result
    """
    logger.info("Login attempt received")

    try:
        logger.info(f"Login attempt received with password length: {len(request.password)}")
        logger.info(f"Auth manager initialized: {bool(auth_manager)}")
        logger.info(f"Password from config loaded: {bool(auth_manager.password_hash)}")

        if auth_manager.verify_password(request.password):
            session_token = auth_manager.create_session()

            # Create response data with session token
            response_data = {
                "success": True,
                "message": "Authentication successful",
                "session_token": session_token
            }

            try:
                # Try to set cookie in response
                from fastapi.responses import Response
                actual_response = Response(
                    content=json.dumps(response_data),
                    media_type="application/json"
                )
                auth_manager.set_auth_cookie(actual_response, session_token)
                logger.info(f"User authenticated successfully with cookie")
                return actual_response
            except Exception as cookie_error:
                # Fallback: return response without cookie if cookie setting fails
                logger.warning(f"Cookie setting failed, using token-only auth: {cookie_error}")
                logger.info(f"User authenticated successfully (token-only)")
                return response_data
        else:
            logger.warning("Invalid password attempt")
            return {
                "success": False,
                "message": "Invalid password"
            }

    except Exception as e:
        logger.error(f"Login error: {e}")
        return {
            "success": False,
            "message": "Authentication failed"
        }

@app.post("/auth/logout")
async def logout_endpoint(request: Request):
    """
    Logout user and remove session

    Args:
        request: FastAPI request object

    Returns:
        LogoutResponse with logout result
    """
    logger.info("Logout attempt received")

    try:
        session_token = auth_manager.extract_token_from_request(request)
        if session_token and auth_manager.remove_session(session_token):
            logger.info("User logged out successfully")
            return {
                "success": True,
                "message": "Logged out successfully"
            }
        else:
            logger.warning("No valid session found for logout")
            return {
                "success": False,
                "message": "No valid session found"
            }

    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {
            "success": False,
            "message": "Logout failed"
        }

@app.get("/auth/status")
async def auth_status_endpoint(request: Request):
    """
    Check authentication status

    Args:
        request: FastAPI request object

    Returns:
        AuthStatusResponse with authentication status
    """
    try:
        session_token = auth_manager.extract_token_from_request(request)
        if session_token and auth_manager.validate_session(session_token):
            return {
                "authenticated": True,
                "message": "User is authenticated"
            }
        else:
            return {
                "authenticated": False,
                "message": "User is not authenticated"
            }

    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return {
            "authenticated": False,
            "message": "Authentication status check failed"
        }

@app.get("/debug/version")
async def debug_version_endpoint():
    """Debug endpoint to check version info"""
    return {
        "backend_version": "2.0.0",
        "auth_system": "enabled",
        "token_auth_fallback": "enabled",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/config")
async def debug_config_endpoint():
    """
    Debug endpoint to check configuration loading
    """
    try:
        return {
            "auth_config_loaded": bool(config.get("auth_settings")),
            "auth_settings": {
                "password_present": bool(auth_settings.get("password")),
                "session_timeout": auth_settings.get("session_timeout_minutes"),
                "cookie_secret_present": bool(auth_settings.get("cookie_secret"))
            },
            "auth_manager_initialized": bool(auth_manager),
            "active_sessions": auth_manager.get_session_count()
        }
    except Exception as e:
        return {"error": str(e)}

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

@app.post("/models/switch", response_model=Dict[str, Any])
async def switch_model_endpoint(request: Request, model_name: str = None):
    """
    Switch to a different model
    Args:
        request: FastAPI request object for authentication
        model_name: New model name (can be passed in form data)
    Returns:
        Dictionary with switch result
    """
    # Check authentication
    get_current_user(auth_manager, request)

    try:
        # Get model name from form data or JSON
        if not model_name:
            if request.headers.get("content-type", "").startswith("application/json"):
                data = await request.json()
                model_name = data.get("model_name")
            else:
                form_data = await request.form()
                model_name = form_data.get("model_name")

        if not model_name:
            return {"success": False, "message": "Model name is required"}

        # Check if model is available
        available_models = ollama_client.list_models()
        if model_name not in available_models:
            return {
                "success": False,
                "message": f"Model '{model_name}' is not available. Available models: {', '.join(available_models[:5])}"
            }

        # Update configuration
        config["default_model"] = model_name

        # Create new Ollama client with updated model
        new_ollama_client = OllamaClient(
            base_url=config["api_settings"]["ollama_base_url"],
            model=model_name,
            timeout=config["api_settings"]["ollama_timeout"]
        )

        # Update global client reference
        global ollama_client
        ollama_client = new_ollama_client

        logger.info(f"Switched to model: {model_name}")
        return {
            "success": True,
            "message": f"Successfully switched to model: {model_name}",
            "new_model": model_name,
            "available_models": available_models
        }

    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        return {"success": False, "message": f"Failed to switch model: {str(e)}"}

@app.get("/login", response_class=HTMLResponse)
async def login_page_endpoint(request: Request):
    """
    Serve the login page

    Returns:
        HTML login page
    """
    if templates and os.path.exists("templates/login.html"):
        return templates.TemplateResponse("login.html", {"request": request})
    else:
        # Fallback basic login HTML if template not found
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Login Required</title></head>
        <body>
            <h1>Authentication Required</h1>
            <p>Please login to access the financial assistant.</p>
            <form method="post" action="/auth/login">
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
        </body>
        </html>
        """)

@app.get("/", response_class=HTMLResponse)
async def ui_endpoint(request: Request):
    """
    Serve the main web UI (authentication handled client-side)

    Returns:
        HTML page with the web interface
    """
    # Always serve the HTML - let client-side JavaScript handle authentication
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

# ==================== KNOWLEDGE FEEDING API ENDPOINTS ====================

@app.post("/api/knowledge/add", response_model=ApiResponse)
async def add_knowledge_entry(entry: KnowledgeEntry, request: Request):
    """
    Add a single knowledge entry programmatically

    Args:
        entry: Knowledge entry to add
        request: FastAPI request object for authentication

    Returns:
        ApiResponse with success status
    """
    get_current_user(auth_manager, request)
    logger.info(f"Adding knowledge entry: {entry.topic}")

    try:
        # Convert knowledge entry to lesson format
        lesson_content = f"Topic: {entry.topic}\n\nContent: {entry.content}\n\nSource: {entry.source or 'API Upload'}\nConfidence: {entry.confidence}\nTags: {', '.join(entry.tags)}"

        # Store as lesson in memory
        memory_manager.add_lesson_memory(
            lesson_title=f"Knowledge: {entry.topic}",
            lesson_content=lesson_content,
            category=entry.category.value,
            confidence=entry.confidence,
            tags=entry.tags
        )

        logger.info(f"✅ Successfully added knowledge entry: {entry.topic}")

        return ApiResponse(
            success=True,
            message=f"Knowledge entry '{entry.topic}' added successfully",
            data={"topic": entry.topic, "category": entry.category.value}
        )

    except Exception as e:
        logger.error(f"Failed to add knowledge entry: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to add knowledge entry: {str(e)}"
        )

@app.post("/api/knowledge/bulk", response_model=ApiResponse)
async def add_bulk_knowledge(request: BulkKnowledgeRequest, http_request: Request):
    """
    Add multiple knowledge entries in bulk

    Args:
        request: Bulk knowledge request
        http_request: FastAPI request object for authentication

    Returns:
        ApiResponse with bulk operation results
    """
    get_current_user(auth_manager, http_request)
    logger.info(f"Adding bulk knowledge: {len(request.knowledge_entries)} entries")

    try:
        results = {
            "successful": [],
            "failed": [],
            "total": len(request.knowledge_entries)
        }

        for entry in request.knowledge_entries:
            try:
                # Convert to lesson format
                lesson_content = f"Topic: {entry.topic}\n\nContent: {entry.content}\n\nSource: {entry.source or 'API Bulk Upload'}\nConfidence: {entry.confidence}\nTags: {', '.join(entry.tags)}\nPriority: {entry.priority}"

                memory_manager.add_lesson_memory(
                    lesson_title=f"Knowledge: {entry.topic}",
                    lesson_content=lesson_content,
                    category=entry.category.value,
                    confidence=entry.confidence,
                    tags=entry.tags
                )

                results["successful"].append(entry.topic)
                logger.info(f"✅ Added bulk knowledge entry: {entry.topic}")

            except Exception as e:
                results["failed"].append({"topic": entry.topic, "error": str(e)})
                logger.error(f"Failed to add bulk knowledge entry {entry.topic}: {e}")

        success_count = len(results["successful"])
        total_count = results["total"]

        return ApiResponse(
            success=success_count > 0,
            message=f"Bulk upload completed: {success_count}/{total_count} entries added successfully",
            data=results
        )

    except Exception as e:
        logger.error(f"Bulk knowledge upload failed: {e}")
        return ApiResponse(
            success=False,
            message=f"Bulk upload failed: {str(e)}"
        )

@app.post("/api/lessons/add", response_model=ApiResponse)
async def add_lesson_entry(lesson: LessonEntry, request: Request):
    """
    Add a structured lesson programmatically

    Args:
        lesson: Lesson entry to add
        request: FastAPI request object for authentication

    Returns:
        ApiResponse with success status
    """
    get_current_user(auth_manager, request)
    logger.info(f"Adding lesson: {lesson.title}")

    try:
        # Format lesson content
        lesson_content = f"""Situation: {lesson.situation}

What was learned: {lesson.lesson}

Correct approach: {lesson.correct_approach or 'N/A'}

Wrong approach: {lesson.wrong_approach or 'N/A'}

Confidence: {lesson.confidence}
Tags: {', '.join(lesson.tags)}"""

        memory_manager.add_lesson_memory(
            lesson_title=lesson.title,
            lesson_content=lesson_content,
            category=lesson.category.value,
            confidence=lesson.confidence,
            tags=lesson.tags
        )

        logger.info(f"✅ Successfully added lesson: {lesson.title}")

        return ApiResponse(
            success=True,
            message=f"Lesson '{lesson.title}' added successfully",
            data={"title": lesson.title, "category": lesson.category.value}
        )

    except Exception as e:
        logger.error(f"Failed to add lesson: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to add lesson: {str(e)}"
        )

@app.post("/api/corrections/add", response_model=ApiResponse)
async def add_correction_entry(correction: CorrectionEntry, request: Request):
    """
    Add a correction entry programmatically

    Args:
        correction: Correction entry to add
        request: FastAPI request object for authentication

    Returns:
        ApiResponse with success status
    """
    get_current_user(auth_manager, request)
    logger.info(f"Adding correction: {correction.topic}")

    try:
        # Format correction content
        correction_content = f"""Incorrect Statement: {correction.incorrect_statement}

Correct Statement: {correction.correct_statement}

Topic: {correction.topic}

Explanation: {correction.explanation or 'No explanation provided'}

Confidence: {correction.confidence}

Type: User Correction"""

        memory_manager.add_lesson_memory(
            lesson_title=f"Correction: {correction.topic}",
            lesson_content=correction_content,
            category=correction.category.value,
            confidence=correction.confidence,
            tags=[correction.topic, "correction", "api_upload"]
        )

        logger.info(f"✅ Successfully added correction: {correction.topic}")

        return ApiResponse(
            success=True,
            message=f"Correction for '{correction.topic}' added successfully",
            data={"topic": correction.topic, "category": correction.category.value}
        )

    except Exception as e:
        logger.error(f"Failed to add correction: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to add correction: {str(e)}"
        )

@app.post("/api/definitions/add", response_model=ApiResponse)
async def add_definition_entry(definition: DefinitionEntry, request: Request):
    """
    Add a definition entry programmatically

    Args:
        definition: Definition entry to add
        request: FastAPI request object for authentication

    Returns:
        ApiResponse with success status
    """
    get_current_user(auth_manager, request)
    logger.info(f"Adding definition: {definition.term}")

    try:
        # Format definition content
        definition_content = f"""Term: {definition.term}

Definition: {definition.definition}

Expanded Form: {definition.expanded_form or 'N/A'}

Context: {definition.context or 'General context'}

Examples: {' | '.join(definition.examples) if definition.examples else 'No examples provided'}

Type: Definition"""

        memory_manager.add_lesson_memory(
            lesson_title=f"Definition: {definition.term}",
            lesson_content=definition_content,
            category=definition.category.value,
            confidence=1.0,  # Definitions should have high confidence
            tags=[definition.term, "definition", "api_upload"] + definition.examples
        )

        logger.info(f"✅ Successfully added definition: {definition.term}")

        return ApiResponse(
            success=True,
            message=f"Definition for '{definition.term}' added successfully",
            data={"term": definition.term, "category": definition.category.value}
        )

    except Exception as e:
        logger.error(f"Failed to add definition: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to add definition: {str(e)}"
        )

@app.get("/api/knowledge/stats", response_model=KnowledgeStats)
async def get_knowledge_stats(request: Request):
    """
    Get statistics about stored knowledge

    Args:
        request: FastAPI request object for authentication

    Returns:
        KnowledgeStats with detailed statistics
    """
    get_current_user(auth_manager, request)

    try:
        # Get lesson statistics
        lesson_stats = lesson_manager.get_lesson_statistics()

        # Get memory statistics by searching for different categories
        categories = ["corrections", "definitions", "trading", "general"]
        entries_by_category = {}

        for category in categories:
            try:
                results = memory_manager.search_lessons(category, category=category, n=100)
                entries_by_category[category] = len(results)
            except Exception as e:
                entries_by_category[category] = 0
                logger.warning(f"Failed to get stats for category {category}: {e}")

        return KnowledgeStats(
            total_entries=lesson_stats.get("total_lessons", 0),
            entries_by_category=entries_by_category,
            last_updated=lesson_stats.get("last_lesson_added", datetime.now().isoformat()),
            total_lessons=lesson_stats.get("total_lessons", 0),
            total_corrections=entries_by_category.get("corrections", 0),
            total_definitions=entries_by_category.get("definitions", 0)
        )

    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@app.delete("/api/knowledge/clear", response_model=ApiResponse)
async def clear_all_knowledge(request: Request):
    """
    Clear all stored knowledge (use with caution!)

    Args:
        request: FastAPI request object for authentication

    Returns:
        ApiResponse with operation result
    """
    get_current_user(auth_manager, request)
    logger.warning("⚠️ Clear all knowledge requested")

    try:
        # Clear memory collection
        memory_manager.clear_all()

        logger.warning("⚠️ All knowledge cleared successfully")

        return ApiResponse(
            success=True,
            message="All knowledge has been cleared successfully",
            data={"action": "clear_all", "timestamp": datetime.now().isoformat()}
        )

    except Exception as e:
        logger.error(f"Failed to clear knowledge: {e}")
        return ApiResponse(
            success=False,
            message=f"Failed to clear knowledge: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Local Financial Assistant API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)