"""
main.py - FastAPI application for local financial AI assistant
Provides chat, memory management, and health endpoints for vLLM integration
"""

import json
import asyncio
import requests
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import os

from memory import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local Financial Assistant",
    description="AI-powered financial analysis and trading strategy assistant",
    version="1.0.0"
)

# Mount static files and templates
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    templates = None

# Initialize memory manager
memory_manager = MemoryManager()

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

# vLLM API client
class VLLMClient:
    """Client for interacting with vLLM OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:8000", model: str = "phi3"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat completion request to vLLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the completion

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API request failed: {e}")
            raise HTTPException(status_code=503, detail=f"vLLM service unavailable: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from vLLM: {e}")
            raise HTTPException(status_code=500, detail="Invalid response from model service")

# Initialize vLLM client
vllm_client = VLLMClient(model=config.get("default_model", "phi3"))

def build_contextual_prompt(user_message: str, memory_context: int = 3) -> List[Dict[str, str]]:
    """
    Build a prompt with system context, memory, and user message

    Args:
        user_message: The user's input message
        memory_context: Number of memory entries to include

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

    # Retrieve relevant memory
    try:
        memory_entries = memory_manager.get_memory(user_message, n=memory_context)
        if memory_entries:
            memory_context_str = "<recent_context>\n"
            for i, entry in enumerate(memory_entries, 1):
                memory_context_str += f"Memory {i}: {entry}\n"
            memory_context_str += "</recent_context>"
            messages.append({"role": "system", "content": memory_context_str})
    except Exception as e:
        logger.warning(f"Failed to retrieve memory: {e}")

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

        # Get response from vLLM
        ai_response = await vllm_client.chat_completion(messages)

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

        return HealthResponse(
            status="ok",
            model=config.get("default_model", "phi3"),
            timestamp=datetime.now().isoformat(),
            memory_status=memory_status
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
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
    logger.info(f"Configuration loaded: {config.get('default_model', 'phi3')} model")
    logger.info("Memory system initialized")

    # Check if UI files are available
    if os.path.exists("templates/index.html"):
        logger.info("Web UI template found - web interface available at http://localhost:8080")
    else:
        logger.warning("Web UI template not found - only API endpoints available")

    if os.path.exists("static"):
        logger.info("Static files directory found")
    else:
        logger.warning("Static files directory not found")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Local Financial Assistant API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)