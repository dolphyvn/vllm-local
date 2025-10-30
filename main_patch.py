"""
main.py patch - Integration instructions for financial embeddings
Copy these changes to your production main.py
"""

# ===== ADD THESE IMPORTS AT THE TOP OF main.py =====

# After the existing imports, add:
try:
    from financial_memory_manager import FinancialMemoryManager
    from financial_embedding_config import FinancialEmbeddingConfig
    FINANCIAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    FINANCIAL_EMBEDDINGS_AVAILABLE = False
    logger.warning("Financial embeddings not available. Using default memory manager.")


# ===== MODIFY THE INITIALIZATION SECTION =====

# Replace this section in main.py:
# OLD CODE:
# memory_manager = MemoryManager()
# lesson_manager = LessonManager()
# rag_enhancer = RAGEnhancer(memory_manager)

# NEW CODE:
# Check if financial embeddings are enabled in config
financial_enabled = config.get("financial_embedding", {}).get("enabled", False)

if financial_enabled and FINANCIAL_EMBEDDINGS_AVAILABLE:
    embedding_model = config.get("financial_embedding", {}).get("model_type", "finance-technical-v1")
    logger.info(f"📊 Initializing Financial Memory Manager with model: {embedding_model}")

    memory_manager = FinancialMemoryManager(
        collection_name=config.get("memory_settings", {}).get("collection_name", "financial_trading_memory"),
        persist_directory=config.get("memory_settings", {}).get("persist_directory", "./chroma_db"),
        embedding_model=embedding_model
    )
    logger.info("✅ Financial Memory Manager initialized successfully")
else:
    logger.info("Using default Memory Manager")
    memory_manager = MemoryManager()

lesson_manager = LessonManager()
rag_enhancer = RAGEnhancer(memory_manager)


# ===== ADD ENHANCED TRADING CONTEXT FUNCTION =====

# Add this function after the existing prompt building functions:

def build_enhanced_trading_context(user_message: str, context_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build enhanced trading context with financial embedding model

    Args:
        user_message: User's trading query
        context_data: Enhanced context data

    Returns:
        Enhanced message list with trading context
    """
    messages = []

    # Enhanced system prompt for trading analysis
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    trading_prompt = f"""{system_prompt}

TRADING ANALYSIS GUIDELINES:
- Always consider technical indicators and their signals
- Analyze multiple timeframes for comprehensive view
- Include risk management considerations
- Reference historical patterns when relevant
- Mention confidence levels in your analysis

You have access to enhanced trading context from previous analyses. Use this information to provide more accurate and contextually relevant trading insights."""

    messages.append({"role": "system", "content": trading_prompt})

    # Add trading memories with enhanced formatting
    if context_data.get("conversations"):
        trading_context_str = "📊 PREVIOUS TRADING ANALYSES:\n"
        for i, entry in enumerate(context_data["conversations"], 1):
            trading_context_str += f"Analysis {i}: {entry}\n"
        trading_context_str += "\nFocus on similar patterns and outcomes from these previous analyses.\n"
        messages.append({"role": "system", "content": trading_context_str})

    # Add lesson memories with emphasis on trading lessons
    if context_data.get("lessons"):
        lesson_context_str = "📈 TRADING LESSONS & PATTERNS:\n"
        for i, lesson in enumerate(context_data["lessons"], 1):
            lesson_context_str += f"Lesson {i}: {lesson}\n"
        lesson_context_str += "\nCRITICAL: Apply these lessons and patterns to improve current analysis accuracy.\n"
        messages.append({"role": "system", "content": lesson_context_str})

    # Check for trading-specific queries and add appropriate context
    trading_keywords = ["rsi", "macd", "support", "resistance", "trend", "breakout", "volume", "price", "indicator"]
    if any(keyword in user_message.lower() for keyword in trading_keywords):
        trading_instruction = "\n🎯 TECHNICAL ANALYSIS FOCUS: This query involves technical analysis. Provide specific indicator readings, signal strength, and actionable trading considerations.\n"
        messages.append({"role": "system", "content": trading_instruction})

    # Add user message
    messages.append({"role": "user", "content": user_message})

    return messages


# ===== MODIFY THE CHAT ENDPOINT =====

# In the chat_endpoint function, replace this line:
# messages = build_enhanced_contextual_prompt(request.message, context_data)

# With this conditional logic:
financial_enabled = config.get("financial_embedding", {}).get("enabled", False)

if financial_enabled and hasattr(memory_manager, 'search_trading_memories'):
    # Use enhanced trading context
    messages = build_enhanced_trading_context(request.message, context_data)
else:
    # Use original context
    messages = build_enhanced_contextual_prompt(request.message, context_data)


# ===== ADD HEALTH CHECK FOR FINANCIAL EMBEDDINGS =====

# Add this new endpoint after the existing health endpoint:

@app.get("/health/financial")
async def financial_health_check():
    """
    Health check for financial embedding system

    Returns:
        Financial system status and configuration
    """
    try:
        financial_enabled = config.get("financial_embedding", {}).get("enabled", False)

        if not financial_enabled:
            return {
                "status": "disabled",
                "message": "Financial embeddings are not enabled",
                "financial_config": None
            }

        # Check if FinancialMemoryManager is being used
        is_financial_manager = hasattr(memory_manager, 'search_trading_memories')

        # Get model configuration
        model_config = config.get("financial_embedding", {})

        return {
            "status": "enabled",
            "financial_manager_active": is_financial_manager,
            "embedding_model": model_config.get("model_type", "unknown"),
            "dimensions": model_config.get("dimensions", 0),
            "memory_collection": config.get("memory_settings", {}).get("collection_name", "unknown"),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Financial health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ===== INSTALLATION INSTRUCTIONS =====

"""
TO COMPLETE THE INTEGRATION:

1. Copy the financial files to your production server:
   scp financial_embedding_config.py root@your-server:/opt/vllm-local/
   scp financial_memory_manager.py root@your-server:/opt/vllm-local/

2. Apply the main.py changes:
   - Add the imports
   - Modify the memory manager initialization
   - Add the enhanced context function
   - Update the chat endpoint
   - Add the financial health check endpoint

3. Install dependencies:
   pip3 install sentence-transformers torch

4. Test the integration:
   curl http://localhost:8080/health/financial

5. Restart your application:
   python3 main.py

The finance-technical-v1 model will now:
- Better understand technical indicator terminology
- Recognize trading patterns and setups
- Provide more relevant context for trading queries
- Handle market-specific language and concepts
"""