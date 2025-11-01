"""
main.py - FastAPI application for local financial AI assistant
Provides chat, memory management, and health endpoints for vLLM integration
"""

import json
import asyncio
import requests
import aiohttp
import base64
import mimetypes
import glob
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
import os
import aiofiles
import uuid
import queue
import threading
import pandas as pd
import numpy as np
import re

from memory import MemoryManager
from lessons import LessonManager
from rag_enhancer import RAGEnhancer
from auth import AuthManager, get_current_user
from knowledge_feeder import (
    KnowledgeEntry, BulkKnowledgeRequest, LessonEntry,
    CorrectionEntry, DefinitionEntry, ApiResponse, KnowledgeStats,
    KnowledgeCategory
)
from web_search import WebSearchTool, TradingNewsAPI, TradingTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM client for trading analysis
try:
    from openai import AsyncOpenAI
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    AsyncOpenAI = None

# Ollama client configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

async def get_ollama_client():
    """Get or create Ollama client"""
    if not OLLAMA_AVAILABLE:
        return None
    return AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

def _is_ollama_available() -> bool:
    """Check if Ollama is available"""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def _is_trading_query(query_text: str) -> bool:
    """Check if query is trading-related"""
    trading_keywords = [
        'trade', 'trading', 'market', 'price', 'buy', 'sell', 'signal',
        'indicator', 'rsi', 'macd', 'ema', 'sma', 'bollinger', 'volatility',
        'trend', 'support', 'resistance', 'breakout', 'reversal', 'momentum',
        'xauusd', 'gold', 'forex', 'currency', 'pip', 'lot', 'leverage',
        'stop loss', 'take profit', 'risk/reward', 'timeframe', 'candlestick'
    ]
    query_lower = query_text.lower()
    return any(keyword in query_lower for keyword in trading_keywords)

async def get_llm_trading_analysis(prompt: str, data: dict = None, model: str = "gemma3:1b") -> str:
    """
    Get trading analysis from local LLM using existing OllamaClient
    """
    try:
        # Use the detailed prompt directly (it already contains all the technical data)
        # The create_detailed_llm_prompt function already formats everything properly

        # Build messages for Ollama
        messages = [
            {
                "role": "system",
                "content": "You are an expert trading analyst and technical analysis specialist. CRITICAL: You MUST follow the exact format requested in the prompt. The user specifically asked for numbered sections with TRADE DIRECTION, ENTRY PRICE, STOP LOSS, TAKE PROFIT, RISK/REWARD, TECHNICAL REASONING, and RISK MANAGEMENT. Do NOT write a generic analysis - follow the exact structure requested!"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        logger.info(f"Sending prompt to LLM (model: {model}), prompt length: {len(prompt)}")

        # Use existing OllamaClient with the specified model
        client = OllamaClient(base_url=OLLAMA_BASE_URL, model=model)
        ai_response = await client.chat_completion(messages)

        logger.info(f"LLM response received, length: {len(ai_response) if ai_response else 0}")

        return ai_response

    except Exception as e:
        logger.error(f"Error getting LLM trading analysis: {e}")
        return f"Error getting LLM analysis: {e}"

async def get_llm_trading_analysis_old(query_text: str, context_data: Dict[str, Any]) -> Optional[str]:
    """Get LLM analysis for trading queries"""
    if not _is_ollama_available():
        return None

    try:
        client = await get_ollama_client()
        if not client:
            return None

        prompt = _create_trading_analysis_prompt(query_text, context_data)

        response = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert trading analyst with deep knowledge of technical analysis, risk management, and market psychology."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error getting LLM trading analysis: {e}")
        return None

def _create_trading_analysis_prompt(query_text: str, context_data: Dict[str, Any]) -> str:
    """Create enhanced trading analysis prompt for LLM"""

    # Extract relevant context
    similar_patterns = context_data.get('similar_patterns', [])
    enhanced_context = context_data.get('enhanced_context', {})
    current_indicators = enhanced_context.get('current_indicators', {})
    base_recommendation = context_data.get('base_recommendation', {})

    # Determine query type for specialized prompts
    query_type = _classify_trading_query(query_text)

    prompt = f"""
=== TRADING ANALYSIS REQUEST ===
Query: {query_text}
Query Type: {query_type}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

CURRENT MARKET CONDITIONS:
"""

    # Add indicator data with context
    if current_indicators:
        prompt += "\n📊 Technical Indicators:\n"

        # Categorize indicators
        momentum_indicators = {}
        trend_indicators = {}
        volatility_indicators = {}
        volume_indicators = {}

        for indicator, value in current_indicators.items():
            if isinstance(value, (int, float)):
                if 'rsi' in indicator.lower() or 'macd' in indicator.lower():
                    momentum_indicators[indicator] = value
                elif 'ema' in indicator.lower() or 'sma' in indicator.lower() or 'vwap' in indicator.lower():
                    trend_indicators[indicator] = value
                elif 'atr' in indicator.lower() or 'bb' in indicator.lower():
                    volatility_indicators[indicator] = value
                elif 'volume' in indicator.lower():
                    volume_indicators[indicator] = value

        if momentum_indicators:
            prompt += "  Momentum:\n"
            for indicator, value in momentum_indicators.items():
                interpretation = _interpret_indicator(indicator, value)
                prompt += f"    - {indicator.upper()}: {value:.2f} ({interpretation})\n"

        if trend_indicators:
            prompt += "  Trend:\n"
            for indicator, value in trend_indicators.items():
                interpretation = _interpret_indicator(indicator, value)
                prompt += f"    - {indicator.upper()}: {value:.2f} ({interpretation})\n"

        if volatility_indicators:
            prompt += "  Volatility:\n"
            for indicator, value in volatility_indicators.items():
                interpretation = _interpret_indicator(indicator, value)
                prompt += f"    - {indicator.upper()}: {value:.2f} ({interpretation})\n"

        if volume_indicators:
            prompt += "  Volume:\n"
            for indicator, value in volume_indicators.items():
                interpretation = _interpret_indicator(indicator, value)
                prompt += f"    - {indicator.upper()}: {value:.2f} ({interpretation})\n"

    # Add base recommendation from RAG
    if base_recommendation:
        prompt += f"\n🤖 RAG System Recommendation:\n"
        prompt += f"  - Strategy: {base_recommendation.get('strategy', 'HOLD')}\n"
        prompt += f"  - Confidence: {base_recommendation.get('confidence', 50):.1f}%\n"
        prompt += f"  - Risk/Reward: {base_recommendation.get('risk_reward_ratio', 0):.1f}:1\n"
        if base_recommendation.get('signals'):
            prompt += f"  - Signals: {', '.join(base_recommendation['signals'])}\n"

    # Add similar patterns with detailed analysis
    if similar_patterns:
        prompt += f"\n📈 Similar Historical Patterns (top {len(similar_patterns[:3])}):\n"
        for i, pattern in enumerate(similar_patterns[:3], 1):
            pattern_data = pattern.get('pattern', {})
            outcome = pattern.get('outcome', {})

            prompt += f"\n  Pattern {i} - {pattern.get('timeframe', 'Unknown')}:\n"

            # Key metrics
            if pattern_data:
                prompt += f"    RSI: {pattern_data.get('rsi', 'N/A')}"
                if pattern_data.get('ema_20') and pattern_data.get('ema_50'):
                    prompt += f" | EMA20/50: {pattern_data.get('ema_20', 'N/A')}/{pattern_data.get('ema_50', 'N/A')}"
                if pattern_data.get('vwap'):
                    prompt += f" | VWAP: {pattern_data.get('vwap', 'N/A')}"
                prompt += "\n"

            # Outcome
            prompt += f"    Result: {outcome.get('result', 'Unknown')}"
            if outcome.get('confidence'):
                prompt += f" (Confidence: {outcome['confidence']:.1f}%)"
            prompt += "\n"

            # Performance if available
            if outcome.get('future_candles'):
                future_candle = outcome['future_candles'][0]
                price_change = future_candle.get('price_change_pct', 0)
                max_profit = future_candle.get('max_profit_pct', 0)
                max_loss = future_candle.get('max_loss_pct', 0)

                prompt += f"    Performance: {price_change:+.2f}% | Max: {max_profit:+.2f}% | Min: {max_loss:+.2f}%\n"

    # Add query-specific analysis template
    prompt += f"\n{'='*50}\n"
    prompt += f"ANALYSIS REQUIREMENTS ({query_type.upper()}):\n"
    prompt += f"{'='*50}\n"

    if query_type == "entry_exit":
        prompt += _get_entry_exit_template()
    elif query_type == "risk_management":
        prompt += _get_risk_management_template()
    elif query_type == "market_analysis":
        prompt += _get_market_analysis_template()
    elif query_type == "pattern_recognition":
        prompt += _get_pattern_recognition_template()
    else:
        prompt += _get_general_trading_template()

    prompt += f"\n{'='*50}\n"
    prompt += "IMPORTANT REMINDERS:\n"
    prompt += "- This is for educational and analysis purposes only\n"
    prompt += "- Always use proper risk management (1-2% max per trade)\n"
    prompt += "- Consider market context and news events\n"
    prompt += "- Paper trade strategies before real money\n"
    prompt += f"- Current market session: {_get_current_session()}\n"

    return prompt

def _classify_trading_query(query_text: str) -> str:
    """Classify the type of trading query"""
    query_lower = query_text.lower()

    if any(word in query_lower for word in ['entry', 'exit', 'buy', 'sell', 'take profit', 'stop loss']):
        return "entry_exit"
    elif any(word in query_lower for word in ['risk', 'position size', 'leverage', 'margin']):
        return "risk_management"
    elif any(word in query_lower for word in ['trend', 'momentum', 'volatility', 'market condition']):
        return "market_analysis"
    elif any(word in query_lower for word in ['pattern', 'formation', 'setup', 'signal']):
        return "pattern_recognition"
    else:
        return "general"

def _interpret_indicator(indicator: str, value: float) -> str:
    """Provide quick interpretation of indicator values"""
    indicator_lower = indicator.lower()

    if 'rsi' in indicator_lower:
        if value >= 70:
            return "Overbought"
        elif value <= 30:
            return "Oversold"
        else:
            return "Neutral"
    elif 'macd' in indicator_lower:
        return "Positive" if value > 0 else "Negative"
    elif 'atr' in indicator_lower:
        return "High" if value > 0.01 else "Low"
    elif 'vwap' in indicator_lower:
        return "Above VWAP" if value > 0 else "Below VWAP"
    else:
        return "Normal"

def _get_current_session() -> str:
    """Get current trading session"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour

    if 0 <= hour < 8:
        return "Asian Session"
    elif 8 <= hour < 13:
        return "London Session"
    elif 13 <= hour < 17:
        return "US Session"
    elif 17 <= hour < 22:
        return "US Late Session"
    else:
        return "Pacific Session"

def _get_entry_exit_template() -> str:
    """Get specialized template for entry/exit analysis"""
    return """
Provide specific entry/exit recommendations:

1. **ENTRY ANALYSIS**:
   - Optimal entry price and timing
   - Entry confirmation signals
   - Entry rationale and confluence factors

2. **EXIT STRATEGY**:
   - Take profit levels (partial/complete)
   - Stop loss placement and logic
   - Trailing stop recommendations

3. **POSITION SIZING**:
   - Recommended position size (% of portfolio)
   - Risk per trade calculation
   - Adjustments for volatility

4. **EXECUTION PLAN**:
   - Order type recommendations
   - Market vs limit orders
   - Entry timing and session considerations

5. **TRADE INVALIDATION**:
   - Conditions that invalidate the setup
   - Early warning signs
   - Contingency plans
"""

def _get_risk_management_template() -> str:
    """Get specialized template for risk management analysis"""
    return """
Focus on risk management and position sizing:

1. **RISK ASSESSMENT**:
   - Current market risk level
   - Volatility considerations
   - Correlation with existing positions

2. **POSITION SIZING**:
   - Recommended risk per trade (1-2% rule)
   - Account size considerations
   - Volatility-adjusted sizing

3. **RISK CONTROLS**:
   - Stop loss strategy and placement
   - Maximum drawdown limits
   - Portfolio heat management

4. **RISK/REWARD OPTIMIZATION**:
   - Minimum acceptable R/R ratio
   - Profit target adjustments
   - Risk scaling methodology

5. **PORTFOLIO CONTEXT**:
   - Correlation with existing trades
   - Sector/diversification considerations
   - Overall portfolio risk balance
"""

def _get_market_analysis_template() -> str:
    """Get specialized template for market analysis"""
    return """
Provide comprehensive market analysis:

1. **TREND ANALYSIS**:
   - Short, medium, long-term trend direction
   - Trend strength and momentum
   - Potential trend reversal signals

2. **MARKET STRUCTURE**:
   - Key support and resistance levels
   - Market structure breaks
   - Supply/demand zone analysis

3. **VOLATILITY & MOMENTUM**:
   - Current volatility regime
   - Momentum shifts and divergences
   - Volatility expectations

4. **INTERMARKET ANALYSIS**:
   - Correlated markets influence
   - Risk-on/risk-off sentiment
   - Currency strength impacts

5. **MARKET SENTIMENT**:
   - Recent price action interpretation
   - Volume and participation analysis
   - Potential catalysts and risks
"""

def _get_pattern_recognition_template() -> str:
    return """
Provide detailed pattern analysis:

1. **PATTERN IDENTIFICATION**:
   - Pattern type and completion status
   - Pattern reliability rating
   - Timeframe context

2. **PATTERN VALIDATION**:
   - Confirming signals and confluence
   - Volume confirmation
   - Failure risk assessment

3. **PATTERN PROJECTIONS**:
   - Measured price targets
   - Time projections
   - Failure scenarios

4. **HISTORICAL CONTEXT**:
   - Similar pattern performance
   - Current market compatibility
   - Seasonal/periodic factors

5. **TRADING IMPLICATIONS**:
   - Bias direction and strength
   - Entry trigger points
   - Risk parameters for this pattern type
"""

def _get_general_trading_template() -> str:
    """Get general trading analysis template"""
    return """
Provide comprehensive trading analysis:

1. **MARKET OVERVIEW**:
   - Current market conditions
   - Key technical levels
   - Overall sentiment

2. **TRADING OPPORTUNITIES**:
   - Potential setups identified
   - Risk/reward assessment
   - Confidence level

3. **RISK CONSIDERATIONS**:
   - Key risk factors
   - Market uncertainty
   - Risk mitigation strategies

4. **RECOMMENDATIONS**:
   - Specific actionable advice
   - Alternative scenarios
   - Monitoring requirements

5. **CONFIDENCE & TIMING**:
   - Analysis confidence (1-10)
   - Optimal time horizon
   - Key catalysts to watch
"""

def generate_enhanced_recommendation_from_analysis(context_data: Dict[str, Any], llm_analysis: str) -> Dict[str, Any]:
    """Generate enhanced trading recommendation combining RAG and LLM analysis"""

    # Base recommendation from RAG
    base_recommendation = context_data.get('base_recommendation', {})

    # Parse LLM analysis for key insights
    llm_insights = _parse_llm_analysis(llm_analysis) if llm_analysis else {}

    enhanced_recommendation = {
        'strategy': base_recommendation.get('strategy', 'HOLD'),
        'confidence': base_recommendation.get('confidence', 50),
        'entry_price': base_recommendation.get('entry_price'),
        'stop_loss': base_recommendation.get('stop_loss'),
        'take_profit': base_recommendation.get('take_profit'),
        'risk_reward_ratio': base_recommendation.get('risk_reward_ratio', 0),

        # Enhanced components
        'llm_insights': llm_insights,
        'market_analysis': llm_insights.get('market_analysis', 'No LLM analysis available'),
        'risk_assessment': llm_insights.get('risk_assessment', 'No risk assessment available'),
        'enhanced_signals': _combine_signals(base_recommendation, llm_insights),
        'reasoning': f"RAG Analysis: {base_recommendation.get('reasoning', '')} | LLM Analysis: {llm_insights.get('summary', '')}",

        # Metadata
        'analysis_sources': ['RAG Database', 'LLM Model'],
        'analysis_timestamp': datetime.now().isoformat(),
        'enhanced_confidence': _calculate_enhanced_confidence(base_recommendation, llm_insights)
    }

    return enhanced_recommendation

def _parse_llm_analysis(llm_analysis: str) -> Dict[str, Any]:
    """Parse LLM analysis response for key insights"""
    insights = {
        'summary': '',
        'market_analysis': '',
        'risk_assessment': '',
        'trading_opportunities': [],
        'confidence_level': 5,
        'key_levels': []
    }

    if not llm_analysis:
        return insights

    # Extract confidence level
    import re
    confidence_match = re.search(r'confidence[:\s]*(\d+)/?10?', llm_analysis.lower())
    if confidence_match:
        insights['confidence_level'] = int(confidence_match.group(1))

    # Extract summary (first few sentences)
    sentences = llm_analysis.split('.')
    if sentences:
        insights['summary'] = '. '.join(sentences[:2]).strip()

    # Extract key price levels
    price_pattern = r'[\$]?(\d+\.?\d*)'
    price_matches = re.findall(price_pattern, llm_analysis)
    if price_matches:
        insights['key_levels'] = [float(p) for p in price_matches[:5]]  # Top 5 levels

    # Store full analysis
    insights['market_analysis'] = llm_analysis

    return insights

def _combine_signals(base_recommendation: Dict[str, Any], llm_insights: Dict[str, Any]) -> List[str]:
    """Combine RAG and LLM signals"""
    signals = []

    # RAG signals
    rag_signals = base_recommendation.get('signals', [])
    if rag_signals:
        signals.extend([f"RAG: {signal}" for signal in rag_signals])

    # LLM signals
    if llm_insights.get('confidence_level', 0) >= 7:
        signals.append("LLM: High confidence analysis")

    if llm_insights.get('key_levels'):
        signals.append(f"LLM: Key support/resistance levels identified")

    if llm_insights.get('trading_opportunities'):
        signals.append("LLM: Trading opportunities detected")

    return signals

def _calculate_enhanced_confidence(base_recommendation: Dict[str, Any], llm_insights: Dict[str, Any]) -> float:
    """Calculate enhanced confidence score combining RAG and LLM"""
    rag_confidence = base_recommendation.get('confidence', 50)
    llm_confidence = llm_insights.get('confidence_level', 5) * 10  # Convert 1-10 to 10-100

    # Weighted average (giving slightly more weight to LLM for qualitative analysis)
    enhanced_confidence = (rag_confidence * 0.4) + (llm_confidence * 0.6)

    return min(100, max(0, enhanced_confidence))

def convert_mt5_to_rag_format(content: bytes, symbol: str, timeframe: str) -> bytes:
    """
    Convert MT5 CSV format to RAG-compatible format
    MT5 Format: timestamp,open,high,low,close,tick_volume,spread,real_volume
    RAG Format: TimeFrame,Symbol,Candle,DateTime,Open,High,Low,Close,Volume,HL,Body
    """
    try:
        import pandas as pd
        from datetime import datetime
        import io

        # Read MT5 format
        df = pd.read_csv(io.BytesIO(content))

        # Convert to RAG format
        rag_data = []

        # Calculate candle numbers (descending from most recent)
        total_candles = len(df)

        for idx, row in df.iterrows():
            # Calculate candle number (descending)
            candle_num = total_candles - idx

            # Convert timestamp format
            dt = pd.to_datetime(row['timestamp'])
            datetime_str = dt.strftime('%Y.%m.%d %H:%M')

            # Calculate additional columns
            high_low = row['high'] - row['low']
            body = abs(row['close'] - row['open'])

            # Create RAG format row
            rag_row = {
                'TimeFrame': f'PERIOD_{timeframe.upper()}',
                'Symbol': symbol.upper(),
                'Candle': candle_num,
                'DateTime': datetime_str,
                'Open': row['open'],
                'High': row['high'],
                'Low': row['low'],
                'Close': row['close'],
                'Volume': row['tick_volume'],
                'HL': high_low,
                'Body': body
            }
            rag_data.append(rag_row)

        # Convert to DataFrame and then to CSV
        rag_df = pd.DataFrame(rag_data)

        # Convert to CSV string
        csv_content = rag_df.to_csv(index=False)

        logger.info(f"Converted {len(rag_data)} candles from MT5 format to RAG format")
        logger.info(f"Sample converted row: {rag_data[0] if rag_data else 'No data'}")

        return csv_content.encode('utf-8')

    except Exception as e:
        logger.error(f"Failed to convert MT5 format to RAG format: {str(e)}")
        # Return original content if conversion fails
        return content

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

# Initialize web search and trading news tools
web_search_tool = WebSearchTool()
trading_news_api = TradingNewsAPI(web_search_tool)
trading_tools = TradingTools(web_search_tool, trading_news_api)
logger.info("Web search and trading news tools initialized")

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
    collections: Optional[List[str]] = None  # Collections to query: ["financial_memory", "trading_patterns", "live_analysis"]

class StreamChatRequest(BaseModel):
    message: str
    model: str = config.get("default_model", "phi3")
    memory_context: int = 3
    files: Optional[List[Dict[str, Any]]] = None
    collections: Optional[List[str]] = None  # Collections to query
    timeframe: Optional[str] = None  # Trading timeframe for live analysis

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


def is_trade_query(message: str) -> bool:
    """Detect if user is asking for trade recommendations"""
    trade_keywords = [
        "trade setup", "trading opportunity", "best setup",
        "entry", "stop loss", "take profit",
        "should i buy", "should i sell",
        "long setup", "short setup",
        "swing trade", "day trade",
        "market direction", "trade recommendation",
        "what's the trade", "give me a trade",
        "buy or sell", "go long", "go short"
    ]
    return any(keyword in message.lower() for keyword in trade_keywords)


def extract_symbol(message: str) -> Optional[str]:
    """Extract trading symbol from message"""
    # Common trading symbols
    symbols = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "US30", "NAS100", "SPX500"]

    for symbol in symbols:
        if symbol.lower() in message.lower():
            return symbol
    return None


def query_live_analysis(symbol: str, timeframe: str = "M15") -> Optional[Dict[str, Any]]:
    """
    Query live analysis from ChromaDB for trade recommendations

    Args:
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        Live analysis data or None if not found
    """
    try:
        # Import ChromaLiveAnalyzer
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, "scripts")
        sys.path.append(scripts_dir)

        from chroma_live_analyzer import ChromaLiveAnalyzer

        analyzer = ChromaLiveAnalyzer()
        return analyzer.get_latest_analysis(symbol, timeframe)

    except Exception as e:
        logger.error(f"Failed to query live analysis: {e}")
        return None


def format_live_analysis(analysis: Dict[str, Any]) -> str:
    """Format live analysis data for display"""
    if not analysis or 'metadata' not in analysis:
        return "No live analysis available"

    metadata = analysis['metadata']

    return f"""
📊 LIVE ANALYSIS - {metadata.get('symbol', 'Unknown')} {metadata.get('timeframe', 'Unknown')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 TRADE SETUP: {metadata.get('trade_direction', 'UNKNOWN')}
📊 Current Price: {metadata.get('current_price', 0):.2f}
🎯 Entry Price: {metadata.get('entry_price', 0):.2f}
🛡️ Stop Loss: {metadata.get('stop_loss', 0):.2f}
🎉 Take Profit: {metadata.get('take_profit', 0):.2f}
📈 Risk/Reward: {metadata.get('risk_reward_ratio', 0):.1f}:1
💪 Confidence: {metadata.get('confidence', 0)}%

📋 PATTERN DETAILS:
• Pattern: {metadata.get('pattern_name', 'Unknown')}
• Type: {metadata.get('pattern_type', 'Unknown')}
• Direction: {metadata.get('pattern_direction', 'Unknown')}
• Pattern Confidence: {metadata.get('pattern_confidence', 0)}%

📈 MARKET CONTEXT:
• Trend: {metadata.get('trend', 'Unknown')}
• RSI State: {metadata.get('rsi_state', 'Unknown')}
• Volume: {metadata.get('volume_state', 'Unknown')}
• Session: {metadata.get('session', 'Unknown')}

📊 TECHNICAL INDICATORS:
• RSI: {metadata.get('rsi', 0):.1f}
• MACD: {metadata.get('macd', 0):.2f}
• Volume Ratio: {metadata.get('volume_ratio', 0):.1f}x
• ATR: {metadata.get('atr', 0):.2f}

📅 Timestamp: {metadata.get('timestamp', 'Unknown')}
    """.strip()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint that processes user messages and returns AI responses
    Now includes automatic web search for current information!

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
        # Check if web search is needed for current information
        needs_web_search = trading_tools.should_use_web_search(request.message)

        web_context = ""
        if needs_web_search:
            logger.info("🌐 Web search triggered - fetching real-time information")

            try:
                # Determine what to search for based on query
                query_lower = request.message.lower()

                if "news" in query_lower:
                    # Extract symbol if mentioned
                    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
                    symbol = next((s for s in symbols if s.lower() in query_lower), "XAUUSD")
                    web_context = trading_news_api.get_symbol_news(symbol)
                    logger.info(f"📰 Fetched news for {symbol}")

                elif "calendar" in query_lower or "event" in query_lower:
                    web_context = trading_news_api.get_economic_calendar()
                    logger.info("📅 Fetched economic calendar")

                elif "sentiment" in query_lower:
                    symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
                    symbol = next((s for s in symbols if s.lower() in query_lower), "XAUUSD")
                    web_context = trading_news_api.get_market_sentiment(symbol)
                    logger.info(f"📊 Fetched market sentiment for {symbol}")

                elif "overview" in query_lower or ("market" in query_lower and "today" in query_lower):
                    web_context = trading_news_api.get_market_overview()
                    logger.info("🌍 Fetched market overview")

                else:
                    # General web search
                    web_context = web_search_tool.search_web(request.message, max_results=3)
                    logger.info("🔍 Performed general web search")

            except Exception as e:
                logger.warning(f"Web search failed, continuing with RAG only: {e}")
                web_context = ""

        # Enhanced RAG context retrieval with collection selection
        collections = request.collections if request.collections else ["financial_memory"]
        logger.info(f"Querying collections: {collections}")
        context_data = rag_enhancer.enhance_query_with_rag(
            request.message,
            max_context=request.memory_context,
            collections=collections
        )

        # Check if this is a trade query (Phase 4)
        if is_trade_query(request.message):
            logger.info("🎯 Trade query detected")

            # Extract symbol from message
            symbol = extract_symbol(request.message)

            if symbol:
                # Query live analysis from ChromaDB
                live_analysis = query_live_analysis(symbol, request.timeframe or "M15")

                if live_analysis:
                    # Found recent analysis
                    enhanced_message = f"""User question: {request.message}

📊 LIVE MARKET ANALYSIS for {symbol}:
{format_live_analysis(live_analysis)}

Please provide a comprehensive trade recommendation based on this analysis.
Explain the setup, entry/SL/TP levels, and reasoning clearly."""
                else:
                    # No recent analysis found
                    enhanced_message = f"""User question: {request.message}

⚠️ No recent live analysis found for {symbol}.
Please upload latest data first:
curl -F "file=@{symbol}_M15_200.csv" http://localhost:8080/upload

Or check historical patterns from RAG knowledge base."""
            else:
                enhanced_message = request.message
        elif web_context:
            enhanced_message = f"""User question: {request.message}

📡 REAL-TIME INFORMATION FROM THE WEB:
{web_context}

Please provide a comprehensive answer using both the above real-time information and your knowledge."""
        else:
            enhanced_message = request.message

        # Build contextual prompt with enhanced memory
        messages = build_enhanced_contextual_prompt(enhanced_message, context_data)

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
            # Check if web search is needed for current information
            needs_web_search = trading_tools.should_use_web_search(request.message)

            web_context = ""
            if needs_web_search:
                logger.info("🌐 Web search triggered in streaming mode")

                try:
                    # Determine what to search for based on query
                    query_lower = request.message.lower()

                    if "news" in query_lower:
                        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
                        symbol = next((s for s in symbols if s.lower() in query_lower), "XAUUSD")
                        web_context = trading_news_api.get_symbol_news(symbol)
                        logger.info(f"📰 Fetched news for {symbol}")

                    elif "calendar" in query_lower or "event" in query_lower:
                        web_context = trading_news_api.get_economic_calendar()
                        logger.info("📅 Fetched economic calendar")

                    elif "sentiment" in query_lower:
                        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
                        symbol = next((s for s in symbols if s.lower() in query_lower), "XAUUSD")
                        web_context = trading_news_api.get_market_sentiment(symbol)
                        logger.info(f"📊 Fetched market sentiment for {symbol}")

                    elif "overview" in query_lower or ("market" in query_lower and "today" in query_lower):
                        web_context = trading_news_api.get_market_overview()
                        logger.info("🌍 Fetched market overview")

                    else:
                        web_context = web_search_tool.search_web(request.message, max_results=3)
                        logger.info("🔍 Performed general web search")

                except Exception as e:
                    logger.warning(f"Web search failed in streaming mode: {e}")
                    web_context = ""

            # Enhanced RAG context retrieval with collection selection
            collections = request.collections if request.collections else ["financial_memory"]
            logger.info(f"[Streaming] Querying collections: {collections}")
            context_data = rag_enhancer.enhance_query_with_rag(
                request.message,
                max_context=request.memory_context,
                collections=collections
            )

            # Add file information to context if files are provided
            if request.files:
                file_context = "\n\n📎 ATTACHED FILES:\n"
                file_analysis_prompts = []

                try:
                    for i, file in enumerate(request.files, 1):
                        # Safely access file fields with fallbacks
                        file_name = file.get('name', 'Unknown file')
                        file_content_type = file.get('content_type', file.get('type', 'unknown'))
                        file_size = file.get('size', 0)

                        file_context += f"File {i}: {file_name} ({file_content_type}, {file_size} bytes)\n"
                        if file.get('content'):
                            content = file['content']
                            if content['type'] == 'text':
                                file_context += f"Content: {content['content'][:500]}...\n"
                                file_analysis_prompts.append(f"Please analyze the text file '{file_name}' and provide insights about its content.")
                            elif content['type'] == 'image':
                                file_context += f"Image uploaded ({content.get('format', 'unknown format')})\n"
                                file_analysis_prompts.append(f"Please analyze the image '{file_name}' and describe what you see.")
                            else:
                                file_context += f"Document uploaded for analysis\n"
                                file_analysis_prompts.append(f"Please analyze the document '{file_name}' and provide a summary or key insights.")

                    # Add explicit analysis instruction
                    if file_analysis_prompts:
                        analysis_instruction = "\n\nIMPORTANT: Please acknowledge and analyze the uploaded file(s) above. The user specifically wants you to examine and comment on the file content they've shared."
                        enhanced_message = f"{request.message}{file_context}{analysis_instruction}"
                    else:
                        enhanced_message = f"{request.message}\n\n{file_context}"

                    # Update context with file information
                    if 'conversations' not in context_data:
                        context_data['conversations'] = []
                    context_data['conversations'].append(f"[File Upload] User uploaded {len(request.files)} file(s): {', '.join([f.get('name', 'Unknown file') for f in request.files])}")
                except Exception as e:
                    logger.error(f"Error processing file attachments: {e}")
                    # Fallback to original message if file processing fails
                    enhanced_message = request.message
            else:
                enhanced_message = request.message

            # Check if this is a trade query (Phase 4)
            if is_trade_query(request.message):
                logger.info("🎯 Trade query detected in streaming mode")

                # Extract symbol from message
                symbol = extract_symbol(request.message)

                if symbol:
                    # Query live analysis from ChromaDB
                    live_analysis = query_live_analysis(symbol, request.timeframe or "M15")

                    if live_analysis:
                        # Found recent analysis
                        enhanced_message = f"""{enhanced_message}

📊 LIVE MARKET ANALYSIS for {symbol}:
{format_live_analysis(live_analysis)}

Please provide a comprehensive trade recommendation based on this analysis.
Explain the setup, entry/SL/TP levels, and reasoning clearly."""
                    else:
                        # No recent analysis found
                        enhanced_message = f"""{enhanced_message}

⚠️ No recent live analysis found for {symbol}.
Please upload latest data first:
curl -F "file=@{symbol}_M15_200.csv" http://localhost:8080/upload

Or check historical patterns from RAG knowledge base."""
                else:
                    enhanced_message = enhanced_message

            # Add web context if available
            elif web_context:
                enhanced_message = f"""{enhanced_message}

📡 REAL-TIME INFORMATION FROM THE WEB:
{web_context}

Please provide a comprehensive answer using both the above real-time information and your knowledge."""

            # Build contextual prompt with enhanced memory and file content
            messages = build_enhanced_contextual_prompt(enhanced_message, context_data)

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

    # Declare global at the beginning
    global ollama_client

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
            model=model_name
        )

        # Update global client reference
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

@app.post("/api/upload", response_model=Dict[str, Any])
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Upload a file and return its content for AI processing
    Args:
        request: FastAPI request object for authentication
        file: Uploaded file
    Returns:
        Dictionary with file information and content
    """
    # Check authentication
    get_current_user(auth_manager, request)

    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{file_id}{file_extension}"
        file_path = os.path.join(uploads_dir, safe_filename)

        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Determine file type and process content
        file_content = None
        if file.content_type.startswith('image/'):
            # For images, encode as base64 and store the reference
            file_content = {
                "type": "image",
                "format": file.content_type,
                "size": len(content),
                "encoding": "base64",
                "data": base64.b64encode(content).decode('utf-8')
            }
        elif file.content_type.startswith('text/') or file.filename.endswith('.txt'):
            # For text files, read the content directly
            try:
                text_content = content.decode('utf-8')
                file_content = {
                    "type": "text",
                    "format": file.content_type,
                    "size": len(text_content),
                    "content": text_content[:10000]  # Limit to 10k characters
                }
            except UnicodeDecodeError:
                file_content = {
                    "type": "binary",
                    "format": file.content_type,
                    "size": len(content),
                    "note": "Binary file - content not displayed"
                }
        else:
            # For other files (PDF, etc.), provide metadata
            file_content = {
                "type": "document",
                "format": file.content_type,
                "size": len(content),
                "note": f"Document file uploaded for processing"
            }

        result = {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "url": f"/api/files/{file_id}",
            "content": file_content
        }

        logger.info(f"File uploaded successfully: {file.filename} ({len(content)} bytes)")
        return result

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        return {
            "success": False,
            "message": f"Failed to upload file: {str(e)}"
        }

async def process_full_history_to_rag(file_path: str, symbol: str, timeframe: str):
    """
    Background task to process full history (*_0.csv) files through RAG pipeline

    Pipeline: CSV → Structured JSON → Pattern Detection → ChromaDB

    Args:
        file_path: Path to the uploaded CSV file
        symbol: Trading symbol (e.g., XAUUSD)
        timeframe: Timeframe (e.g., M15, H1)
    """
    import subprocess
    import os

    try:
        logger.info(f"🚀 Starting RAG pipeline for {file_path}")
        logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}")

        # Run the process_pipeline.sh script
        pipeline_script = "./scripts/process_pipeline.sh"

        if not os.path.exists(pipeline_script):
            logger.error(f"Pipeline script not found: {pipeline_script}")
            return

        # Execute pipeline in background
        result = subprocess.run(
            [pipeline_script, file_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        if result.returncode == 0:
            logger.info(f"✅ RAG pipeline completed successfully for {file_path}")
            logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        else:
            logger.error(f"❌ RAG pipeline failed for {file_path}")
            logger.error(f"Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error(f"⏱️  RAG pipeline timeout for {file_path}")
    except Exception as e:
        logger.error(f"💥 RAG pipeline error for {file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def process_live_data_analysis(file_path: str, symbol: str, timeframe: str):
    """
    Background task to process live data (*_200.csv) files through live analysis pipeline

    Pipeline: CSV → Live Trading Analyzer → Trade Recommendations → ChromaDB Storage

    Args:
        file_path: Path to the uploaded CSV file
        symbol: Trading symbol (e.g., XAUUSD)
        timeframe: Timeframe (e.g., M15, H1)
    """
    import subprocess
    import os
    import sys

    try:
        logger.info(f"🎯 Starting live analysis for {file_path}")
        logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}")

        # Get the scripts directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, "scripts")

        # Path to live trading analyzer
        analyzer_script = os.path.join(scripts_dir, "live_trading_analyzer.py")

        if not os.path.exists(analyzer_script):
            logger.error(f"Live trading analyzer not found: {analyzer_script}")
            return

        # Build command with --add-to-rag flag for ChromaDB storage
        cmd = [
            sys.executable, analyzer_script,
            "--input", file_path,
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--add-to-rag"  # Store in ChromaDB
        ]

        # Set environment variables for proper path resolution
        env = os.environ.copy()
        env["PYTHONPATH"] = scripts_dir + ":" + env.get("PYTHONPATH", "")

        logger.info(f"📊 Executing: {' '.join(cmd)}")

        # Execute live analysis in background
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout (faster than RAG pipeline)
            env=env,
            cwd=current_dir  # Run from current directory
        )

        if result.returncode == 0:
            logger.info(f"✅ Live analysis completed successfully for {file_path}")
            logger.info(f"   Output: {result.stdout}")

            # Update live file in data/live/ directory
            live_dir = os.path.join(current_dir, "data", "live")
            os.makedirs(live_dir, exist_ok=True)

            live_filename = f"{symbol.upper()}_{timeframe}_LIVE.csv"
            live_path = os.path.join(live_dir, live_filename)

            # Copy file to live directory
            import shutil
            shutil.copy2(file_path, live_path)
            logger.info(f"📁 Copied to live directory: {live_path}")

        else:
            logger.error(f"❌ Live analysis failed for {file_path}")
            logger.error(f"Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error(f"⏱️  Live analysis timeout for {file_path}")
    except Exception as e:
        logger.error(f"💥 Live analysis error for {file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())


@app.post("/upload", response_model=Dict[str, Any])
async def upload_mt5_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    timeframe: Optional[str] = Form(None),
    candles: Optional[str] = Form(None)
):
    """
    Upload MT5 CSV data file for live trading analysis
    Receives CSV files from MT5 EA and saves them with proper naming convention

    Args:
        request: FastAPI request object (no authentication required for EA)
        file: Uploaded CSV file from MT5 EA
        symbol: Trading symbol (e.g., XAUUSD, BTCUSD) - optional
        timeframe: Timeframe (e.g., M1, M5, M15, M30, H1, H4, H8, D1, W1, MN1) - optional
        candles: Number of candles in the file - optional

    Returns:
        Dictionary with upload status and file information
    """
    logger.info(f"MT5 CSV Upload Request: file={file.filename}, symbol={symbol}, timeframe={timeframe}, candles={candles}")
    logger.info(f"Request headers: {dict(request.headers)}")

    try:
        # Check if file was actually uploaded
        if not file or not file.filename:
            logger.error("No file provided in upload request")
            return {
                "success": False,
                "message": "No file provided in upload request",
                "error": "no_file_provided"
            }

        logger.info(f"Processing file: {file.filename}, size: {file.size}, content_type: {file.content_type}")

        # Validate inputs (more flexible - allow missing form fields)
        if not file.filename.endswith('.csv'):
            return {
                "success": False,
                "message": "Only CSV files are allowed",
                "error": "invalid_file_type",
                "received_filename": file.filename
            }

        # Extract metadata from filename if form fields are missing
        if not symbol or not timeframe or not candles:
            logger.info("Form fields missing, attempting to extract from filename")
            filename_pattern = r'^([A-Z_]+)_PERIOD_([A-Z0-9]+)_(\d+)\.csv$'
            match = re.match(filename_pattern, file.filename.upper())

            if match:
                symbol = symbol or match.group(1)
                timeframe = timeframe or match.group(2)
                candles = candles or match.group(3)
                logger.info(f"Extracted from filename: symbol={symbol}, timeframe={timeframe}, candles={candles}")
            else:
                # Try to extract from common patterns
                parts = file.filename.replace('.csv', '').upper().split('_')
                logger.info(f"Filename parts: {parts}")

                if len(parts) >= 3:
                    symbol = symbol or parts[0]
                    timeframe = timeframe or parts[2] if parts[2] in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1', 'W1', 'MN1'] else 'M15'
                    try:
                        candles = candles or next((p for p in parts if p.isdigit()), '200')
                    except StopIteration:
                        candles = '200'
                    logger.info(f"Extracted from parts: symbol={symbol}, timeframe={timeframe}, candles={candles}")
                else:
                    # Default values if we can't extract
                    symbol = symbol or file.filename.split('_')[0].replace('.CSV', '')
                    timeframe = timeframe or 'M15'
                    candles = candles or '200'
                    logger.info(f"Using defaults: symbol={symbol}, timeframe={timeframe}, candles={candles}")

        logger.info(f"Final parameters: symbol={symbol}, timeframe={timeframe}, candles={candles}")

        # Validate symbol format (letters and underscores only)
        if not re.match(r'^[A-Z_]+$', symbol.upper()):
            return {
                "success": False,
                "message": f"Invalid symbol format: {symbol}. Use format like XAUUSD, BTCUSD",
                "error": "invalid_symbol",
                "received_symbol": symbol
            }

        # Validate timeframe format (comprehensive MT5 timeframe support)
        valid_timeframes = [
            # Minutes
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30',
            # Hours
            'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12',
            # Days
            'D1',
            # Weeks
            'W1',
            # Months
            'MN1'
        ]
        timeframe_upper = timeframe.upper()
        if timeframe_upper not in valid_timeframes:
            return {
                "success": False,
                "message": f"Invalid timeframe: {timeframe}. Valid: {', '.join(valid_timeframes)}",
                "error": "invalid_timeframe",
                "received_timeframe": timeframe
            }

        # Validate candles count
        try:
            candles_int = int(candles)

            # If candles count is 0 or invalid, set a reasonable default based on timeframe
            if candles_int < 1:
                timeframe_upper = timeframe.upper()
                if timeframe_upper in ['M1', 'M5']:
                    candles_int = 1000  # 1000 candles for minute timeframes
                elif timeframe_upper in ['M15']:
                    candles_int = 500   # 500 candles for M15
                elif timeframe_upper in ['M30', 'H1']:
                    candles_int = 200   # 200 candles for M30/H1
                elif timeframe_upper in ['H4']:
                    candles_int = 100   # 100 candles for H4
                else:
                    candles_int = 200   # Default 200 candles

                logger.warning(f"Invalid candles count ({candles}), using default: {candles_int} for {timeframe_upper}")

            if candles_int > 10000:
                return {
                    "success": False,
                    "message": f"Invalid candles count: {candles}. Must be between 1 and 10000",
                    "error": "invalid_candles",
                    "received_candles": candles
                }
        except ValueError:
            # If parsing fails, set default based on timeframe
            timeframe_upper = timeframe.upper()
            if timeframe_upper in ['M1', 'M5']:
                candles_int = 1000
            elif timeframe_upper in ['M15']:
                candles_int = 500
            elif timeframe_upper in ['M30', 'H1']:
                candles_int = 200
            elif timeframe_upper in ['H4']:
                candles_int = 100
            else:
                candles_int = 200

            logger.warning(f"Failed to parse candles count ({candles}), using default: {candles_int} for {timeframe_upper}")

        # Ensure data directory exists
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Use original filename to preserve naming convention
        filename = file.filename
        file_path = os.path.join(data_dir, filename)

        logger.info(f"Using original filename: {filename}")

        logger.info(f"Reading file content...")

        # Read and validate CSV content
        content = await file.read()
        logger.info(f"File content read: {len(content)} bytes")

        if not content:
            logger.error("Empty file content")
            return {
                "success": False,
                "message": "Empty file content",
                "error": "empty_file"
            }

        # Log first few bytes of content for debugging
        logger.info(f"File content preview (first 100 bytes): {content[:100]}")

        try:
            # Try to decode content
            content_str = None
            encoding_used = "utf-8"

            try:
                content_str = content.decode('utf-8')
                logger.info("Successfully decoded as UTF-8")
            except UnicodeDecodeError:
                try:
                    content_str = content.decode('utf-16-le')
                    encoding_used = "utf-16-le"
                    logger.info("Successfully decoded as UTF-16 LE")
                except UnicodeDecodeError:
                    try:
                        content_str = content.decode('utf-16-be')
                        encoding_used = "utf-16-be"
                        logger.info("Successfully decoded as UTF-16 BE")
                    except UnicodeDecodeError:
                        # Try latin-1 as fallback
                        content_str = content.decode('latin-1', errors='replace')
                        encoding_used = "latin-1"
                        logger.warning("Decoded as latin-1 with replacement")

            if not content_str.strip():
                logger.error("Content is empty after decoding")
                return {
                    "success": False,
                    "message": "File content is empty or unreadable",
                    "error": "empty_content"
                }

            logger.info(f"Content sample (first 200 chars): {content_str[:200]}")

            # Validate CSV structure using pandas
            from io import StringIO
            df_test = pd.read_csv(StringIO(content_str))
            logger.info(f"Successfully parsed CSV: {len(df_test)} rows, {len(df_test.columns)} columns")
            logger.info(f"CSV columns: {list(df_test.columns)}")

            # Check if this is already RAG format or MT5 format
            rag_columns = ['TimeFrame', 'Symbol', 'Candle', 'DateTime', 'Open', 'High', 'low', 'Close', 'Volume', 'HL', 'Body']
            mt5_columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

            is_rag_format = all(col in df_test.columns for col in ['TimeFrame', 'Symbol', 'DateTime'])
            is_mt5_format = all(col in df_test.columns for col in ['timestamp', 'open', 'high', 'low', 'close'])

            if not is_rag_format and not is_mt5_format:
                logger.error(f"Unrecognized CSV format. Found columns: {list(df_test.columns)}")
                return {
                    "success": False,
                    "message": f"Unrecognized CSV format. Expected RAG or MT5 format.",
                    "found_columns": list(df_test.columns),
                    "error": "unrecognized_format"
                }

            # If it's RAG format, skip conversion and use content as-is
            if is_rag_format:
                logger.info("Detected RAG format - using content directly without conversion")
                rag_content = content_str.encode('utf-8')

                # Extract candle count from actual data
                actual_rows = len(df_test)
                candles_int = actual_rows  # Update to match actual rows

                # Validate row count
                if abs(actual_rows - candles_int) > 10:  # Allow small discrepancy
                    logger.warning(f"Row count mismatch: expected {candles_int}, found {actual_rows}")

                logger.info(f"RAG format validated: {actual_rows} rows")
            else:
                # Original MT5 format validation
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
                missing_columns = [col for col in required_columns if col not in df_test.columns]

                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    return {
                        "success": False,
                        "message": f"CSV missing required columns: {', '.join(missing_columns)}",
                        "required_columns": required_columns,
                        "found_columns": list(df_test.columns),
                        "error": "invalid_csv_structure"
                    }

            # Continue validation for MT5 format only
            if not is_rag_format:
                # Validate row count matches expected candles
                actual_rows = len(df_test)
                if abs(actual_rows - candles_int) > 10:  # Allow small discrepancy
                    logger.warning(f"Row count mismatch: expected {candles_int}, found {actual_rows}")
                    return {
                        "success": False,
                        "message": f"Row count mismatch. Expected: {candles_int}, Found: {actual_rows}",
                        "error": "row_count_mismatch"
                    }

                # Validate data format (check first few rows)
                try:
                    # Test timestamp format
                    pd.to_datetime(df_test['timestamp'].head())

                    # Test numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close']
                    for col in numeric_cols:
                        pd.to_numeric(df_test[col].head())

                    # Test integer columns
                    int_cols = ['tick_volume', 'spread', 'real_volume']
                    for col in int_cols:
                        pd.to_numeric(df_test[col].head())

                    logger.info("MT5 data format validation passed")
                except Exception as e:
                    logger.error(f"MT5 CSV data format validation failed: {e}")
                    return {
                        "success": False,
                        "message": f"MT5 CSV data format validation failed: {str(e)}",
                        "error": "invalid_data_format"
                    }
            else:
                # For RAG format, just validate basic structure
                actual_rows = len(df_test)
                logger.info(f"RAG format detected with {actual_rows} rows")

        except Exception as e:
            logger.error(f"CSV parsing failed: {e}")
            return {
                "success": False,
                "message": f"CSV parsing failed: {str(e)}",
                "error": "csv_parse_error"
            }

        # Convert MT5 format to RAG format only if it's not already RAG format
        if not is_rag_format:
            rag_content = convert_mt5_to_rag_format(content, symbol.upper(), timeframe_upper)
            logger.info("Converted MT5 format to RAG format")
        else:
            logger.info("Using existing RAG format content without conversion")

        # Save the file in RAG format
        logger.info(f"Saving file to: {file_path}")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(rag_content)

        # Also save to live data directory for intraday timeframes (for live trading)
        live_updated = False
        live_path = None
        intraday_timeframes = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12']

        if timeframe_upper in intraday_timeframes:
            live_dir = os.path.join(data_dir, "live")
            os.makedirs(live_dir, exist_ok=True)
            live_filename = f"{symbol.upper()}_{timeframe_upper}_LIVE.csv"
            live_path = os.path.join(live_dir, live_filename)

            logger.info(f"Saving live file to: {live_path}")
            async with aiofiles.open(live_path, 'wb') as f:
                await f.write(rag_content)

            live_updated = True
            logger.info(f"Live data updated: {live_filename} (RAG format)")

        # Check if this is a full history file (*_0.csv) and trigger RAG pipeline
        rag_processing = False
        live_processing = False
        if filename.endswith("_0.csv") or filename.endswith("_0.CSV"):
            logger.info(f"📊 Full history file detected: {filename}")
            logger.info(f"   Scheduling RAG pipeline processing in background...")

            # Trigger background processing
            background_tasks.add_task(
                process_full_history_to_rag,
                file_path,
                symbol.upper(),
                timeframe_upper
            )
            rag_processing = True
            logger.info(f"   ✅ RAG pipeline scheduled for background processing")
        elif filename.endswith("_200.csv") or filename.endswith("_200.CSV"):
            logger.info(f"📈 Live data file detected: {filename}")
            logger.info(f"   Scheduling live analysis in background...")

            # Trigger live analysis background task (Phase 3)
            background_tasks.add_task(
                process_live_data_analysis,
                file_path,
                symbol.upper(),
                timeframe_upper
            )
            live_processing = True
            logger.info(f"   ✅ Live analysis scheduled for background processing")
        else:
            logger.info(f"📁 Standard file: {filename}")
            logger.info(f"   Saved to data directory")

        # Return success response
        # Update success message based on file type
        if rag_processing:
            message = f"Full history file uploaded. RAG pipeline processing started in background."
        elif live_processing:
            message = f"Live data uploaded. Live analysis started in background."
        elif filename.endswith("_200.csv") or filename.endswith("_200.CSV"):
            message = f"Live data uploaded successfully. Ready for LLM analysis."
        else:
            message = f"MT5 CSV data uploaded successfully"

        result = {
            "success": True,
            "message": message,
            "filename": filename,
            "filepath": file_path,
            "symbol": symbol.upper(),
            "timeframe": timeframe_upper,
            "candles": candles_int,
            "actual_rows": actual_rows,
            "file_size": len(content),
            "encoding": encoding_used,
            "live_updated": live_updated,
            "live_path": live_path if live_updated else None,
            "rag_processing": rag_processing,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ MT5 CSV Upload Successful: {symbol} {timeframe_upper} {candles_int} candles -> {filename}")
        return result

    except Exception as e:
        logger.error(f"❌ MT5 CSV Upload Failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"Upload failed: {str(e)}",
            "error": "upload_failed"
        }

@app.post("/upload/simple", response_model=Dict[str, Any])
async def upload_mt5_csv_simple(request: Request):
    """
    Alternative simple upload endpoint that doesn't use FastAPI's multipart parsing
    This can handle problematic multipart requests from MT5 EA

    Args:
        request: FastAPI request object

    Returns:
        Dictionary with upload status
    """
    logger.info("Using simple upload endpoint (alternative method)")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Content-Type: {request.headers.get('content-type', 'Unknown')}")
    logger.info(f"Content-Length: {request.headers.get('content-length', 'Unknown')}")

    try:
        # Read raw request body
        body = await request.body()
        logger.info(f"Received raw body: {len(body)} bytes")

        if not body:
            return {
                "success": False,
                "message": "Empty request body",
                "error": "empty_body"
            }

        # For now, let's try to save the body as a file with a default name
        # This is a very basic approach that should work with most multipart formats
        import hashlib
        file_hash = hashlib.md5(body).hexdigest()[:8]

        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Use a simple filename approach
        filename = f"MT5_UPLOAD_{file_hash}.csv"
        file_path = os.path.join(data_dir, filename)

        logger.info(f"Saving file to: {file_path}")

        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(body)

        # Also save to live directory for intraday data (default to M15 for unknown)
        live_dir = os.path.join(data_dir, "live")
        os.makedirs(live_dir, exist_ok=True)
        live_filename = "UNKNOWN_M15_LIVE.csv"  # Default to M15 for simple uploads
        live_path = os.path.join(live_dir, live_filename)

        async with aiofiles.open(live_path, 'wb') as f:
            await f.write(body)

        return {
            "success": True,
            "message": "File uploaded successfully via simple endpoint",
            "filename": filename,
            "filepath": file_path,
            "method": "simple",
            "file_size": len(body),
            "live_updated": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Simple upload failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "message": f"Simple upload failed: {str(e)}",
            "error": "simple_upload_failed"
        }

@app.get("/upload/status", response_model=Dict[str, Any])
async def upload_status():
    """
    Get status of uploaded MT5 data files

    Returns:
        Dictionary with available data files and their information
    """
    try:
        data_dir = "data"
        live_dir = os.path.join(data_dir, "live")

        # Get all MT5 data files
        mt5_files = []

        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv') and '_PERIOD_' in filename:
                    file_path = os.path.join(data_dir, filename)
                    stat = os.stat(file_path)

                    # Parse filename to extract info
                    try:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 4 and parts[1] == 'PERIOD':
                            symbol = parts[0]
                            timeframe = parts[2]
                            candles = parts[3]

                            mt5_files.append({
                                "filename": filename,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "candles": candles,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "path": file_path
                            })
                    except:
                        continue

        # Get live data files
        live_files = []
        if os.path.exists(live_dir):
            for filename in os.listdir(live_dir):
                if filename.endswith('_LIVE.csv'):
                    file_path = os.path.join(live_dir, filename)
                    stat = os.stat(file_path)

                    # Extract symbol from various live file formats
                    symbol = filename
                    for tf in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12']:
                        symbol = symbol.replace(f'_{tf}_LIVE.csv', '')
                    symbol = symbol.replace('UNKNOWN_', '')  # Handle unknown symbol case
                    live_files.append({
                        "filename": filename,
                        "symbol": symbol,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "path": file_path
                    })

        return {
            "success": True,
            "mt5_files": mt5_files,
            "live_files": live_files,
            "total_mt5_files": len(mt5_files),
            "total_live_files": len(live_files),
            "data_directory": data_dir,
            "live_directory": live_dir,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        return {
            "success": False,
            "message": f"Failed to get status: {str(e)}",
            "mt5_files": [],
            "live_files": []
        }

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

@app.post("/query", response_model=Dict[str, Any])
async def query_endpoint(request: Request, query_request: Dict[str, Any]):
    """
    Enhanced query endpoint that supports trading analysis with LLM integration

    Args:
        query_request: Dictionary with:
            - query (str): The question/query text
            - model (str, optional): LLM model to use (default: gemma3:1b)
            - max_context (int, optional): Number of RAG context items (default: 5)

    Example:
        {
            "query": "Give me best trade setup for XAUUSD",
            "model": "llama3.1:8b",
            "max_context": 5
        }
    """
    try:
        logger.info(f"Received query request: {str(query_request)[:100]}...")

        # Get the raw query from the request body
        query_text = query_request.get("query", "")

        # Get model selection (default: gemma3:1b)
        model = query_request.get("model", "gemma3:1b")
        logger.info(f"Using LLM model: {model}")

        # Enhance query with RAG context
        context_data = rag_enhancer.enhance_query_with_rag(query_text, max_context=query_request.get("max_context", 5))

        # Check if this is a trading-related query
        is_trading_query = _is_trading_query(query_text)

        result = {
            "query": query_text,
            "model_used": model,
            "context": context_data,
            "is_trading_query": is_trading_query
        }

        # If trading query, get LLM analysis
        if is_trading_query and _is_ollama_available():
            try:
                llm_analysis = await get_llm_trading_analysis(query_text, context_data, model=model)
                result["llm_analysis"] = llm_analysis
                result["enhanced_recommendation"] = generate_enhanced_recommendation_from_analysis(context_data, llm_analysis)
            except Exception as e:
                logger.error(f"Failed to get LLM analysis: {e}")
                result["llm_error"] = str(e)

        return result

    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/api/enhanced-analysis")
async def enhanced_live_analysis(http_request: Request):
    """Enhanced live analysis endpoint for comprehensive multi-timeframe analysis"""
    try:
        user = get_current_user(auth_manager, http_request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")

        data = await http_request.json()
        file_path = data.get('file_path')
        save_rag = data.get('save_rag', True)

        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")

        # Validate file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        logger.info(f"Enhanced analysis request for {file_path}")

        # Extract symbol from filename for multi-timeframe analysis
        import re
        filename = os.path.basename(file_path)
        symbol_match = re.match(r'^([A-Z0-9]+)_PERIOD_([MH][0-9,]+[DW]?)_200\.csv$', filename)

        if not symbol_match:
            raise HTTPException(status_code=400, detail="Invalid filename format. Expected: SYMBOL_PERIOD_TIMEFRAME_200.csv")

        symbol = symbol_match.group(1)
        timeframe = symbol_match.group(2)

        # Use the multi-timeframe analyzer
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

        try:
            from multi_timeframe_analyzer import process_symbol_all_timeframes
        except ImportError as e:
            logger.error(f"Failed to import multi_timeframe_analyzer: {e}")
            logger.error(f"Current Python path: {sys.path}")
            raise HTTPException(status_code=500, detail=f"Import error: {e}")

        logger.info(f"Processing multi-timeframe analysis for symbol: {symbol}")

        # Process all timeframes for the symbol
        success = process_symbol_all_timeframes("./data", symbol, "./data/rag_processed")

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to process multi-timeframe analysis for {symbol}")

        # Load the comprehensive analysis
        analysis_file = os.path.join("./data/rag_processed", f"{symbol}.json")
        if not os.path.exists(analysis_file):
            raise HTTPException(status_code=500, detail=f"Analysis file not found: {analysis_file}")

        def clean_nan_values(obj):
                """Recursively clean NaN and infinite values from data structure"""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    else:
                        return obj
                else:
                    return obj

        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                comprehensive_analysis = json.load(f)

            # Clean NaN values from the analysis
            comprehensive_analysis = clean_nan_values(comprehensive_analysis)

            logger.info(f"Successfully loaded multi-timeframe analysis for {symbol}")
            logger.info(f"  - Timeframes analyzed: {len(comprehensive_analysis['timeframe_analysis'])}")
            logger.info(f"  - Total coverage: {comprehensive_analysis['overall_market_context']['time_coverage_days']:.1f} days")

            return {
                "success": True,
                "comprehensive_analysis": comprehensive_analysis,
                "file_info": {
                    "symbol": symbol,
                    "original_file": filename,
                    "timeframe": timeframe,
                    "analysis_type": "multi-timeframe",
                    "total_timeframes": len(comprehensive_analysis['timeframe_analysis']),
                    "coverage_days": comprehensive_analysis['overall_market_context']['time_coverage_days']
                }
            }

        except Exception as e:
            logger.error(f"Error loading analysis file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load analysis: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.get("/api/available-files")
async def get_available_files(http_request: Request):
    """Get list of available CSV files for analysis"""
    try:
        user = get_current_user(auth_manager, http_request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")

        data_dir = "./data"
        if not os.path.exists(data_dir):
            return {"files": []}

        csv_files = []
        for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            stat = os.stat(file_path)
            csv_files.append({
                "name": os.path.basename(file_path),
                "path": file_path,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: x['modified'], reverse=True)

        return {"files": csv_files}

    except Exception as e:
        logger.error(f"Available files endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get files: {str(e)}")

@app.get("/api/available-models")
async def get_available_models_endpoint(http_request: Request):
    """Get list of available LLM models"""
    try:
        user = get_current_user(auth_manager, http_request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")

        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                return {"models": models}
        except:
            pass

        # Fallback models
        return {"models": ['gemma3:1b', 'gemma2:2b', 'qwen3:0.6b', 'qwen3:14b']}

    except Exception as e:
        logger.error(f"Available models endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

def _is_trading_query(query: str) -> bool:
    """Check if query is related to trading/finance"""
    trading_keywords = [
        'trade', 'trading', 'buy', 'sell', 'market', 'stock', 'forex', 'crypto',
        'xauusd', 'gold', 'silver', 'currency', 'price', 'chart', 'indicator',
        'rsi', 'macd', 'sma', 'ema', 'vwap', 'support', 'resistance',
        'entry', 'exit', 'stop loss', 'take profit', 'profit', 'loss'
    ]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in trading_keywords)

def _is_ollama_available() -> bool:
    """Check if Ollama is available"""
    try:
        ollama_client.check_model()
        return True
    except:
        return False

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

# ============================================================================
# Web Search and Trading News Endpoints
# ============================================================================

@app.get("/api/web/search")
async def web_search_endpoint(
    query: str,
    max_results: int = 5,
    request: Request = None
):
    """
    Search the web for information

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        request: FastAPI request for authentication

    Returns:
        Search results
    """
    get_current_user(auth_manager, request)
    logger.info(f"Web search requested: {query}")

    try:
        results = web_search_tool.search_web(query, max_results)
        return {
            "success": True,
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/search")
async def news_search_endpoint(
    query: str,
    max_results: int = 5,
    request: Request = None
):
    """
    Search for recent news articles

    Args:
        query: Search query string
        max_results: Maximum number of articles (default: 5)
        request: FastAPI request for authentication

    Returns:
        News search results
    """
    get_current_user(auth_manager, request)
    logger.info(f"News search requested: {query}")

    try:
        results = web_search_tool.search_news(query, max_results)
        return {
            "success": True,
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"News search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/symbol/{symbol}")
async def symbol_news_endpoint(
    symbol: str,
    request: Request = None
):
    """
    Get latest news for a specific trading symbol

    Args:
        symbol: Trading symbol (e.g., XAUUSD, EURUSD, BTCUSD)
        request: FastAPI request for authentication

    Returns:
        Symbol-specific news
    """
    get_current_user(auth_manager, request)
    logger.info(f"Symbol news requested: {symbol}")

    try:
        news = trading_news_api.get_symbol_news(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "news": news,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Symbol news error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/market-sentiment/{symbol}")
async def market_sentiment_endpoint(
    symbol: str,
    request: Request = None
):
    """
    Get current market sentiment for a symbol

    Args:
        symbol: Trading symbol
        request: FastAPI request for authentication

    Returns:
        Market sentiment analysis
    """
    get_current_user(auth_manager, request)
    logger.info(f"Market sentiment requested: {symbol}")

    try:
        sentiment = trading_news_api.get_market_sentiment(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/economic-calendar")
async def economic_calendar_endpoint(request: Request = None):
    """
    Get today's important economic events

    Args:
        request: FastAPI request for authentication

    Returns:
        Economic calendar data
    """
    get_current_user(auth_manager, request)
    logger.info("Economic calendar requested")

    try:
        calendar = trading_news_api.get_economic_calendar()
        return {
            "success": True,
            "calendar": calendar,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Economic calendar error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/market-overview")
async def market_overview_endpoint(request: Request = None):
    """
    Get general market overview

    Args:
        request: FastAPI request for authentication

    Returns:
        Market overview data
    """
    get_current_user(auth_manager, request)
    logger.info("Market overview requested")

    try:
        overview = trading_news_api.get_market_overview()
        return {
            "success": True,
            "overview": overview,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/technical-analysis/{symbol}")
async def technical_analysis_news_endpoint(
    symbol: str,
    request: Request = None
):
    """
    Get latest technical analysis for a symbol

    Args:
        symbol: Trading symbol
        request: FastAPI request for authentication

    Returns:
        Technical analysis from news sources
    """
    get_current_user(auth_manager, request)
    logger.info(f"Technical analysis news requested: {symbol}")

    try:
        analysis = trading_news_api.get_technical_analysis(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Technical analysis news error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/web-enhanced")
async def web_enhanced_chat_endpoint(
    request: ChatRequest,
    http_request: Request
):
    """
    Enhanced chat endpoint with automatic web search integration

    Automatically detects when web search is needed and enriches
    the LLM context with real-time information from the internet.

    Args:
        request: ChatRequest containing message and parameters
        http_request: FastAPI request for authentication

    Returns:
        ChatResponse with web-enhanced AI response
    """
    get_current_user(auth_manager, http_request)
    logger.info(f"Web-enhanced chat request: {request.message[:100]}...")

    try:
        # Check if web search is needed
        needs_web_search = trading_tools.should_use_web_search(request.message)

        web_context = ""
        if needs_web_search:
            logger.info("Web search triggered for query")

            # Determine what to search for
            if "news" in request.message.lower():
                # Extract symbol if mentioned
                symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"]
                symbol = next((s for s in symbols if s.lower() in request.message.lower()), "XAUUSD")
                web_context = trading_news_api.get_symbol_news(symbol)

            elif "calendar" in request.message.lower() or "event" in request.message.lower():
                web_context = trading_news_api.get_economic_calendar()

            elif "sentiment" in request.message.lower():
                symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
                symbol = next((s for s in symbols if s.lower() in request.message.lower()), "XAUUSD")
                web_context = trading_news_api.get_market_sentiment(symbol)

            elif "overview" in request.message.lower() or "market" in request.message.lower():
                web_context = trading_news_api.get_market_overview()

            else:
                # General web search
                web_context = web_search_tool.search_web(request.message, max_results=3)

        # Enhanced RAG context retrieval
        context_data = rag_enhancer.enhance_query_with_rag(request.message, max_context=request.memory_context)

        # Build enhanced prompt with web context if available
        if web_context:
            enhanced_message = f"""User question: {request.message}

📡 REAL-TIME INFORMATION FROM THE WEB:
{web_context}

Please provide a comprehensive answer using both the above real-time information and your knowledge."""
        else:
            enhanced_message = request.message

        # Build contextual prompt with enhanced memory
        messages = build_enhanced_contextual_prompt(enhanced_message, context_data)

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

        logger.info(f"Generated web-enhanced response of length: {len(ai_response)}")
        return response

    except Exception as e:
        logger.error(f"Web-enhanced chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live-analysis/{symbol}/{timeframe}")
async def get_live_analysis(symbol: str, timeframe: str):
    """
    Get latest live analysis for a symbol and timeframe

    Args:
        symbol: Trading symbol (e.g., XAUUSD, BTCUSD)
        timeframe: Timeframe (e.g., M15, H1, H4)

    Returns:
        Latest live analysis data
    """
    try:
        # Validate symbol
        symbol = symbol.upper()
        valid_symbols = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "US30", "NAS100", "SPX500"]
        if symbol not in valid_symbols:
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")

        # Validate timeframe
        timeframe = timeframe.upper()
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        if timeframe not in valid_timeframes:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")

        # Query live analysis
        analysis = query_live_analysis(symbol, timeframe)

        if not analysis:
            return {
                "success": False,
                "message": f"No recent analysis found for {symbol} {timeframe}",
                "suggestion": "Upload latest data first: curl -F \"file=@{symbol}_{timeframe}_200.csv\" http://localhost:8080/upload"
            }

        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/high-confidence-setups")
async def get_high_confidence_setups(symbol: Optional[str] = None, min_confidence: int = 70):
    """
    Get high confidence trade setups

    Args:
        symbol: Optional symbol filter
        min_confidence: Minimum confidence threshold (default 70)

    Returns:
        List of high confidence setups
    """
    try:
        # Import ChromaLiveAnalyzer
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, "scripts")
        sys.path.append(scripts_dir)

        from chroma_live_analyzer import ChromaLiveAnalyzer

        # Validate min_confidence
        if not 0 <= min_confidence <= 100:
            raise HTTPException(status_code=400, detail="min_confidence must be between 0 and 100")

        # Validate symbol if provided
        if symbol:
            symbol = symbol.upper()
            valid_symbols = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "ETHUSD", "US30", "NAS100", "SPX500"]
            if symbol not in valid_symbols:
                raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")

        analyzer = ChromaLiveAnalyzer()
        setups = analyzer.get_high_confidence_setups(symbol, min_confidence)

        return {
            "success": True,
            "count": len(setups),
            "filters": {
                "symbol": symbol,
                "min_confidence": min_confidence
            },
            "setups": setups,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"High confidence setups endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live-analysis-stats")
async def get_live_analysis_stats():
    """
    Get statistics about live analysis collection

    Returns:
        Collection statistics
    """
    try:
        # Import ChromaLiveAnalyzer
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, "scripts")
        sys.path.append(scripts_dir)

        from chroma_live_analyzer import ChromaLiveAnalyzer

        analyzer = ChromaLiveAnalyzer()
        stats = analyzer.get_collection_stats()

        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Live analysis stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Local Financial Assistant API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)