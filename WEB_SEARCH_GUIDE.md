# Web Search Integration Guide

Your local LLM now has access to real-time internet information! This guide explains how to use the new web search capabilities.

## Features

### 1. Automatic Web Search in Chat

When you ask questions that need current information, the system automatically searches the web:

**Triggers automatic web search:**
- "What's the latest gold news?"
- "Show me today's economic calendar"
- "What is the current market sentiment for XAUUSD?"
- "Give me an overview of the forex market"

**Uses RAG knowledge only:**
- "What is RSI indicator?"
- "Calculate moving average"
- "Explain technical analysis"

### 2. Web Search API Endpoints

#### General Web Search
```bash
GET /api/web/search?query=XAUUSD+forecast&max_results=5
```

#### News Search
```bash
GET /api/news/search?query=gold+trading&max_results=5
```

#### Symbol-Specific News
```bash
GET /api/news/symbol/XAUUSD
GET /api/news/symbol/EURUSD
GET /api/news/symbol/BTCUSD
```

#### Market Sentiment
```bash
GET /api/news/market-sentiment/XAUUSD
```

#### Economic Calendar
```bash
GET /api/news/economic-calendar
```

#### Market Overview
```bash
GET /api/news/market-overview
```

#### Technical Analysis News
```bash
GET /api/news/technical-analysis/XAUUSD
```

### 3. Web-Enhanced Chat Endpoint

Use the enhanced chat endpoint for automatic web integration:

```bash
POST /api/chat/web-enhanced
Content-Type: application/json

{
  "message": "What's the latest news about gold prices?",
  "model": "gemma3:1b",
  "memory_context": 5
}
```

## Usage Examples

### Python Example

```python
import requests

# Get latest news for XAUUSD
response = requests.get(
    "http://localhost:8080/api/news/symbol/XAUUSD",
    headers={"Cookie": "session_token=YOUR_TOKEN"}
)
news = response.json()
print(news)

# Web-enhanced chat
response = requests.post(
    "http://localhost:8080/api/chat/web-enhanced",
    json={
        "message": "What's happening with gold prices today?",
        "model": "gemma3:1b"
    },
    headers={"Cookie": "session_token=YOUR_TOKEN"}
)
answer = response.json()
print(answer['response'])
```

### cURL Example

```bash
# Get economic calendar
curl "http://localhost:8080/api/news/economic-calendar" \
  -H "Cookie: session_token=YOUR_TOKEN"

# Search the web
curl "http://localhost:8080/api/web/search?query=XAUUSD+forecast&max_results=3" \
  -H "Cookie: session_token=YOUR_TOKEN"

# Get market sentiment
curl "http://localhost:8080/api/news/market-sentiment/XAUUSD" \
  -H "Cookie: session_token=YOUR_TOKEN"
```

## How It Works

### Architecture

```
User Query → Detection Layer → Web Search/RAG → Context Building → LLM → Response
```

1. **Query Analysis**: System detects if query needs real-time information
2. **Web Search**: Searches DuckDuckGo for relevant information
3. **Context Enhancement**: Combines web results with RAG knowledge
4. **LLM Processing**: LLM generates answer using both sources
5. **Response**: User receives comprehensive, up-to-date answer

### Supported Symbols

The system recognizes these common trading symbols:
- **Forex**: XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY
- **Crypto**: BTCUSD, ETHUSD
- Add more in `web_search.py`

### Search Provider

- **Provider**: DuckDuckGo (free, no API key needed)
- **Privacy**: No tracking, no personal data shared
- **Rate Limits**: Automatic handling with delays
- **Results**: 3-5 results per query (configurable)

## Configuration

### Customize Web Search

Edit `web_search.py` to customize:

```python
# Change number of results
web_search_tool.search_web(query, max_results=10)

# Add custom symbols
symbol_map = {
    "XAUUSD": "gold XAU/USD",
    "CUSTOM": "your custom search term"
}

# Modify search keywords
web_keywords = [
    "news", "today", "current", "latest",
    "your", "custom", "keywords"
]
```

### Adjust Auto-Detection

Modify `trading_tools.should_use_web_search()` to change when web search triggers:

```python
def should_use_web_search(self, query: str) -> bool:
    web_keywords = [
        "news", "today", "current", "latest", "now"
        # Add your keywords
    ]
    return any(keyword in query.lower() for keyword in web_keywords)
```

## Testing

Run the test script to verify everything works:

```bash
python3 test_web_search.py
```

Expected output:
```
✅ ALL TESTS PASSED!

Web search integration is working correctly!
Your LLM now has access to real-time internet information.
```

## Troubleshooting

### Rate Limiting

**Issue**: "202 Ratelimit" error
**Solution**:
- Wait a few seconds between requests
- DuckDuckGo has automatic rate limiting
- For production, consider caching results

### No Results Found

**Issue**: Search returns no results
**Solution**:
- Check internet connection
- Try different search terms
- DuckDuckGo may be temporarily unavailable
- System falls back to RAG knowledge

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'duckduckgo_search'`
**Solution**:
```bash
pip3 install -r requirements.txt
```

## Examples of Questions Your LLM Can Now Answer

**With Web Search:**
- "What's the latest gold price forecast?"
- "Show me today's forex news"
- "What are today's economic events?"
- "What's the current sentiment on EURUSD?"
- "Give me a market overview for today"

**Without Web Search (uses RAG):**
- "Explain moving averages"
- "What is a bull flag pattern?"
- "How do I calculate RSI?"
- "Analyze this price data" (with uploaded file)

## Benefits

✅ **Real-time Information**: Get current market news and forecasts
✅ **Economic Calendar**: Track important events automatically
✅ **Market Sentiment**: Understand current market psychology
✅ **Privacy-Focused**: Uses DuckDuckGo (no tracking)
✅ **Free**: No API costs or subscriptions
✅ **Automatic**: System detects when to search
✅ **Comprehensive**: Combines web + RAG knowledge

## Next Steps

1. **Start the server**: `python3 -m uvicorn main:app --port 8080 --reload`
2. **Test an endpoint**: Visit `/api/news/symbol/XAUUSD`
3. **Try web-enhanced chat**: Use `/api/chat/web-enhanced`
4. **Monitor logs**: Watch for web search triggers

## API Documentation

Full API documentation available at: `http://localhost:8080/docs`

---

**Note**: Web search respects DuckDuckGo's rate limits. For high-frequency applications, consider implementing caching or using alternative search APIs.

**Privacy**: All web searches go through DuckDuckGo, which doesn't track users. Your trading system remains private.
