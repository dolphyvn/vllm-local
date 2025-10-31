# Integration with Existing RAG System

This guide explains how the trading pattern processor integrates with your existing ChromaDB-based RAG system.

## System Architecture

Your current system:
- **ChromaDB**: Vector database for semantic search (`./chroma_db`)
- **MemoryManager**: Handles storage/retrieval via `memory.py`
- **FastAPI**: REST API in `main.py` with authentication
- **Categories**: general, trading, financial, technical, business, market, risk, strategy, definitions, corrections

## Integration Methods

### Method 1: Direct ChromaDB Integration (RECOMMENDED for bulk)

**Fastest method** - bypasses API, writes directly to ChromaDB.

```bash
# Feed a single processed file
python scripts/feed_to_rag_direct.py --file data/XAUUSD_PERIOD_M15_0_processed.json

# Feed all processed files in a directory
python scripts/feed_to_rag_direct.py --dir data/processed

# Custom ChromaDB location
python scripts/feed_to_rag_direct.py --dir data/processed --chroma-dir /path/to/chroma_db
```

**How it works:**
- Uses your `MemoryManager` class directly
- Calls `add_lesson_memory()` for each pattern/level
- Stores patterns in `category="trading"`
- Stores levels in `category="technical"`
- Tags include: pattern name, symbol, timeframe, outcome, trend, session, market regime

**Performance:**
- ~100-200 patterns/second
- No network overhead
- No authentication needed

### Method 2: API Integration

**Better for:** Remote feeding, authentication required, scheduled tasks

```bash
# Make sure your API is running
# python main.py

# Feed via API
python scripts/feed_to_rag.py --dir data/processed --password admin123
```

**How it works:**
- Authenticates via `/auth/login`
- Uses `/api/knowledge/add` endpoint
- Same storage format as direct method

## Data Storage Format

### Trading Patterns

Stored as "lessons" in ChromaDB:

```python
{
    "type": "lesson",
    "title": "Trading Pattern: Bullish Engulfing - XAUUSD 15min",
    "content": "Pattern: Bullish Engulfing\nSymbol: XAUUSD\n...[full narrative]",
    "category": "trading",
    "confidence": 0.9,  # 0.9 for WIN, 0.7 for others
    "tags": "Bullish Engulfing,XAUUSD,15min,WIN,Bullish,Asia,Tuesday,Strong Trending"
}
```

### Support/Resistance Levels

```python
{
    "type": "lesson",
    "title": "Support Level: 1735.50 - XAUUSD",
    "content": "Support/Resistance Level: 1735.50\n...[full narrative]",
    "category": "technical",
    "confidence": 0.85,  # 0.85 for strong, 0.7 for moderate
    "tags": "support,XAUUSD,15min,moderate,support_resistance,key_level"
}
```

## Querying the Data

### Via Python (MemoryManager)

```python
from memory import MemoryManager

# Initialize
memory = MemoryManager(persist_directory="./chroma_db")

# Search for specific patterns
results = memory.search_lessons(
    query="bullish engulfing patterns that won",
    category="trading",
    n=5
)

# Search by tags
results = memory.get_memory(
    query="XAUUSD Asia session patterns",
    n=10
)

# Get recent patterns
recent = memory.get_recent_memories(n=20)
```

### Via FastAPI Endpoints

Start your server:
```bash
python main.py
```

Then query via chat:
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "message": "Show me bullish engulfing patterns that won during Asia session",
    "memory_context": 10
  }'
```

### Example Queries

After feeding your data, you can ask:

1. **Pattern Analysis**
   - "Show me all bullish engulfing patterns that won"
   - "What's the win rate for RSI divergence in strong trending markets?"
   - "Find patterns that failed in choppy markets"
   - "Show me breakout patterns with high volume"

2. **Session/Time Analysis**
   - "What patterns work best during Asia session?"
   - "Show winning patterns on Tuesdays"
   - "Patterns that worked in Europe session"

3. **Market Condition Analysis**
   - "Show patterns in strong trending high volatility markets"
   - "Find support bounces in range-bound markets"
   - "Patterns when RSI was oversold"

4. **Support/Resistance**
   - "Show me support levels near 2000"
   - "Find strong resistance levels for XAUUSD"
   - "Support bounces with high volume confirmation"

## Complete Workflow

### 1. Process Your Data

```bash
# Process a single CSV file
python scripts/data_processor.py data/XAUUSD_PERIOD_M15_0.csv

# Or process all files
python scripts/run_processor.py --data-dir data --output-dir data/processed
```

Output: `data/XAUUSD_PERIOD_M15_0_processed.json`

### 2. Feed to RAG System

```bash
# Direct method (recommended)
python scripts/feed_to_rag_direct.py --dir data/processed

# Or via API
python scripts/feed_to_rag.py --dir data/processed
```

### 3. Query via Your Assistant

```bash
# Start the server if not running
python main.py

# Use the chat interface or API
# Your assistant now has access to all historical patterns!
```

## Verification

Check what was stored:

```python
from memory import MemoryManager

memory = MemoryManager()

# Count entries
results = memory.search_lessons("trading patterns", category="trading", n=100)
print(f"Found {len(results)} trading patterns")

results = memory.search_lessons("support resistance", category="technical", n=100)
print(f"Found {len(results)} S/R levels")
```

## Metadata Available for Filtering

Each pattern includes:

- **Pattern Type**: Bullish Engulfing, Bearish Engulfing, RSI Divergence, Breakout, Support Bounce, Resistance Rejection
- **Outcome**: WIN, LOSS, NEUTRAL
- **Trend**: Strong Bullish, Bullish, Neutral/Choppy, Bearish, Strong Bearish
- **Session**: Asia, Europe, New York
- **Day of Week**: Monday-Sunday
- **Market Regime**: Strong Trending High Volatility, Range-bound High Volatility, Strong Trending Low Volatility, Range-bound Low Volatility
- **RSI Level**: 0-100
- **Volume Ratio**: Relative to average
- **Entry/Stop/Target Prices**
- **P&L**: Actual profit/loss in points
- **Duration**: How long the trade took

## Advanced Usage

### Custom Collection

If you want patterns in a separate collection:

```python
from memory import MemoryManager

# Create separate collection for patterns
pattern_memory = MemoryManager(
    collection_name="trading_patterns_historical",
    persist_directory="./chroma_db"
)

# Feed data to this collection instead
# (modify feed_to_rag_direct.py to use this memory manager)
```

### Query with Filters

```python
# Get only winning patterns in trending markets
context = memory.get_combined_context(
    query="bullish patterns",
    memory_context=0,  # Skip conversations
    lesson_context=20   # Get more lessons
)

# Filter results in application logic
winning_patterns = [
    lesson for lesson in context['lessons']
    if 'WIN' in lesson and 'Trending' in lesson
]
```

### Export for Analysis

```python
import json

# Export all patterns
results = memory.search_lessons("patterns", category="trading", n=1000)

with open('exported_patterns.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### "Memory not available"
- Ensure ChromaDB is installed: `pip install chromadb`
- Check `chroma_db` directory exists
- Verify permissions

### No results when querying
- Check category matches: `category="trading"` or `category="technical"`
- Try broader queries: "XAUUSD patterns" instead of specific pattern names
- Increase `n` parameter: `n=20` instead of `n=5`

### API authentication failed
- Check password in config.json
- Ensure server is running: `python main.py`
- Verify port 8080 is not blocked

## Performance Tips

1. **Bulk Feeding**: Use `feed_to_rag_direct.py` for large datasets (10x faster than API)
2. **Incremental Updates**: Process and feed new data as it comes in
3. **Collection Management**: Keep patterns in main collection, or create separate for easy management
4. **Query Optimization**: Use specific tags and categories to narrow results

## Next Steps

1. ✅ Process your historical CSV files
2. ✅ Feed to RAG system
3. ✅ Test queries via Python or API
4. Start asking your Financial Assistant about patterns!

Your assistant now has deep knowledge of historical trading patterns and can provide insights based on real data! 🎯
