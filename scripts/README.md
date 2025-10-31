# Trading Data Processing Pipeline for RAG System

This directory contains tools for processing MT5/Sierra Chart CSV data into RAG-friendly narrative format.

## Overview

The system converts raw OHLCV data into comprehensive narrative documents that describe:
- Trading pattern setups with full context
- Support/Resistance levels with historical behavior
- Technical indicators and market conditions
- Trade outcomes and lessons learned

## Files

### `data_processor.py`
Core processing engine that:
- Loads MT5 CSV exports (handles UTF-16LE encoding)
- Calculates technical indicators (RSI, EMA, Bollinger Bands, ATR, MACD, Volume)
- Detects patterns (Bullish/Bearish Engulfing, RSI Divergence, Breakouts, S/R Bounces)
- Generates narrative documents with metadata
- Tracks outcomes (WIN/LOSS/NEUTRAL) by looking forward 20 bars

### `run_processor.py`
Batch processing runner that:
- Processes multiple CSV files automatically
- Outputs JSON files with patterns and levels
- Provides summary statistics

### `feed_to_rag.py`
RAG system integration that:
- Feeds processed patterns to the knowledge base via API
- Categorizes by pattern type and metadata
- Integrates with existing Financial Assistant system

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source /opt/py312/bin/activate

# Install required packages
pip install pandas numpy TA-Lib tqdm

# Install TA-Lib C library (macOS)
brew install ta-lib
```

### 2. Process Single File

The processor automatically detects timeframes from filenames:

```bash
# Process 15-minute data
python scripts/data_processor.py data/XAUUSD_PERIOD_M15_0.csv

# Process 1-hour data
python scripts/data_processor.py data/XAUUSD_PERIOD_H1_0.csv

# Process daily data
python scripts/data_processor.py data/XAUUSD_PERIOD_D1_0.csv

# Process weekly data
python scripts/data_processor.py data/XAUUSD_PERIOD_W1_0.csv
```

**Supported Timeframes:**
- M1 (1-minute), M5 (5-minute), M15 (15-minute), M30 (30-minute)
- H1 (1-hour), H4 (4-hour)
- D1 (Daily), W1 (Weekly)

Output: `data/XAUUSD_PERIOD_*_0_processed.json`

### 3. Process All Files

```bash
python scripts/run_processor.py --data-dir ./data --output-dir ./data/processed
```

### 4. Feed to RAG System

```bash
# Direct method (recommended for bulk - 10x faster)
python scripts/feed_to_rag_direct.py --dir ./data/processed

# Or via API
python scripts/feed_to_rag.py --dir ./data/processed --password admin123
```

## Data Format

### Input CSV Format (MT5 Export)
```csv
TimeFrame,Symbol,Candle,DateTime,Open,High,Low,Close,Volume,HL,Body
PERIOD_M15,XAUUSD,99998,2021.08.10 05:45,1733.71,1734.75,1733.09,1734.71,1013.0,166.0,100.0
```

### Output JSON Format
```json
{
  "patterns": [
    {
      "text": "Pattern: Bullish Engulfing\nSymbol: XAUUSD\n...",
      "metadata": {
        "pattern": "Bullish Engulfing",
        "timestamp": "2021-08-10T05:45:00",
        "entry": 1734.71,
        "stop": 1730.50,
        "target": 1742.92,
        "rsi": 45.2,
        "trend": "Bullish",
        "outcome": "WIN",
        "pnl": 8.21,
        "duration_bars": 5,
        "risk_reward": 2.0,
        "volume_ratio": 1.8,
        "session": "Asia",
        "day_of_week": "Tuesday",
        "market_regime": "Strong Trending Low Volatility"
      }
    }
  ],
  "levels": [
    {
      "text": "Support/Resistance Level: 1735.50\nType: Support...",
      "metadata": {
        "type": "support",
        "level": 1735.50,
        "date": "2021-08-10T06:00:00",
        "symbol": "XAUUSD",
        "timeframe": "15min",
        "strength": "moderate"
      }
    }
  ]
}
```

## Pattern Detection

The system detects the following patterns:

1. **Bullish Engulfing**: Bearish candle followed by larger bullish candle
2. **Bearish Engulfing**: Bullish candle followed by larger bearish candle
3. **RSI Bullish Divergence**: Price lower low, RSI higher low (RSI < 40)
4. **RSI Bearish Divergence**: Price higher high, RSI lower high (RSI > 60)
5. **Breakout**: Close above 20-bar high with volume confirmation (>1.5x avg)
6. **Support Bounce**: Price near swing low, bullish close
7. **Resistance Rejection**: Price near swing high, bearish close

## Technical Indicators

- **RSI(14)**: Momentum oscillator
- **EMA 9/20/50**: Trend indicators
- **Bollinger Bands(20)**: Volatility bands
- **ATR(14)**: Average True Range for volatility
- **MACD(12,26,9)**: Trend momentum
- **Volume Ratio**: Current volume / 20-period SMA

## Narrative Format

Each pattern generates a comprehensive narrative including:

### Setup Description
- Pattern formation details
- Price action description
- Candle characteristics

### Technical Context
- Trend analysis (EMA alignment)
- RSI state and momentum
- Bollinger Band position
- Volume analysis
- MACD state

### Trading Setup
- Entry, stop loss, take profit levels
- Risk/reward ratio
- Position sizing guidelines

### Outcome
- Trade result (WIN/LOSS/NEUTRAL)
- Profit/Loss in points
- Duration and timing
- Max Favorable/Adverse Excursion

### Lessons Learned
- Analysis of why pattern worked or failed
- Key success factors
- Conditions to avoid

### Similar Conditions
- Market regime classification
- Session and time context
- Historical win rate estimation

## Integration with Financial Assistant

The processed data integrates with your existing system:

```python
from scripts.feed_to_rag import TradingPatternFeeder

# Initialize feeder
feeder = TradingPatternFeeder(
    base_url="http://localhost:8080",
    password="admin123"
)

# Feed from processed file
result = feeder.feed_from_file("data/processed/XAUUSD_PERIOD_M15_0_processed.json")

print(f"Fed {result['patterns_success']} patterns")
print(f"Fed {result['levels_success']} levels")
```

## Performance

Processing ~100,000 bars takes approximately:
- Loading + Indicators: ~2-3 seconds
- Pattern Detection: ~6-8 minutes
- Support/Resistance: ~1-2 seconds
- Total: ~6-10 minutes per file

## Customization

### Add New Patterns

Edit `data_processor.py`:

```python
def _is_hammer(self, idx):
    """Detect hammer pattern"""
    curr = self.df.iloc[idx]
    body = abs(curr['close'] - curr['open'])
    lower_wick = min(curr['open'], curr['close']) - curr['low']
    upper_wick = curr['high'] - max(curr['open'], curr['close'])

    return (lower_wick > body * 2 and
            upper_wick < body and
            curr['close'] > curr['open'])
```

Then add to `detect_patterns()`:

```python
if self._is_hammer(i):
    doc = self._create_pattern_document(i, "Hammer")
    patterns.append(doc)
```

### Adjust Timeframes

The system automatically detects timeframe from the file. To change lookback periods:

- Pattern lookback: Change `start_idx` in `detect_patterns()` (default: 100)
- Outcome lookforward: Change `lookforward` in `_calculate_outcome()` (default: 20)
- S/R detection window: Change `lookback` in `_is_bounce()` (default: 100)

## Troubleshooting

### CSV Encoding Issues
If you get encoding errors, the processor automatically tries multiple encodings:
- UTF-16-LE (MT5 default)
- UTF-16
- UTF-8
- Latin1

### Missing Columns
Ensure your CSV has these columns (case-insensitive):
- DateTime/Timestamp/time/date
- Open
- High
- Low
- Close
- Volume/tick_volume/vol

### TA-Lib Installation

If TA-Lib fails to install:

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## Next Steps

1. Process your historical data
2. Feed to RAG system
3. Query patterns via Financial Assistant
4. Use insights for trading decisions

## Example Queries

Once data is in RAG system, you can ask:

- "Show me all bullish engulfing patterns during Asia session that won"
- "What's the win rate for RSI divergence in strong trending markets?"
- "Find support bounces near 2000 level that had high volume"
- "Show patterns that failed in choppy markets"
- "What are the best entry conditions for breakouts?"

---

Based on workflow_resume.md comprehensive implementation
