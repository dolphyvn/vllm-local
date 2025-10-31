#!/usr/bin/env python3
"""
Multi-Timeframe Analyzer
Processes all timeframes for a symbol to provide comprehensive market context
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob
from typing import Dict, List, Optional, Any
import re
from technical_analysis_engine import AdvancedTechnicalAnalyzer

def extract_symbol_info(filename: str) -> Optional[Dict[str, str]]:
    """Extract symbol and timeframe information from filename"""
    # Pattern: SYMBOL_PERIOD_TIMEFRAME_200.csv
    match = re.match(r'^([A-Z0-9]+)_PERIOD_([MH][0-9,]+[DW]?)_200\.csv$', filename)
    if match:
        symbol = match.group(1)
        timeframe = match.group(2)
        return {"symbol": symbol, "timeframe": timeframe, "filename": filename}
    return None

def convert_timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    if timeframe == 'D1':
        return 1440  # 1 day = 1440 minutes
    elif timeframe == 'W1':
        return 10080  # 1 week = 10080 minutes
    elif timeframe == 'MN1':
        return 43200  # 1 month = 43200 minutes
    elif timeframe.endswith('H'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('M'):
        return int(timeframe[:-1])
    else:
        return 1  # default to 1 minute

def read_and_validate_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Read and validate CSV file format"""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"❌ Could not read {filepath} with any encoding")
            return None

        # Standardize column names (case insensitive)
        df.columns = df.columns.str.lower()

        # Handle different column naming conventions
        column_mapping = {
            'date': 'datetime',
            'time': 'datetime',
            'timestamp': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        df = df.rename(columns=column_mapping)

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns {missing_cols} in {filepath}")
            return None

        # Convert datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        else:
            # If no datetime column, create sequential timestamps
            df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1min')

        # Clean data
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[df['high'] >= df['low']]  # Basic validation
        df = df[df['high'] >= df['open']]  # Basic validation
        df = df[df['high'] >= df['close']]  # Basic validation
        df = df[df['low'] <= df['open']]   # Basic validation
        df = df[df['low'] <= df['close']]  # Basic validation

        # Ensure volume column exists
        if 'volume' not in df.columns:
            df['volume'] = 1000  # Default volume

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def calculate_vwap(df: pd.DataFrame) -> np.ndarray:
    """Calculate Volume Weighted Average Price"""
    close_prices = df['close'].values
    volumes = df['volume'].values

    cumulative_volume = np.cumsum(volumes)
    cumulative_price_volume = np.cumsum(close_prices * volumes)
    vwap = cumulative_price_volume / cumulative_volume

    return vwap

def calculate_market_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Market Profile metrics"""
    close_prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    # Point of Control (POC) - price level with highest volume
    price_levels = {}
    for i, (high, low, close, volume) in enumerate(zip(highs, lows, close_prices, volumes)):
        # Distribute volume across the price range of the candle
        price_range = high - low
        if price_range > 0:
            volume_per_point = volume / (price_range * 4)  # Assuming tick size of 0.25
            for price in np.arange(low, high + 0.25, 0.25):
                price_rounded = round(price, 2)
                price_levels[price_rounded] = price_levels.get(price_rounded, 0) + volume_per_point
        else:
            price_rounded = round(close, 2)
            price_levels[price_rounded] = price_levels.get(price_rounded, 0) + volume

    if price_levels:
        poc_price = max(price_levels.keys(), key=lambda x: price_levels[x])

        # Calculate Value Area (70% of total volume)
        sorted_prices = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(price_levels.values())
        target_volume = total_volume * 0.7

        cumulative_volume = 0
        value_area_prices = []

        for price, volume in sorted_prices:
            value_area_prices.append(price)
            cumulative_volume += volume
            if cumulative_volume >= target_volume:
                break

        if value_area_prices:
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            value_area_width = value_area_high - value_area_low
        else:
            value_area_high = value_area_low = value_area_width = None
    else:
        poc_price = value_area_high = value_area_low = value_area_width = None

    # Range calculations
    current_range = abs(highs[-1] - lows[-1]) if len(highs) > 0 else 0
    avg_range = np.mean([abs(h - l) for h, l in zip(highs, lows)]) if len(highs) > 0 else 0

    return {
        "poc_price": float(poc_price) if poc_price else None,
        "value_area_high": float(value_area_high) if value_area_high else None,
        "value_area_low": float(value_area_low) if value_area_low else None,
        "value_area_width": float(value_area_width) if value_area_width else None,
        "current_price_in_value_area": (
            value_area_low <= close_prices[-1] <= value_area_high
            if value_area_low and value_area_high else None
        ),
        "avg_range": float(avg_range),
        "current_range": float(current_range),
        "range_expansion": current_range > (avg_range * 1.2) if avg_range > 0 else False
    }

def process_timeframe_data(filepath: str, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """Process a single timeframe CSV file"""
    print(f"🔄 Processing {timeframe} data for {symbol}...")

    df = read_and_validate_csv(filepath)
    if df is None or len(df) == 0:
        print(f"❌ No valid data in {filepath}")
        return None

    # Calculate indicators
    vwap = calculate_vwap(df)
    market_profile = calculate_market_profile(df)

    # Advanced analysis using technical analysis engine
    technical_analyzer = AdvancedTechnicalAnalyzer()

    # Calculate intraday session analysis
    intraday_sessions = technical_analyzer.calculate_intraday_market_profiles(df)

    # Fixed Range Volume Profiles
    fixed_range_profiles = {
        'session_range': technical_analyzer.calculate_fixed_range_volume_profile(df, 'session'),
        'visible_range': technical_analyzer.calculate_fixed_range_volume_profile(df, 'range'),
    }

    # Gap analysis
    gap_analysis = technical_analyzer.detect_gaps(df)

    # Order block analysis
    order_block_analysis = technical_analyzer.detect_order_blocks(df)

    # Fibonacci retracements
    fibonacci_analysis = technical_analyzer.calculate_fibonacci_retracements(df)

    # Ichimoku Cloud
    ichimoku_analysis = technical_analyzer.calculate_ichimoku_cloud(df)

    # Convert to candle objects
    candles = []
    for idx, row in df.tail(200).iterrows():  # Last 200 candles
        candle = {
            "datetime": row['datetime'].isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row['volume']),
            "price_range": float(row['high'] - row['low']),
            "body_size": float(abs(row['close'] - row['open'])),
            "candle_type": "bullish" if row['close'] > row['open'] else "bearish",
            "vwap": float(vwap[idx]) if idx < len(vwap) else None
        }
        candles.append(candle)

    # Technical patterns
    close_prices = df['close'].values
    volumes = df['volume'].values

    timeframe_data = {
        "timeframe": timeframe,
        "total_candles": len(df),
        "analyzed_candles": len(candles),
        "timeframe_minutes": convert_timeframe_to_minutes(timeframe),
        "time_coverage_hours": convert_timeframe_to_minutes(timeframe) * len(df) / 60,
        "price_analysis": {
            "current_price": float(close_prices[-1]),
            "price_range": {
                "high": float(df['high'].max()),
                "low": float(df['low'].min())
            },
            "average_price": float(close_prices.mean()),
            "price_volatility": float(close_prices.std()),
            "price_change": float(close_prices[-1] - close_prices[0]) if len(close_prices) > 1 else 0,
            "price_change_percent": float((close_prices[-1] - close_prices[0]) / close_prices[0] * 100) if len(close_prices) > 1 and close_prices[0] != 0 else 0
        },
        "volume_analysis": {
            "avg_volume": float(volumes.mean()),
            "current_volume": float(volumes[-1]),
            "total_volume": float(volumes.sum()),
            "volume_ratio": float(volumes[-1] / volumes.mean()) if volumes.mean() > 0 else 0,
            "volume_trend": "increasing" if len(volumes) > 5 and np.mean(volumes[-3:]) > np.mean(volumes[-6:-3]) else "decreasing"
        },
        "technical_patterns": {
            "trend_direction": "uptrend" if close_prices[-1] > close_prices[-min(10, len(close_prices)-1)] else "downtrend",
            "volatility_trend": "increasing" if df['close'].std() > df['close'].head(min(10, len(df)-1)).std() else "stable",
            "price_momentum": float(close_prices[-1] - close_prices[-min(5, len(close_prices)-1)]) if len(close_prices) >= 5 else 0,
            "vwap_analysis": {
                "current_vwap": float(vwap[-1]) if len(vwap) > 0 else None,
                "vwap_deviation_percent": float(abs(close_prices[-1] - vwap[-1]) / vwap[-1] * 100) if len(vwap) > 0 and vwap[-1] != 0 else None,
                "price_above_vwap": close_prices[-1] > vwap[-1] if len(vwap) > 0 else None,
                "vwap_trend": "rising" if len(vwap) > 1 and vwap[-1] > vwap[-min(5, len(vwap)-1)] else "falling"
            },
            "market_profile": market_profile,
            "intraday_sessions": intraday_sessions,
            "fixed_range_profiles": fixed_range_profiles,
            "gap_analysis": gap_analysis,
            "order_block_analysis": order_block_analysis,
            "fibonacci_analysis": fibonacci_analysis,
            "ichimoku_analysis": ichimoku_analysis,
        },
        "all_candles": candles
    }

    print(f"✅ {timeframe}: {len(candles)} candles, price range {timeframe_data['price_analysis']['price_range']['low']:.2f}-{timeframe_data['price_analysis']['price_range']['high']:.2f}")
    return timeframe_data

def create_comprehensive_symbol_analysis(symbol: str, timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive analysis combining all timeframes"""

    # Sort timeframes by coverage (shortest to longest)
    sorted_timeframes = sorted(
        timeframe_data.items(),
        key=lambda x: x[1]['time_coverage_hours']
    )

    # Calculate overall market context
    overall_trend = "mixed"
    trend_votes = {"uptrend": 0, "downtrend": 0, "mixed": 0}

    for timeframe, data in timeframe_data.items():
        trend = data['technical_patterns']['trend_direction']
        trend_votes[trend] = trend_votes.get(trend, 0) + 1

    if trend_votes["uptrend"] > trend_votes["downtrend"]:
        overall_trend = "uptrend"
    elif trend_votes["downtrend"] > trend_votes["uptrend"]:
        overall_trend = "downtrend"

    # Calculate consensus levels
    all_highs = []
    all_lows = []
    vwap_levels = []
    poc_levels = []

    for timeframe, data in timeframe_data.items():
        price_range = data['price_analysis']['price_range']
        all_highs.append(price_range['high'])
        all_lows.append(price_range['low'])

        vwap = data['technical_patterns']['vwap_analysis']['current_vwap']
        if vwap:
            vwap_levels.append(vwap)

        poc = data['technical_patterns']['market_profile']['poc_price']
        if poc:
            poc_levels.append(poc)

    consensus_data = {
        "highest_price": max(all_highs) if all_highs else None,
        "lowest_price": min(all_lows) if all_lows else None,
        "avg_vwap": np.mean(vwap_levels) if vwap_levels else None,
        "avg_poc": np.mean(poc_levels) if poc_levels else None,
        "vwap_confluence": len(set([round(v, 2) for v in vwap_levels if v])) < len(vwap_levels) / 2 if vwap_levels else False,
        "poc_confluence": len(set([round(p, 2) for p in poc_levels if p])) < len(poc_levels) / 2 if poc_levels else False,
        "total_time_coverage_days": sum(data['time_coverage_hours'] for data in timeframe_data.values()) / 24
    }

    # Load custom prompt if available
    custom_prompt = None
    prompt_file = "./prompts/trading_prompt.txt"  # Default custom prompt file
    if os.path.exists(prompt_file):
        custom_prompt = load_custom_prompt(prompt_file)
    elif os.path.exists(args.prompt_file):
        custom_prompt = load_custom_prompt(args.prompt_file)
        prompt_file = args.prompt_file

    # Create comprehensive prompt
    detailed_prompt = create_multi_timeframe_prompt(symbol, timeframe_data, consensus_data, custom_prompt)

    return {
        "symbol": symbol,
        "analysis_timestamp": datetime.now().isoformat(),
        "overall_market_context": {
            "overall_trend": overall_trend,
            "trend_votes": trend_votes,
            "consensus_levels": consensus_data,
            "total_timeframes_analyzed": len(timeframe_data),
            "total_time_coverage_hours": sum(data['time_coverage_hours'] for data in timeframe_data.values()),
            "time_coverage_days": sum(data['time_coverage_hours'] for data in timeframe_data.values()) / 24
        },
        "timeframe_analysis": timeframe_data
    }

def load_custom_prompt(prompt_file: str) -> str:
    """Load custom prompt from file"""
    try:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read().strip()
            print(f"✅ Loaded custom prompt from: {prompt_file}")
            return custom_prompt
        else:
            print(f"⚠️ Custom prompt file not found: {prompt_file}")
            return None
    except Exception as e:
        print(f"❌ Error loading custom prompt: {e}")
        return None

def create_multi_timeframe_prompt(symbol: str, timeframe_data: Dict[str, Any], consensus_data: Dict[str, Any], custom_prompt: str = None) -> str:
    """Create comprehensive trading analysis prompt using all timeframes"""

    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        return f"""
You are {custom_prompt}

SYMBOL: {symbol}
TIMEFRAMES ANALYZED: {len(timeframe_data)}
TOTAL COVERAGE: {consensus_data['total_time_coverage_days']:.1f} days

CURRENT MARKET DATA:
{json.dumps(timeframe_data, indent=2, default=str)}

Please provide your trading analysis based on the comprehensive multi-timeframe data above.
"""

    # Simple trading-focused prompt (default)
    base_prompt = """You are an expert trader, technical analysis, data analysis.

Please look at the data and tell me best trade direction, entry, stoploss and take profit.

SYMBOL: {symbol}
TIMEFRAMES ANALYZED: {len(timeframe_data)}
TOTAL COVERAGE: {consensus_data['total_time_coverage_days']:.1f} days

CURRENT PRICE: ${timeframe_data['M1']['price_analysis']['current_price']:.2f}
CONSENSUS LEVELS:
- Highest: ${consensus_data['highest_price']:.2f}
- Lowest: ${consensus_data['lowest_price']:.2f}
- Average VWAP: ${consensus_data['avg_vwap']:.2f}
- Current in Value Area: {'YES' if any(tf_data['technical_patterns']['market_profile'].get('current_price_in_value_area') for tf_data in timeframe_data.values()) else 'NO'}

"""

    prompt = f"""
You are an expert institutional trader with deep expertise in multi-timeframe analysis, market structure analysis, and comprehensive risk management.

COMPREHENSIVE MULTI-TIMEFRAME ANALYSIS: {symbol}
==============================================

**Market Overview:**
- Symbol: {symbol}
- Overall Trend: {timeframe_data[list(timeframe_data.keys())[0]]['technical_patterns']['trend_direction'].upper()}
- Total Time Coverage: {consensus_data['total_time_coverage_days']:.1f} days
- Timeframes Analyzed: {len(timeframe_data)} (M1, M5, M15, M30, H1, H4, D1)
- Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

**CONSENSUS LEVELS (Across All Timeframes):**
- Highest Price: ${consensus_data['highest_price']:.2f}
- Lowest Price: ${consensus_data['lowest_price']:.2f}
- Average VWAP: ${consensus_data['avg_vwap']:.2f} {'(CONFLUENCE)' if consensus_data['vwap_confluence'] else ''}
- Average POC: ${consensus_data['avg_poc']:.2f} {'(CONFLUENCE)' if consensus_data['poc_confluence'] else ''}
- Current Price: ${timeframe_data['M1']['price_analysis']['current_price']:.2f}

"""

    # Add detailed analysis for each timeframe
    timeframes_in_order = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

    for timeframe in timeframes_in_order:
        if timeframe in timeframe_data:
            data = timeframe_data[timeframe]

            # Extract values safely for formatting
            poc_price = data['technical_patterns']['market_profile']['poc_price']
            value_area_low = data['technical_patterns']['market_profile']['value_area_low']
            value_area_high = data['technical_patterns']['market_profile']['value_area_high']

            poc_str = f"{poc_price:.2f}" if poc_price is not None else 'N/A'
            value_area_low_str = f"{value_area_low:.2f}" if value_area_low is not None else 'N/A'
            value_area_high_str = f"{value_area_high:.2f}" if value_area_high is not None else 'N/A'

            prompt += f"""
**{timeframe} TIMEFRAME ANALYSIS:**
- Coverage: {data['time_coverage_hours']:.1f} hours ({data['analyzed_candles']} candles)
- Current Price: ${data['price_analysis']['current_price']:.2f}
- Price Range: ${data['price_analysis']['price_range']['low']:.2f} - ${data['price_analysis']['price_range']['high']:.2f}
- Price Change: {data['price_analysis']['price_change']:+.2f} ({data['price_analysis']['price_change_percent']:+.2f}%)
- Trend: {data['technical_patterns']['trend_direction']}
- Volume: {data['volume_analysis']['current_volume']:.0f} (avg: {data['volume_analysis']['avg_volume']:.0f}, ratio: {data['volume_analysis']['volume_ratio']:.2f}x)
- VWAP: ${data['technical_patterns']['vwap_analysis']['current_vwap']:.2f} ({'ABOVE' if data['technical_patterns']['vwap_analysis']['price_above_vwap'] else 'BELOW'})
- POC: ${poc_str}
- Value Area: ${value_area_low_str} - ${value_area_high_str}
- Current in Value Area: {'YES' if data['technical_patterns']['market_profile']['current_price_in_value_area'] else 'NO'}
"""

    # Add recent candle data for context
    prompt += f"""
**RECENT PRICE ACTION (Last 20 candles across timeframes):**
"""

    # Add M1 candles for detailed price action
    if 'M1' in timeframe_data:
        m1_candles = timeframe_data['M1']['all_candles'][-20:]
        for i, candle in enumerate(m1_candles):
            prompt += f"M1 Candle {i+1}: {candle['datetime']} - OHLC: {candle['open']:.2f}/{candle['high']:.2f}/{candle['low']:.2f}/{candle['close']:.2f} - Volume: {candle['volume']:.0f}\n"

    prompt += f"""
MULTI-TIMEFRAME ANALYSIS REQUEST:
==============================
Based on the comprehensive multi-timeframe data above, provide detailed trading analysis:

1. **OVERALL MARKET STRUCTURE**:
   - Primary trend direction (considering all timeframes)
   - Key support and resistance levels (from consensus data)
   - Market phase (accumulation, distribution, trend, range)
   - Timeframe synchronization (do all timeframes agree?)

2. **OPTIMAL ENTRY STRATEGY**:
   - Recommended trade direction (BUY/SELL/WAIT) with multi-timeframe confirmation
   - Optimal entry price level with timeframe justification
   - Entry timing (which timeframe gives the best entry signal?)
   - Risk-reward ratio with specific targets

3. **CRITICAL PRICE LEVELS**:
   - Key support levels (from all timeframes)
   - Key resistance levels (from all timeframes)
   - VWAP confluence zones
   - POC/Value Area levels across timeframes
   - Breakout/breakdown levels

4. **STOP LOSS STRATEGY**:
   - Strategic stop-loss placement with timeframe reasoning
   - Stop-loss distance in pips and dollars
   - Timeframe-specific stop-loss considerations
   - Volatility-adjusted stop-loss

5. **TAKE PROFIT TARGETS**:
   - Primary and secondary profit targets
   - Multi-timeframe profit level confluence
   - Projected price movement based on historical patterns
   - Scaling out strategy if applicable

6. **RISK MANAGEMENT**:
   - Position sizing recommendations
   - Risk per trade percentage
   - Maximum drawdown considerations
   - Multi-timeframe risk assessment

7. **MARKET CONTEXT ANALYSIS**:
   - How does current price relate to key consensus levels?
   - VWAP alignment across timeframes
   - Volume analysis confirmation
   - Market Profile acceptance/rejection patterns
   - Timeframe divergences or confirmations

**CRITICAL CONSIDERATIONS:**
- Which timeframe is leading the current move?
- Are there any timeframe divergences that signal potential reversals?
- How does the current price action align with long-term vs short-term structure?
- Volume confirmation across timeframes
- Key psychological price levels
- Recent market volatility and its impact on position sizing

Provide specific, actionable trading recommendations with exact price levels, considering the full {consensus_data['total_time_coverage_days']:.1f} days of market context available across all timeframes.
"""

    return prompt

def process_symbol_all_timeframes(data_dir: str, symbol: str, output_dir: str = "./data/rag_processed") -> bool:
    """Process all available timeframes for a symbol"""

    print(f"\n🔄 Processing comprehensive multi-timeframe analysis for {symbol}")
    print("=" * 60)

    # Find all CSV files for the symbol
    pattern = os.path.join(data_dir, f"{symbol}_PERIOD_*_200.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"❌ No CSV files found for symbol {symbol}")
        return False

    print(f"📁 Found {len(csv_files)} timeframe files for {symbol}")

    # Process each timeframe
    timeframe_data = {}

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        file_info = extract_symbol_info(filename)

        if file_info and file_info['symbol'] == symbol:
            timeframe = file_info['timeframe']

            data = process_timeframe_data(csv_file, symbol, timeframe)
            if data:
                timeframe_data[timeframe] = data

    if not timeframe_data:
        print(f"❌ No valid timeframe data processed for {symbol}")
        return False

    print(f"\n📊 Successfully processed {len(timeframe_data)} timeframes for {symbol}")

    # Create comprehensive analysis
    comprehensive_analysis = create_comprehensive_symbol_analysis(symbol, timeframe_data)

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}.json")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)

        file_size = os.path.getsize(output_file)
        print(f"\n✅ Multi-timeframe analysis saved to: {output_file}")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Timeframes: {len(timeframe_data)}")
        print(f"   Total coverage: {comprehensive_analysis['overall_market_context']['time_coverage_days']:.1f} days")
        print(f"   Symbol: {symbol}")

        return True

    except Exception as e:
        print(f"❌ Error saving analysis: {e}")
        return False

def save_rag_to_json(comprehensive_analysis, symbol, output_dir="./data/rag_processed", add_timestamp=False):
    """
    Save RAG data to JSON file with optional timestamp
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timestamp}.json"
        else:
            filename = f"{symbol}.json"

        output_file = os.path.join(output_dir, filename)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)

        file_size = os.path.getsize(output_file)

        print(f"✅ Multi-timeframe RAG data saved to: {output_file}")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Symbol: {symbol}")
        print(f"   Timeframes: {len(comprehensive_analysis.get('timeframe_analysis', {}))}")
        print(f"   Coverage: {comprehensive_analysis.get('overall_market_context', {}).get('time_coverage_days', 0):.1f} days")

        return output_file

    except Exception as e:
        print(f"❌ Error saving RAG data: {e}")
        return None

def process_with_enhanced_api(symbol, comprehensive_analysis, api_base="http://localhost:8080", model="qwen3:14b"):
    """
    Process the multi-timeframe analysis with the API for detailed analysis
    """
    print(f"\n🔄 Processing multi-timeframe analysis with API using model: {model}")
    print("=" * 60)

    try:
        import requests

        # First, login to get session token
        print("🔐 Authenticating with API...")
        auth_response = requests.post(f"{api_base}/auth/login",
                                    json={"password": "admin123"},
                                    timeout=30,
                                    headers={'Connection': 'close'})

        if auth_response.status_code != 200:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            return False

        session_token = auth_response.json().get('session_token')
        if not session_token:
            print("❌ No session token received")
            return False

        print("✅ Authentication successful")

        # Add comprehensive analysis to knowledge base
        print("📚 Adding comprehensive multi-timeframe analysis to knowledge base...")

        # Convert analysis to JSON string for content
        analysis_content = json.dumps(comprehensive_analysis, indent=2, default=str)

        knowledge_data = {
            "topic": f"Multi-Timeframe Analysis - {symbol} - All Timeframes",
            "content": f"""
Comprehensive Multi-Timeframe Trading Analysis for {symbol}
==============================================

Symbol: {symbol}
Analysis Timestamp: {comprehensive_analysis.get('analysis_timestamp', 'Unknown')}
Total Timeframes: {len(comprehensive_analysis.get('timeframe_analysis', {}))}
Total Coverage: {comprehensive_analysis.get('overall_market_context', {}).get('time_coverage_days', 0):.1f} days
Overall Trend: {comprehensive_analysis.get('overall_market_context', {}).get('overall_trend', 'Unknown')}

Market Context:
{json.dumps(comprehensive_analysis.get('overall_market_context', {}), indent=2, default=str)}

Complete Analysis Data:
{analysis_content}

This comprehensive analysis provides {comprehensive_analysis.get('overall_market_context', {}).get('time_coverage_days', 0):.1f} days of market context across multiple timeframes for {symbol}.
            """.strip(),
            "metadata": {
                "source": f"{symbol}_multi_timeframe",
                "symbol": symbol,
                "analysis_type": "multi_timeframe",
                "model_used": model,
                "timeframes_count": len(comprehensive_analysis.get('timeframe_analysis', {})),
                "coverage_days": comprehensive_analysis.get('overall_market_context', {}).get('time_coverage_days', 0),
                "user": "multi_timeframe_analyzer",
                "timestamp": datetime.now().isoformat()
            }
        }

        # Send to knowledge base
        knowledge_response = requests.post(f"{api_base}/api/knowledge/add",
                                          json=knowledge_data,
                                          timeout=60,
                                          headers={
                                              'Authorization': f'Bearer {session_token}',
                                              'Content-Type': 'application/json',
                                              'Connection': 'close'
                                          })

        if knowledge_response.status_code == 200:
            print("✅ Multi-timeframe analysis added to knowledge base")
        else:
            print(f"⚠️ Knowledge addition failed: {knowledge_response.status_code}")
            if knowledge_response.status_code != 200:
                print(f"Response: {knowledge_response.text[:200]}...")  # Show first 200 chars

        # Get detailed trade recommendation
        print(f"💡 Getting detailed trade recommendation using {model}...")

        chat_data = {
            "message": comprehensive_analysis.get('detailed_prompt', f'Please provide comprehensive trading analysis for {symbol} based on the multi-timeframe data'),
            "model": model,
            "use_rag": True,
            "context_sources": [f"Multi-Timeframe Analysis - {symbol}"]
        }

        # Use the chat endpoint and actually get the response
        chat_response = requests.post(f"{api_base}/chat/stream",
                                      json=chat_data,
                                      timeout=180,
                                      headers={
                                          'Authorization': f'Bearer {session_token}',
                                          'Content-Type': 'application/json',
                                          'Connection': 'close'
                                      })

        if chat_response.status_code == 200:
            print("✅ Chat request sent successfully")
            print(f"💡 Analyzing {symbol} with {model}...")

            # Process the streaming response
            response_text = ""
            for line in chat_response.iter_lines():
                # Handle both string and bytes responses
                if isinstance(line, bytes):
                    line = line.decode('utf-8')  # Decode bytes to string

                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                response_text += delta['content']
                    except json.JSONDecodeError:
                        continue
                elif line.startswith('data:'):  # Handle different prefix formats
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')  # Decode bytes to string
                    data = line[5:]  # Remove 'data:' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                response_text += delta['content']
                    except json.JSONDecodeError:
                        continue

            if response_text.strip():
                print(f"\n🎯 Trading Analysis for {symbol}:")
                print("=" * 50)
                print(response_text)
                print("=" * 50)
                print(f"\n✅ Analysis completed using {model}")
                return True
            else:
                print(f"⚠️ No response received from {model}")
                return False
        else:
            print(f"❌ Chat request failed: {chat_response.status_code}")
            if chat_response.status_code != 200:
                print(f"Response: {chat_response.text}")
            return False

    except Exception as e:
        print(f"❌ API processing error: {e}")
        return False

def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Timeframe Analyzer')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to analyze (e.g., XAUUSD)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='./data/rag_processed', help='Output directory path')
    parser.add_argument('--save-rag', action='store_true', default=True, help='Save RAG data to JSON file (default: True)')
    parser.add_argument('--add-timestamp', action='store_true', help='Add timestamp to filename (default: False)')
    parser.add_argument('--model', type=str, default='qwen3:14b',
                       choices=['gemma3:1b', 'gemma2:2b', 'qwen3:0.6b', 'qwen3:14b'],
                       help='Model to use for analysis (default: qwen3:14b)')
    parser.add_argument('--api-base', type=str, default='http://localhost:8080', help='API base URL (default: http://localhost:8080)')
    parser.add_argument('--send-to-api', action='store_true', help='Send analysis to API (default: False)')
    parser.add_argument('--prompt-file', type=str, default='./prompts/trading_prompt.txt', help='Custom prompt file path (default: ./prompts/trading_prompt.txt)')

    args = parser.parse_args()

    print(f"📊 Configuration:")
    print(f"   Symbol: {args.symbol}")
    print(f"   Data Directory: {args.data_dir}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Save RAG: {args.save_rag}")
    print(f"   Add Timestamp: {args.add_timestamp}")
    print(f"   Model: {args.model}")
    print(f"   API Base: {args.api_base}")
    print(f"   Send to API: {args.send_to_api}")
    print(f"   Prompt File: {args.prompt_file}")
    print()

    success = process_symbol_all_timeframes(args.data_dir, args.symbol, args.output_dir)

    if not success:
        print(f"\n❌ Failed to complete analysis for {args.symbol}")
        return 1

    # Create comprehensive analysis for saving
    print(f"\n🔄 Creating comprehensive analysis for {args.symbol}...")

    # Load the saved analysis
    analysis_file = os.path.join(args.output_dir, f"{args.symbol}.json")
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            comprehensive_analysis = json.load(f)

        # Save with enhanced options
        if args.save_rag:
            saved_file = save_rag_to_json(comprehensive_analysis, args.symbol, args.output_dir, args.add_timestamp)
            if saved_file:
                print(f"📁 Analysis saved to: {saved_file}")

                # Send to API if requested
                if args.send_to_api:
                    api_success = process_with_enhanced_api(args.symbol, comprehensive_analysis, args.api_base, args.model)
                    if api_success:
                        print(f"🎉 Multi-timeframe analysis and API integration completed successfully for {args.symbol}!")
                        print(f"📝 You can now ask the chat: 'Provide trading analysis for {args.symbol}'")
                    else:
                        print(f"⚠️ Analysis saved but API integration failed for {args.symbol}")
                else:
                    print(f"🎉 Multi-timeframe analysis completed successfully for {args.symbol}!")
                    print(f"📁 Analysis saved to: {saved_file}")
                    print(f"📝 You can now ask the chat: 'Provide trading analysis for {args.symbol}'")
            else:
                print(f"❌ Failed to save analysis for {args.symbol}")
                return 1
        else:
            print(f"🎉 Multi-timeframe analysis completed successfully for {args.symbol}!")
            print(f"📁 Analysis available in memory (not saved due to --save-rag flag)")

    except Exception as e:
        print(f"❌ Error loading analysis for saving: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())