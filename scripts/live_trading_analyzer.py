#!/usr/bin/env python3
"""
Live Trading Analyzer - Phase 3 Implementation
Analyzes latest market data and generates trade recommendations

Features:
- Multi-encoding CSV support
- Technical indicators calculation (matching mt5_to_structured_json.py)
- Current pattern detection
- Advanced trade recommendation generation (using trade_recommendation_engine.py)
- ChromaDB storage integration (using chroma_live_analyzer.py)
- Analysis summary generation
- JSON output for integration

Usage:
    python scripts/live_trading_analyzer.py \
        --input data/XAUUSD_M15_200.csv \
        --symbol XAUUSD --timeframe M15 \
        --output data/live_analysis/XAUUSD_M15_analysis.json \
        --add-to-rag
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class LiveTradingAnalyzer:
    """Analyze live market data and generate trading recommendations"""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M15"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df: Optional[pd.DataFrame] = None
        self.analysis_result: Dict[str, Any] = {}

        # Initialize trade recommendation engine (Phase 2)
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from trade_recommendation_engine import TradeRecommendationEngine, PatternInfo
            self.trade_engine = TradeRecommendationEngine()
            self.PatternInfo = PatternInfo
        except ImportError as e:
            print(f"⚠️  Trade recommendation engine not available - using basic recommendations ({e})")
            self.trade_engine = None
            self.PatternInfo = None

        # Initialize ChromaDB analyzer (Phase 3)
        self.chroma_analyzer = None

    def load_csv(self, csv_path: str) -> bool:
        """
        Load CSV with multi-encoding support

        Args:
            csv_path: Path to CSV file

        Returns:
            True if successful, False otherwise
        """
        print(f"📂 Loading CSV: {csv_path}")

        # Try multiple encodings with different delimiters
        encodings = ['utf-16-le', 'utf-16', 'utf-8', 'latin-1', 'cp1252']
        delimiters = [',', '\t', ';']

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    print(f"   Trying {encoding} with delimiter '{delimiter}'...", end=" ")
                    df = pd.read_csv(csv_path, encoding=encoding, delimiter=delimiter)

                    # Check if we got multiple columns
                    if len(df.columns) > 3:
                        print(f"✅ Success! Found {len(df.columns)} columns")
                        self.df = df
                        return True
                    else:
                        print(f"⏭️  Only {len(df.columns)} column(s)")

                except Exception as e:
                    print(f"❌")
                    continue

        print(f"❌ Could not read CSV with any encoding/delimiter combination")
        return False

    def normalize_columns(self) -> bool:
        """Normalize column names to standard format"""
        print(f"🔧 Normalizing columns...")

        if self.df is None:
            return False

        print(f"   Available columns: {list(self.df.columns)}")

        # Column mapping for different formats
        column_map = {
            'DateTime': 'timestamp',
            'Timestamp': 'timestamp',
            'Date': 'timestamp',
            'Time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'TickVolume': 'volume',
            'tick_volume': 'volume',
            'Candle': 'candle_index'
        }

        # Apply mapping
        self.df = self.df.rename(columns=column_map)

        # Ensure required columns exist
        required = ['timestamp', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in self.df.columns]

        if missing:
            print(f"❌ Missing required columns: {missing}")
            print(f"   Available: {list(self.df.columns)}")
            return False

        # Add volume if missing
        if 'volume' not in self.df.columns:
            print(f"⚠️  No volume column, using default value 1000")
            self.df['volume'] = 1000

        print(f"✅ Columns normalized: {list(self.df.columns)}")
        return True

    def parse_timestamps(self) -> bool:
        """Parse and validate timestamps"""
        print(f"🕐 Parsing timestamps...")

        if self.df is None:
            return False

        try:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

            # Sort by timestamp
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)

            # Check for missing timestamps
            if self.df['timestamp'].isna().any():
                na_count = self.df['timestamp'].isna().sum()
                print(f"⚠️  {na_count} invalid timestamps, creating synthetic ones...")

                # Create synthetic timestamps for missing values
                start_date = self.df['timestamp'].dropna().iloc[0] if len(self.df['timestamp'].dropna()) > 0 else datetime.now()
                synthetic_timestamps = pd.date_range(start=start_date, periods=len(self.df), freq=self._get_frequency())
                self.df.loc[self.df['timestamp'].isna(), 'timestamp'] = synthetic_timestamps[self.df['timestamp'].isna()]

            print(f"✅ Timestamps parsed: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            return True

        except Exception as e:
            print(f"❌ Timestamp parsing failed: {e}")
            return False

    def _get_frequency(self) -> str:
        """Get pandas frequency string from timeframe"""
        freq_map = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W', 'MN1': '1M'
        }
        return freq_map.get(self.timeframe, '15min')

    def calculate_indicators(self) -> bool:
        """Calculate technical indicators matching mt5_to_structured_json.py"""
        print(f"📊 Calculating indicators...")

        if self.df is None:
            return False

        try:
            close = self.df['close'].values
            high = self.df['high'].values
            low = self.df['low'].values
            volume = self.df['volume'].values

            # === MOMENTUM INDICATORS ===

            # RSI (14)
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            self.df['rsi'] = 100 - (100 / (1 + rs))
            self.df['rsi'] = self.df['rsi'].fillna(50)

            # MACD (12, 26, 9)
            exp12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            exp26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            self.df['macd'] = exp12 - exp26
            self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
            self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

            # Stochastic Oscillator (14, 3, 3)
            low_14 = pd.Series(low).rolling(window=14, min_periods=1).min()
            high_14 = pd.Series(high).rolling(window=14, min_periods=1).max()
            self.df['stoch_k'] = 100 * ((pd.Series(close) - low_14) / (high_14 - low_14).replace(0, 1))
            self.df['stoch_d'] = self.df['stoch_k'].rolling(window=3, min_periods=1).mean()

            # === TREND INDICATORS ===

            # EMAs
            self.df['ema_9'] = pd.Series(close).ewm(span=9, adjust=False).mean()
            self.df['ema_20'] = pd.Series(close).ewm(span=20, adjust=False).mean()
            self.df['ema_50'] = pd.Series(close).ewm(span=50, adjust=False).mean()
            self.df['ema_200'] = pd.Series(close).ewm(span=200, adjust=False).mean()

            # SMAs
            self.df['sma_20'] = pd.Series(close).rolling(window=20, min_periods=1).mean()
            self.df['sma_50'] = pd.Series(close).rolling(window=50, min_periods=1).mean()

            # === VOLATILITY INDICATORS ===

            # Bollinger Bands (20, 2)
            self.df['bb_middle'] = pd.Series(close).rolling(window=20, min_periods=1).mean()
            bb_std = pd.Series(close).rolling(window=20, min_periods=1).std()
            self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
            self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
            self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']

            # ATR (14)
            high_low = pd.Series(high) - pd.Series(low)
            high_close = np.abs(pd.Series(high) - pd.Series(close).shift())
            low_close = np.abs(pd.Series(low) - pd.Series(close).shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
            self.df['atr'] = self.df['atr'].fillna(self.df['atr'].mean())

            # === VOLUME INDICATORS ===

            # Volume SMA and ratio
            self.df['volume_sma'] = pd.Series(volume).rolling(window=20, min_periods=1).mean()
            self.df['volume_ratio'] = pd.Series(volume) / self.df['volume_sma'].replace(0, 1)

            # VWAP (Volume Weighted Average Price)
            typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
            self.df['vwap'] = (typical_price * pd.Series(volume)).cumsum() / pd.Series(volume).cumsum()

            # === PRICE ACTION ===

            # Price changes
            self.df['price_change_pct'] = pd.Series(close).pct_change() * 100
            self.df['price_change_abs'] = pd.Series(close).diff()

            # Candle body and wicks
            self.df['body_size'] = np.abs(self.df['close'] - self.df['open'])
            self.df['upper_wick'] = pd.Series(high) - pd.DataFrame({'open': self.df['open'], 'close': self.df['close']}).max(axis=1)
            self.df['lower_wick'] = pd.DataFrame({'open': self.df['open'], 'close': self.df['close']}).min(axis=1) - pd.Series(low)
            self.df['hl_range'] = pd.Series(high) - pd.Series(low)

            # Fill NaN values
            self.df = self.df.ffill().bfill()

            print(f"✅ Calculated {len([c for c in self.df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
            return True

        except Exception as e:
            print(f"❌ Indicator calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect_current_pattern(self) -> Dict[str, Any]:
        """Detect current trading pattern"""
        print(f"🔍 Detecting current pattern...")

        if self.df is None or len(self.df) < 10:
            return {"pattern": "insufficient_data", "confidence": 0}

        # Get the last few candles for pattern detection
        recent = self.df.tail(10)
        last = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        pattern_info = {
            "name": "unknown",
            "type": "unknown",
            "direction": "unknown",
            "quality": 0.0,
            "confidence": 0.0
        }

        # Detect bullish engulfing
        if (prev['close'] < prev['open'] and  # Previous candle is bearish
            last['close'] > last['open'] and   # Current candle is bullish
            last['open'] < prev['close'] and   # Opens below previous close
            last['close'] > prev['open']):     # Closes above previous open

            pattern_info = {
                "name": "bullish_engulfing",
                "type": "reversal",
                "direction": "bullish",
                "quality": 0.85,
                "confidence": 85
            }

        # Detect bearish engulfing
        elif (prev['close'] > prev['open'] and  # Previous candle is bullish
              last['close'] < last['open'] and   # Current candle is bearish
              last['open'] > prev['close'] and   # Opens above previous close
              last['close'] < prev['open']):     # Closes below previous open

            pattern_info = {
                "name": "bearish_engulfing",
                "type": "reversal",
                "direction": "bearish",
                "quality": 0.85,
                "confidence": 85
            }

        # Detect hammer (potential bullish reversal)
        elif (last['lower_wick'] > 2 * last['body_size'] and  # Long lower wick
              last['upper_wick'] < 0.1 * last['body_size'] and  # Small upper wick
              last['close'] > prev['close']):                 # Higher close

            pattern_info = {
                "name": "hammer",
                "type": "reversal",
                "direction": "bullish",
                "quality": 0.75,
                "confidence": 75
            }

        # Detect shooting star (potential bearish reversal)
        elif (last['upper_wick'] > 2 * last['body_size'] and   # Long upper wick
              last['lower_wick'] < 0.1 * last['body_size'] and  # Small lower wick
              last['close'] < prev['close']):                 # Lower close

            pattern_info = {
                "name": "shooting_star",
                "type": "reversal",
                "direction": "bearish",
                "quality": 0.75,
                "confidence": 75
            }

        # Detect trend based on moving averages
        elif (last['ema_9'] > last['ema_20'] > last['ema_50']):
            pattern_info = {
                "name": "uptrend_continuation",
                "type": "continuation",
                "direction": "bullish",
                "quality": 0.70,
                "confidence": 70
            }

        elif (last['ema_9'] < last['ema_20'] < last['ema_50']):
            pattern_info = {
                "name": "downtrend_continuation",
                "type": "continuation",
                "direction": "bearish",
                "quality": 0.70,
                "confidence": 70
            }

        # Detect consolidation
        elif abs(last['ema_9'] - last['ema_20']) < 0.01 * last['close']:
            pattern_info = {
                "name": "consolidation",
                "type": "continuation",
                "direction": "neutral",
                "quality": 0.60,
                "confidence": 60
            }

        print(f"✅ Pattern detected: {pattern_info['name']} (confidence: {pattern_info['confidence']}%)")
        return pattern_info

    def get_market_context(self) -> Dict[str, Any]:
        """Analyze current market context"""
        print(f"📋 Analyzing market context...")

        if self.df is None:
            return {}

        last = self.df.iloc[-1]

        # Get trend description
        if last['ema_9'] > last['ema_20'] > last['ema_50']:
            trend = "strong_bullish"
        elif last['ema_9'] > last['ema_20']:
            trend = "bullish"
        elif last['ema_9'] < last['ema_20'] < last['ema_50']:
            trend = "strong_bearish"
        elif last['ema_9'] < last['ema_20']:
            trend = "bearish"
        else:
            trend = "neutral"

        # Get RSI state
        if last['rsi'] >= 70:
            rsi_state = "overbought"
        elif last['rsi'] <= 30:
            rsi_state = "oversold"
        elif last['rsi'] > 55:
            rsi_state = "bullish"
        elif last['rsi'] < 45:
            rsi_state = "bearish"
        else:
            rsi_state = "neutral"

        # Get volume state
        if last['volume_ratio'] >= 2.0:
            vol_state = "very_high"
        elif last['volume_ratio'] >= 1.5:
            vol_state = "high"
        elif last['volume_ratio'] >= 0.8:
            vol_state = "normal"
        else:
            vol_state = "low"

        # Get trading session
        hour = last['timestamp'].hour
        if 0 <= hour < 8:
            session = "asia"
        elif 8 <= hour < 13:
            session = "london"
        elif 13 <= hour < 17:
            session = "newyork"
        elif 17 <= hour < 22:
            session = "us_late"
        else:
            session = "pacific"

        # Find support and resistance levels
        recent_highs = self.df['high'].tail(20).max()
        recent_lows = self.df['low'].tail(20).min()
        current_price = last['close']

        resistance_levels = [
            recent_highs,
            last['bb_upper'],
            last['ema_200']
        ]
        support_levels = [
            recent_lows,
            last['bb_lower'],
            last['ema_200']
        ]

        # Clean and sort levels
        resistance_levels = sorted([x for x in resistance_levels if x > current_price and not np.isnan(x)])
        support_levels = sorted([x for x in support_levels if x < current_price and not np.isnan(x)], reverse=True)

        context = {
            "trend": trend,
            "rsi_state": rsi_state,
            "volume_state": vol_state,
            "session": session,
            "current_price": float(current_price),
            "price_position": {
                "distance_to_resistance": float(resistance_levels[0] - current_price) if resistance_levels else 0,
                "distance_to_support": float(current_price - support_levels[0]) if support_levels else 0
            },
            "levels": {
                "support": [float(x) for x in support_levels[:3]],
                "resistance": [float(x) for x in resistance_levels[:3]]
            }
        }

        print(f"✅ Context analyzed: {trend} trend, {rsi_state} RSI, {vol_state} volume")
        return context

    def generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        print(f"📝 Generating analysis summary...")

        if self.df is None:
            return {}

        # Get components
        pattern = self.detect_current_pattern()
        context = self.get_market_context()
        last = self.df.iloc[-1]

        # Calculate key metrics
        price_change_24h = ((last['close'] - self.df['close'].iloc[-96]) / self.df['close'].iloc[-96] * 100) if len(self.df) >= 96 else 0
        volatility = self.df['price_change_pct'].tail(20).std()
        volume_surge = last['volume_ratio']

        # Generate summary text
        summary_parts = [
            f"{self.symbol} {self.timeframe} Analysis",
            f"Price: {last['close']:.2f} ({price_change_24h:+.2f}%)",
            f"Pattern: {pattern['name'].replace('_', ' ').title()} ({pattern['confidence']}% confidence)",
            f"Trend: {context['trend'].replace('_', ' ').title()}",
            f"RSI: {last['rsi']:.0f} ({context['rsi_state']})",
            f"Volume: {volume_surge:.1f}x average ({context['volume_state']})",
            f"Session: {context['session'].title()}",
            f"ATR: {last['atr']:.2f}",
            f"Volatility: {volatility:.2f}%"
        ]

        summary = " | ".join(summary_parts)

        # Generate trading recommendation (advanced if available)
        if self.trade_engine and self.PatternInfo:
            recommendation = self._generate_advanced_recommendation(pattern, context, last)
        else:
            recommendation = self._generate_basic_recommendation(pattern, context, last)

        analysis = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": last['timestamp'].isoformat(),
            "current_price": float(last['close']),
            "pattern": pattern,
            "context": context,
            "indicators": {
                "momentum": {
                    "rsi": float(last['rsi']),
                    "macd": float(last['macd']),
                    "macd_signal": float(last['macd_signal']),
                    "stoch_k": float(last['stoch_k']),
                    "stoch_d": float(last['stoch_d'])
                },
                "trend": {
                    "ema_9": float(last['ema_9']),
                    "ema_20": float(last['ema_20']),
                    "ema_50": float(last['ema_50']),
                    "sma_20": float(last['sma_20']),
                    "sma_50": float(last['sma_50'])
                },
                "volatility": {
                    "atr": float(last['atr']),
                    "bb_upper": float(last['bb_upper']),
                    "bb_middle": float(last['bb_middle']),
                    "bb_lower": float(last['bb_lower'])
                },
                "volume": {
                    "volume_ratio": float(last['volume_ratio']),
                    "vwap": float(last['vwap'])
                }
            },
            "market_metrics": {
                "price_change_24h_pct": float(price_change_24h),
                "volatility_20d": float(volatility),
                "volume_surge": float(volume_surge)
            },
            "recommendation": recommendation,
            "summary": summary,
            "data_points": len(self.df),
            "analysis_time": datetime.now().isoformat()
        }

        self.analysis_result = analysis
        print(f"✅ Analysis summary generated")
        return analysis

    def _generate_basic_recommendation(self, pattern: Dict, context: Dict, last_candle: pd.Series) -> Dict[str, Any]:
        """Generate basic trade recommendation"""

        # Simple recommendation logic based on pattern and context
        direction = "HOLD"
        confidence = 50
        reasoning = []

        # Bullish signals
        bullish_signals = 0
        bearish_signals = 0

        if pattern['direction'] == 'bullish' and pattern['confidence'] >= 70:
            bullish_signals += 1
            reasoning.append(f"Bullish {pattern['name']} pattern detected")

        if context['rsi_state'] == 'oversold':
            bullish_signals += 1
            reasoning.append("RSI oversold - potential bounce")

        if context['trend'] in ['bullish', 'strong_bullish']:
            bullish_signals += 1
            reasoning.append(f"Trend is {context['trend']}")

        if context['volume_state'] in ['high', 'very_high']:
            bullish_signals += 0.5
            reasoning.append("High volume supports move")

        # Bearish signals
        if pattern['direction'] == 'bearish' and pattern['confidence'] >= 70:
            bearish_signals += 1
            reasoning.append(f"Bearish {pattern['name']} pattern detected")

        if context['rsi_state'] == 'overbought':
            bearish_signals += 1
            reasoning.append("RSI overbought - potential pullback")

        if context['trend'] in ['bearish', 'strong_bearish']:
            bearish_signals += 1
            reasoning.append(f"Trend is {context['trend']}")

        # Determine direction and confidence
        if bullish_signals > bearish_signals + 1:
            direction = "LONG"
            confidence = min(50 + (bullish_signals * 15), 85)
        elif bearish_signals > bullish_signals + 1:
            direction = "SHORT"
            confidence = min(50 + (bearish_signals * 15), 85)
        else:
            reasoning.append("Conflicting signals - wait for clarity")

        # Calculate basic levels
        current_price = float(last_candle['close'])
        atr = float(last_candle['atr'])

        recommendation = {
            "direction": direction,
            "confidence": int(confidence),
            "entry_price": current_price,
            "stop_loss": current_price - (2 * atr) if direction == "LONG" else current_price + (2 * atr),
            "take_profit": current_price + (3 * atr) if direction == "LONG" else current_price - (3 * atr),
            "risk_reward_ratio": 1.5,
            "reasoning": "; ".join(reasoning) if reasoning else "Insufficient signals",
            "strength": "Strong" if confidence >= 75 else "Moderate" if confidence >= 60 else "Weak"
        }

        return recommendation

    def _generate_advanced_recommendation(self, pattern: Dict, context: Dict, last_candle: pd.Series) -> Dict[str, Any]:
        """Generate advanced trade recommendation using trade recommendation engine"""
        print(f"🎯 Using advanced trade recommendation engine...")

        try:
            # Create PatternInfo object
            pattern_info = self.PatternInfo(
                name=pattern['name'],
                type=pattern['type'],
                direction=pattern['direction'],
                quality=pattern['quality'],
                confidence=pattern['confidence']
            )

            # Generate trade setup using the advanced engine
            trade_setup = self.trade_engine.generate_trade_setup(
                df=self.df,
                pattern=pattern_info,
                symbol=self.symbol,
                timeframe=self.timeframe
            )

            # Convert to the expected format
            return {
                "direction": trade_setup.get('direction', 'HOLD'),
                "confidence": trade_setup.get('confidence', 50),
                "entry_price": trade_setup.get('entry_price', last_candle['close']),
                "stop_loss": trade_setup.get('stop_loss', last_candle['close']),
                "take_profit": trade_setup.get('take_profit', last_candle['close']),
                "risk_reward_ratio": trade_setup.get('risk_reward_ratio', 1.5),
                "risk_pct": trade_setup.get('risk_pct', 1.0),
                "reward_pct": trade_setup.get('reward_pct', 1.5),
                "reasoning": trade_setup.get('reasoning', 'Advanced recommendation'),
                "strength": trade_setup.get('strength', 'Moderate'),
                "entry_reason": trade_setup.get('entry_reason', 'Pattern-based entry'),
                "stop_loss_reason": trade_setup.get('stop_loss_reason', 'Risk management'),
                "take_profit_reason": trade_setup.get('take_profit_reason', 'R:R based target'),
                "confidence_breakdown": trade_setup.get('confidence_breakdown', {})
            }

        except Exception as e:
            print(f"⚠️  Advanced recommendation failed: {e}")
            print(f"   Falling back to basic recommendation...")
            return self._generate_basic_recommendation(pattern, context, last_candle)

    def store_in_chromadb(self, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis in ChromaDB if --add-to-rag flag is used"""
        if not self.chroma_analyzer:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from chroma_live_analyzer import ChromaLiveAnalyzer
                self.chroma_analyzer = ChromaLiveAnalyzer()
            except ImportError as e:
                print(f"⚠️  ChromaDB not available - skipping storage ({e})")
                return False

        try:
            success = self.chroma_analyzer.store_live_analysis(analysis_data)
            if success:
                print(f"✅ Analysis stored in ChromaDB 'live_analysis' collection")
            return success
        except Exception as e:
            print(f"❌ Failed to store in ChromaDB: {e}")
            return False

    def save_json(self, output_path: str, data: Dict[str, Any]) -> bool:
        """Save analysis to JSON file"""
        print(f"💾 Saving analysis to: {output_path}")

        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            file_size = os.path.getsize(output_path)
            print(f"✅ Saved {file_size / 1024:.1f} KB")
            return True

        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False

    def analyze(self, csv_path: str) -> Dict[str, Any]:
        """Complete analysis pipeline"""
        print("=" * 70)
        print("LIVE TRADING ANALYZER")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Input: {csv_path}")
        print("=" * 70)

        # Step 1: Load CSV
        if not self.load_csv(csv_path):
            return {"error": "Failed to load CSV"}

        # Step 2: Normalize columns
        if not self.normalize_columns():
            return {"error": "Failed to normalize columns"}

        # Step 3: Parse timestamps
        if not self.parse_timestamps():
            return {"error": "Failed to parse timestamps"}

        # Step 4: Calculate indicators
        if not self.calculate_indicators():
            return {"error": "Failed to calculate indicators"}

        # Step 5: Generate analysis
        analysis = self.generate_analysis_summary()
        if not analysis:
            return {"error": "Failed to generate analysis"}

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"📊 Pattern: {analysis['pattern']['name']} ({analysis['pattern']['confidence']}% confidence)")
        print(f"📈 Direction: {analysis['recommendation']['direction']}")
        print(f"🎯 Confidence: {analysis['recommendation']['confidence']}%")
        print(f"💡 Entry: {analysis['recommendation']['entry_price']:.2f}")
        print(f"🛡️  SL: {analysis['recommendation']['stop_loss']:.2f}")
        print(f"🎉 TP: {analysis['recommendation']['take_profit']:.2f}")
        print(f"📏 R:R: {analysis['recommendation']['risk_reward_ratio']}:1")

        return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Live Trading Analyzer - Phase 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze XAUUSD M15 data
  python scripts/live_trading_analyzer.py \\
      --input data/XAUUSD_M15_200.csv \\
      --symbol XAUUSD --timeframe M15 \\
      --output data/live_analysis/XAUUSD_M15_analysis.json

  # Analyze and store in ChromaDB
  python scripts/live_trading_analyzer.py \\
      --input data/XAUUSD_M15_200.csv \\
      --symbol XAUUSD --timeframe M15 \\
      --add-to-rag

  # Analyze BTCUSD H1 data with full features
  python scripts/live_trading_analyzer.py \\
      --input data/BTCUSD_H1_200.csv \\
      --symbol BTCUSD --timeframe H1 \\
      --output data/live_analysis/BTCUSD_H1_analysis.json \\
      --add-to-rag
        """
    )

    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol')
    parser.add_argument('--timeframe', default='M15', help='Timeframe')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--add-to-rag', action='store_true', help='Store analysis in ChromaDB')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Create analyzer
    analyzer = LiveTradingAnalyzer(args.symbol, args.timeframe)

    # Run analysis
    result = analyzer.analyze(args.input)

    if "error" in result:
        print(f"❌ Analysis failed: {result['error']}")
        return 1

    # Save if output specified
    if args.output:
        if not analyzer.save_json(args.output, result):
            return 1
        print(f"\n💾 Analysis saved to: {args.output}")
    else:
        # Save to default location
        default_output = f"data/live_analysis/{args.symbol}_{args.timeframe}_analysis.json"
        if analyzer.save_json(default_output, result):
            print(f"\n💾 Analysis saved to: {default_output}")

    # Store in ChromaDB if requested
    if args.add_to_rag:
        print(f"\n📊 Storing analysis in ChromaDB...")
        if analyzer.store_in_chromadb(result):
            print(f"✅ Analysis stored in ChromaDB 'live_analysis' collection")
            print(f"   You can query it later with:")
            print(f"   python scripts/chroma_live_analyzer.py")
        else:
            print(f"❌ Failed to store in ChromaDB")

    return 0


if __name__ == "__main__":
    sys.exit(main())