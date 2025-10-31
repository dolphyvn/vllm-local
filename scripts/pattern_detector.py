#!/usr/bin/env python3
"""
Pattern Detector - Trading Pattern Recognition
Optimized for small LLM + ChromaDB RAG system

Detects trading patterns from structured JSON and formats for RAG ingestion:
- Candlestick patterns (engulfing, doji, hammer, etc.)
- Technical patterns (support/resistance bounces, breakouts)
- Divergences (RSI, MACD)
- Outcome tracking (forward-looking analysis)

Usage:
    python scripts/pattern_detector.py \
        --input data/structured/XAUUSD_M15_full.json \
        --output data/patterns/XAUUSD_M15_patterns.json \
        --lookforward 20
"""

import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path


class PatternDetector:
    """Detect and format trading patterns for RAG"""

    def __init__(self, lookforward: int = 20, min_pattern_quality: float = 0.6):
        """
        Initialize pattern detector

        Args:
            lookforward: Bars to look forward for outcome analysis
            min_pattern_quality: Minimum quality threshold (0-1)
        """
        self.lookforward = lookforward
        self.min_pattern_quality = min_pattern_quality
        self.candles: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def load_structured_json(self, json_path: str) -> bool:
        """Load structured JSON from mt5_to_structured_json.py"""
        print(f"📂 Loading: {json_path}")

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            self.metadata = data.get('metadata', {})
            self.candles = data.get('candles', [])

            print(f"✅ Loaded {len(self.candles)} candles")
            print(f"   Symbol: {self.metadata.get('symbol')}")
            print(f"   Timeframe: {self.metadata.get('timeframe')}")
            return True

        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False

    # ==================== PATTERN DETECTION ====================

    def detect_all_patterns(self) -> List[Dict[str, Any]]:
        """Detect all patterns and return structured results"""
        print(f"\n🔍 Detecting patterns...")
        print(f"   Lookback start: candle {100}")
        print(f"   Lookforward: {self.lookforward} bars")

        patterns = []
        start_idx = 100  # Need history for context
        end_idx = len(self.candles) - self.lookforward  # Need future for outcome

        if end_idx <= start_idx:
            print(f"❌ Insufficient data. Need at least {start_idx + self.lookforward} candles")
            return []

        for i in range(start_idx, end_idx):
            current = self.candles[i]

            # Detect various pattern types
            detected = []

            # Candlestick patterns
            if pattern := self._detect_bullish_engulfing(i):
                detected.append(('Bullish Engulfing', pattern))

            if pattern := self._detect_bearish_engulfing(i):
                detected.append(('Bearish Engulfing', pattern))

            if pattern := self._detect_hammer(i):
                detected.append(('Hammer', pattern))

            if pattern := self._detect_shooting_star(i):
                detected.append(('Shooting Star', pattern))

            if pattern := self._detect_doji(i):
                detected.append(('Doji', pattern))

            # Technical patterns
            if pattern := self._detect_rsi_divergence(i, 'bullish'):
                detected.append(('RSI Bullish Divergence', pattern))

            if pattern := self._detect_rsi_divergence(i, 'bearish'):
                detected.append(('RSI Bearish Divergence', pattern))

            if pattern := self._detect_breakout(i):
                detected.append(('Breakout', pattern))

            if pattern := self._detect_support_bounce(i):
                detected.append(('Support Bounce', pattern))

            if pattern := self._detect_resistance_rejection(i):
                detected.append(('Resistance Rejection', pattern))

            # Format each detected pattern
            for pattern_name, pattern_data in detected:
                formatted = self._format_pattern(i, pattern_name, pattern_data)
                if formatted and formatted.get('quality_score', 0) >= self.min_pattern_quality:
                    patterns.append(formatted)

            # Progress indicator
            if (i - start_idx + 1) % 500 == 0:
                print(f"   Processed {i - start_idx + 1}/{end_idx - start_idx} candles...")

        print(f"✅ Detected {len(patterns)} patterns (quality >= {self.min_pattern_quality})")
        return patterns

    def _detect_bullish_engulfing(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect bullish engulfing pattern"""
        if idx < 1:
            return None

        curr = self.candles[idx]
        prev = self.candles[idx - 1]

        curr_ohlc = curr['ohlc']
        prev_ohlc = prev['ohlc']

        # Conditions
        prev_bearish = prev_ohlc['close'] < prev_ohlc['open']
        curr_bullish = curr_ohlc['close'] > curr_ohlc['open']
        engulfs = (curr_ohlc['open'] <= prev_ohlc['close'] and
                   curr_ohlc['close'] >= prev_ohlc['open'])

        if prev_bearish and curr_bullish and engulfs:
            # Calculate quality score
            body_ratio = curr['indicators']['price_action']['body_size'] / prev['indicators']['price_action']['body_size']
            volume_score = min(curr['indicators']['volume']['volume_ratio'] / 1.5, 1.0)
            quality = min((body_ratio * 0.5 + volume_score * 0.5), 1.0)

            return {
                'pattern_type': 'candlestick',
                'direction': 'bullish',
                'quality': quality,
                'details': {
                    'prev_candle': prev_ohlc,
                    'curr_candle': curr_ohlc,
                    'body_ratio': body_ratio,
                    'volume_ratio': curr['indicators']['volume']['volume_ratio']
                }
            }
        return None

    def _detect_bearish_engulfing(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect bearish engulfing pattern"""
        if idx < 1:
            return None

        curr = self.candles[idx]
        prev = self.candles[idx - 1]

        curr_ohlc = curr['ohlc']
        prev_ohlc = prev['ohlc']

        # Conditions
        prev_bullish = prev_ohlc['close'] > prev_ohlc['open']
        curr_bearish = curr_ohlc['close'] < curr_ohlc['open']
        engulfs = (curr_ohlc['open'] >= prev_ohlc['close'] and
                   curr_ohlc['close'] <= prev_ohlc['open'])

        if prev_bullish and curr_bearish and engulfs:
            body_ratio = curr['indicators']['price_action']['body_size'] / prev['indicators']['price_action']['body_size']
            volume_score = min(curr['indicators']['volume']['volume_ratio'] / 1.5, 1.0)
            quality = min((body_ratio * 0.5 + volume_score * 0.5), 1.0)

            return {
                'pattern_type': 'candlestick',
                'direction': 'bearish',
                'quality': quality,
                'details': {
                    'prev_candle': prev_ohlc,
                    'curr_candle': curr_ohlc,
                    'body_ratio': body_ratio,
                    'volume_ratio': curr['indicators']['volume']['volume_ratio']
                }
            }
        return None

    def _detect_hammer(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect hammer pattern (bullish reversal)"""
        curr = self.candles[idx]
        pa = curr['indicators']['price_action']

        body_size = pa['body_size']
        lower_wick = pa['lower_wick']
        upper_wick = pa['upper_wick']
        hl_range = pa['hl_range']

        # Hammer conditions
        has_long_lower_wick = lower_wick > (body_size * 2)
        has_small_upper_wick = upper_wick < (body_size * 0.5)
        body_in_upper_half = (curr['ohlc']['close'] > (curr['ohlc']['low'] + hl_range * 0.6))

        if has_long_lower_wick and has_small_upper_wick and body_in_upper_half:
            quality = min((lower_wick / hl_range) * 1.2, 1.0)

            return {
                'pattern_type': 'candlestick',
                'direction': 'bullish',
                'quality': quality,
                'details': {
                    'lower_wick_ratio': lower_wick / hl_range,
                    'body_position': 'upper_half'
                }
            }
        return None

    def _detect_shooting_star(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect shooting star pattern (bearish reversal)"""
        curr = self.candles[idx]
        pa = curr['indicators']['price_action']

        body_size = pa['body_size']
        lower_wick = pa['lower_wick']
        upper_wick = pa['upper_wick']
        hl_range = pa['hl_range']

        # Shooting star conditions
        has_long_upper_wick = upper_wick > (body_size * 2)
        has_small_lower_wick = lower_wick < (body_size * 0.5)
        body_in_lower_half = (curr['ohlc']['close'] < (curr['ohlc']['low'] + hl_range * 0.4))

        if has_long_upper_wick and has_small_lower_wick and body_in_lower_half:
            quality = min((upper_wick / hl_range) * 1.2, 1.0)

            return {
                'pattern_type': 'candlestick',
                'direction': 'bearish',
                'quality': quality,
                'details': {
                    'upper_wick_ratio': upper_wick / hl_range,
                    'body_position': 'lower_half'
                }
            }
        return None

    def _detect_doji(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect doji pattern (indecision)"""
        curr = self.candles[idx]
        pa = curr['indicators']['price_action']

        body_size = pa['body_size']
        hl_range = pa['hl_range']

        # Doji condition: very small body relative to range
        is_doji = body_size < (hl_range * 0.1)

        if is_doji and hl_range > 0:
            quality = 1.0 - (body_size / hl_range)

            return {
                'pattern_type': 'candlestick',
                'direction': 'neutral',
                'quality': quality,
                'details': {
                    'body_to_range_ratio': body_size / hl_range if hl_range > 0 else 0
                }
            }
        return None

    def _detect_rsi_divergence(self, idx: int, div_type: str) -> Optional[Dict[str, Any]]:
        """Detect RSI divergence"""
        lookback = 20
        if idx < lookback:
            return None

        curr = self.candles[idx]
        window = self.candles[idx - lookback:idx]

        curr_rsi = curr['indicators']['momentum']['rsi']
        curr_price = curr['ohlc']['close']

        if div_type == 'bullish':
            # Price lower low, RSI higher low
            window_low_price = min(c['ohlc']['low'] for c in window)
            window_low_rsi = min(c['indicators']['momentum']['rsi'] for c in window)

            price_ll = curr['ohlc']['low'] < window_low_price
            rsi_hl = curr_rsi > window_low_rsi
            rsi_oversold = curr_rsi < 40

            if price_ll and rsi_hl and rsi_oversold:
                divergence_strength = (curr_rsi - window_low_rsi) / 20  # Normalize
                quality = min(divergence_strength, 1.0)

                return {
                    'pattern_type': 'divergence',
                    'direction': 'bullish',
                    'quality': quality,
                    'details': {
                        'divergence_type': 'rsi_bullish',
                        'current_rsi': curr_rsi,
                        'previous_rsi_low': window_low_rsi,
                        'rsi_improvement': curr_rsi - window_low_rsi
                    }
                }

        elif div_type == 'bearish':
            # Price higher high, RSI lower high
            window_high_price = max(c['ohlc']['high'] for c in window)
            window_high_rsi = max(c['indicators']['momentum']['rsi'] for c in window)

            price_hh = curr['ohlc']['high'] > window_high_price
            rsi_lh = curr_rsi < window_high_rsi
            rsi_overbought = curr_rsi > 60

            if price_hh and rsi_lh and rsi_overbought:
                divergence_strength = (window_high_rsi - curr_rsi) / 20
                quality = min(divergence_strength, 1.0)

                return {
                    'pattern_type': 'divergence',
                    'direction': 'bearish',
                    'quality': quality,
                    'details': {
                        'divergence_type': 'rsi_bearish',
                        'current_rsi': curr_rsi,
                        'previous_rsi_high': window_high_rsi,
                        'rsi_decline': window_high_rsi - curr_rsi
                    }
                }

        return None

    def _detect_breakout(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect breakout pattern"""
        lookback = 20
        if idx < lookback:
            return None

        curr = self.candles[idx]
        window = self.candles[idx - lookback:idx]

        recent_high = max(c['ohlc']['high'] for c in window)
        close = curr['ohlc']['close']
        volume_ratio = curr['indicators']['volume']['volume_ratio']
        rsi = curr['indicators']['momentum']['rsi']

        # Breakout conditions
        breakout = close > recent_high
        volume_confirm = volume_ratio > 1.5
        momentum = rsi > 50

        if breakout and volume_confirm and momentum:
            breakout_strength = (close - recent_high) / recent_high
            quality = min((breakout_strength * 100 + volume_ratio * 0.3), 1.0)

            return {
                'pattern_type': 'technical',
                'direction': 'bullish',
                'quality': quality,
                'details': {
                    'breakout_level': recent_high,
                    'breakout_strength_pct': breakout_strength * 100,
                    'volume_ratio': volume_ratio
                }
            }
        return None

    def _detect_support_bounce(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect support bounce pattern"""
        lookback = 100
        if idx < lookback:
            return None

        curr = self.candles[idx]
        window = self.candles[idx - lookback:idx]

        # Find swing lows
        swing_lows = []
        for i in range(5, len(window) - 5):
            if window[i]['ohlc']['low'] == min(c['ohlc']['low'] for c in window[i-5:i+6]):
                swing_lows.append(window[i]['ohlc']['low'])

        if not swing_lows:
            return None

        curr_low = curr['ohlc']['low']
        atr = curr['indicators']['volatility']['atr']
        tolerance = atr * 0.5

        # Check if current low is near any swing low
        near_support = any(abs(curr_low - level) < tolerance for level in swing_lows)
        bullish_close = curr['ohlc']['close'] > curr['ohlc']['open']

        if near_support and bullish_close:
            closest_support = min(swing_lows, key=lambda x: abs(x - curr_low))
            distance = abs(curr_low - closest_support)
            quality = max(1.0 - (distance / atr), 0.5)

            return {
                'pattern_type': 'technical',
                'direction': 'bullish',
                'quality': quality,
                'details': {
                    'support_level': closest_support,
                    'distance_from_support': distance,
                    'atr_distance': distance / atr
                }
            }
        return None

    def _detect_resistance_rejection(self, idx: int) -> Optional[Dict[str, Any]]:
        """Detect resistance rejection pattern"""
        lookback = 100
        if idx < lookback:
            return None

        curr = self.candles[idx]
        window = self.candles[idx - lookback:idx]

        # Find swing highs
        swing_highs = []
        for i in range(5, len(window) - 5):
            if window[i]['ohlc']['high'] == max(c['ohlc']['high'] for c in window[i-5:i+6]):
                swing_highs.append(window[i]['ohlc']['high'])

        if not swing_highs:
            return None

        curr_high = curr['ohlc']['high']
        atr = curr['indicators']['volatility']['atr']
        tolerance = atr * 0.5

        # Check if current high is near any swing high
        near_resistance = any(abs(curr_high - level) < tolerance for level in swing_highs)
        bearish_close = curr['ohlc']['close'] < curr['ohlc']['open']

        if near_resistance and bearish_close:
            closest_resistance = min(swing_highs, key=lambda x: abs(x - curr_high))
            distance = abs(curr_high - closest_resistance)
            quality = max(1.0 - (distance / atr), 0.5)

            return {
                'pattern_type': 'technical',
                'direction': 'bearish',
                'quality': quality,
                'details': {
                    'resistance_level': closest_resistance,
                    'distance_from_resistance': distance,
                    'atr_distance': distance / atr
                }
            }
        return None

    # ==================== OUTCOME ANALYSIS ====================

    def _analyze_outcome(self, idx: int, direction: str) -> Dict[str, Any]:
        """
        Analyze pattern outcome by looking forward

        Args:
            idx: Current candle index
            direction: 'bullish', 'bearish', or 'neutral'

        Returns:
            Outcome analysis dictionary
        """
        curr = self.candles[idx]
        entry_price = curr['ohlc']['close']
        atr = curr['indicators']['volatility']['atr']

        # Define targets and stops based on direction
        if direction == 'bullish':
            target = entry_price + (atr * 2)
            stop = entry_price - atr
        elif direction == 'bearish':
            target = entry_price - (atr * 2)
            stop = entry_price + atr
        else:  # neutral
            target = entry_price + (atr * 1.5)
            stop = entry_price - (atr * 1.5)

        # Look forward
        end_idx = min(idx + self.lookforward + 1, len(self.candles))
        future = self.candles[idx + 1:end_idx]

        if not future:
            return {
                'result': 'UNKNOWN',
                'reason': 'insufficient_data',
                'pnl_points': 0,
                'pnl_pct': 0,
                'duration_bars': 0,
                'mfe': 0,  # Maximum Favorable Excursion
                'mae': 0   # Maximum Adverse Excursion
            }

        # Calculate MFE and MAE
        if direction == 'bullish' or direction == 'neutral':
            mfe = max(c['ohlc']['high'] for c in future) - entry_price
            mae = min(c['ohlc']['low'] for c in future) - entry_price
        else:
            mfe = entry_price - min(c['ohlc']['low'] for c in future)
            mae = entry_price - max(c['ohlc']['high'] for c in future)

        # Check if target or stop hit
        for i, candle in enumerate(future):
            if direction == 'bullish':
                if candle['ohlc']['high'] >= target:
                    return {
                        'result': 'WIN',
                        'reason': 'target_hit',
                        'pnl_points': target - entry_price,
                        'pnl_pct': ((target - entry_price) / entry_price) * 100,
                        'duration_bars': i + 1,
                        'mfe': mfe,
                        'mae': mae,
                        'exit_price': target
                    }
                elif candle['ohlc']['low'] <= stop:
                    return {
                        'result': 'LOSS',
                        'reason': 'stop_hit',
                        'pnl_points': stop - entry_price,
                        'pnl_pct': ((stop - entry_price) / entry_price) * 100,
                        'duration_bars': i + 1,
                        'mfe': mfe,
                        'mae': mae,
                        'exit_price': stop
                    }

            elif direction == 'bearish':
                if candle['ohlc']['low'] <= target:
                    return {
                        'result': 'WIN',
                        'reason': 'target_hit',
                        'pnl_points': entry_price - target,
                        'pnl_pct': ((entry_price - target) / entry_price) * 100,
                        'duration_bars': i + 1,
                        'mfe': mfe,
                        'mae': mae,
                        'exit_price': target
                    }
                elif candle['ohlc']['high'] >= stop:
                    return {
                        'result': 'LOSS',
                        'reason': 'stop_hit',
                        'pnl_points': entry_price - stop,
                        'pnl_pct': ((entry_price - stop) / entry_price) * 100,
                        'duration_bars': i + 1,
                        'mfe': mfe,
                        'mae': mae,
                        'exit_price': stop
                    }

        # Neither target nor stop hit
        final_price = future[-1]['ohlc']['close']
        return {
            'result': 'NEUTRAL',
            'reason': 'time_exit',
            'pnl_points': final_price - entry_price if direction == 'bullish' else entry_price - final_price,
            'pnl_pct': ((final_price - entry_price) / entry_price) * 100,
            'duration_bars': len(future),
            'mfe': mfe,
            'mae': mae,
            'exit_price': final_price
        }

    # ==================== PATTERN FORMATTING ====================

    def _format_pattern(self, idx: int, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format pattern into RAG-optimized structure"""
        curr = self.candles[idx]
        direction = pattern_data.get('direction', 'neutral')

        # Analyze outcome
        outcome = self._analyze_outcome(idx, direction)

        # Create structured pattern document
        pattern_id = f"{pattern_name.lower().replace(' ', '_')}_{curr['timestamp'].replace(':', '').replace('-', '')}"

        # Calculate overall quality score
        pattern_quality = pattern_data.get('quality', 0.7)
        volume_score = min(curr['indicators']['volume']['volume_ratio'] / 2.0, 1.0)
        trend_alignment = self._calculate_trend_alignment(curr, direction)
        quality_score = (pattern_quality * 0.5 + volume_score * 0.2 + trend_alignment * 0.3)

        # Create summary (optimized for embeddings)
        summary = (
            f"{pattern_name} detected on {self.metadata['symbol']} {self.metadata['timeframe']} | "
            f"Direction: {direction} | "
            f"Entry: {curr['ohlc']['close']:.2f} | "
            f"RSI: {curr['indicators']['momentum']['rsi']:.0f} | "
            f"Trend: {curr['context']['trend']} | "
            f"Volume: {curr['indicators']['volume']['volume_ratio']:.1f}x | "
            f"Session: {curr['context']['session']} | "
            f"Outcome: {outcome['result']} ({outcome['pnl_pct']:+.2f}%)"
        )

        return {
            "pattern_id": pattern_id,
            "symbol": self.metadata['symbol'],
            "timeframe": self.metadata['timeframe'],
            "timestamp": curr['timestamp'],

            "pattern": {
                "name": pattern_name,
                "type": pattern_data['pattern_type'],
                "direction": direction,
                "quality": pattern_quality,
                "details": pattern_data.get('details', {})
            },

            "entry": {
                "price": curr['ohlc']['close'],
                "candle_index": idx,
                "ohlc": curr['ohlc']
            },

            "indicators": curr['indicators'],
            "context": curr['context'],

            "outcome": outcome,

            "quality_score": quality_score,

            "summary": summary,

            "metadata": {
                "detected_at": datetime.now().isoformat(),
                "lookforward_bars": self.lookforward
            }
        }

    def _calculate_trend_alignment(self, candle: Dict[str, Any], direction: str) -> float:
        """Calculate how well the pattern aligns with the trend"""
        trend = candle['context']['trend']

        if direction == 'bullish' and 'bullish' in trend:
            return 1.0
        elif direction == 'bearish' and 'bearish' in trend:
            return 1.0
        elif direction == 'neutral':
            return 0.7
        else:
            return 0.3

    # ==================== EXPORT ====================

    def save_patterns(self, output_path: str, patterns: List[Dict[str, Any]]) -> bool:
        """Save detected patterns to JSON"""
        print(f"\n💾 Saving patterns to: {output_path}")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Calculate statistics
            total = len(patterns)
            wins = sum(1 for p in patterns if p['outcome']['result'] == 'WIN')
            losses = sum(1 for p in patterns if p['outcome']['result'] == 'LOSS')
            neutrals = total - wins - losses

            avg_quality = sum(p['quality_score'] for p in patterns) / total if total > 0 else 0
            avg_pnl = sum(p['outcome']['pnl_points'] for p in patterns) / total if total > 0 else 0

            output = {
                "metadata": {
                    "source": self.metadata,
                    "detection_settings": {
                        "lookforward_bars": self.lookforward,
                        "min_pattern_quality": self.min_pattern_quality
                    },
                    "statistics": {
                        "total_patterns": total,
                        "wins": wins,
                        "losses": losses,
                        "neutrals": neutrals,
                        "win_rate": (wins / total * 100) if total > 0 else 0,
                        "avg_quality_score": avg_quality,
                        "avg_pnl_points": avg_pnl
                    },
                    "generated_at": datetime.now().isoformat()
                },
                "patterns": patterns
            }

            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)

            file_size = os.path.getsize(output_path)
            print(f"✅ Saved {file_size / 1024:.1f} KB")
            print(f"\n📊 Statistics:")
            print(f"   Total patterns: {total}")
            print(f"   Wins: {wins} ({wins/total*100:.1f}%)" if total > 0 else "   Wins: 0")
            print(f"   Losses: {losses} ({losses/total*100:.1f}%)" if total > 0 else "   Losses: 0")
            print(f"   Neutrals: {neutrals} ({neutrals/total*100:.1f}%)" if total > 0 else "   Neutrals: 0")
            print(f"   Avg quality: {avg_quality:.2f}")
            print(f"   Avg P&L: {avg_pnl:+.2f} points")

            return True

        except Exception as e:
            print(f"❌ Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Detect trading patterns from structured JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect patterns with default settings
  python scripts/pattern_detector.py \\
      --input data/structured/XAUUSD_M15_full.json \\
      --output data/patterns/XAUUSD_M15_patterns.json

  # Adjust lookforward and quality threshold
  python scripts/pattern_detector.py \\
      --input data/structured/XAUUSD_M15_full.json \\
      --output data/patterns/XAUUSD_M15_patterns.json \\
      --lookforward 30 --min-quality 0.7
        """
    )

    parser.add_argument('--input', required=True, help='Input structured JSON file')
    parser.add_argument('--output', required=True, help='Output patterns JSON file')
    parser.add_argument('--lookforward', type=int, default=20, help='Bars to look forward for outcome (default: 20)')
    parser.add_argument('--min-quality', type=float, default=0.6, help='Minimum pattern quality (0-1, default: 0.6)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    print("=" * 70)
    print("PATTERN DETECTOR")
    print("=" * 70)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Lookforward:  {args.lookforward} bars")
    print(f"Min Quality:  {args.min_quality}")
    print("=" * 70)

    # Initialize detector
    detector = PatternDetector(
        lookforward=args.lookforward,
        min_pattern_quality=args.min_quality
    )

    # Load data
    if not detector.load_structured_json(args.input):
        return 1

    # Detect patterns
    patterns = detector.detect_all_patterns()

    if not patterns:
        print("⚠️  No patterns detected")
        return 1

    # Save results
    if not detector.save_patterns(args.output, patterns):
        return 1

    print("\n" + "=" * 70)
    print("✅ PATTERN DETECTION COMPLETE")
    print("=" * 70)
    print(f"💾 Output: {args.output}")
    print("\n💡 Next step:")
    print(f"   python scripts/rag_structured_feeder.py --input {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
