#!/usr/bin/env python3
"""
Trade Recommendation Engine - Phase 2 Implementation
Generates actionable trade setups with Entry/SL/TP calculations

Features:
- Entry price calculation based on patterns and market structure
- Stop Loss calculation with ATR and support/resistance levels
- Take Profit calculation with risk/reward ratios
- Confidence scoring system
- Detailed reasoning generation
- Risk management integration

Usage:
    from scripts.trade_recommendation_engine import TradeRecommendationEngine

    engine = TradeRecommendationEngine()
    setup = engine.generate_trade_setup(market_data, pattern_info, indicators)
"""

import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketLevels:
    """Market structure levels"""
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    atr: float
    round_numbers: List[float]
    fibonacci_levels: Dict[str, float]


@dataclass
class PatternInfo:
    """Pattern information"""
    name: str
    type: str  # reversal, continuation, breakout
    direction: str  # bullish, bearish, neutral
    quality: float  # 0-1
    confidence: int  # 0-100
    entry_trigger: Optional[float] = None
    stop_level: Optional[float] = None


class TradeRecommendationEngine:
    """Generate actionable trade setups with Entry/SL/TP"""

    def __init__(self, risk_per_trade: float = 1.0, min_rr_ratio: float = 1.5):
        """
        Initialize trade recommendation engine

        Args:
            risk_per_trade: Risk per trade in percentage (default 1%)
            min_rr_ratio: Minimum risk/reward ratio (default 1.5)
        """
        self.risk_per_trade = risk_per_trade
        self.min_rr_ratio = min_rr_ratio
        self.psychological_levels = [0.0, 10.0, 20.0, 50.0, 80.0, 90.0, 100.0]  # For XAUUSD-like prices

    def calculate_market_levels(self, df: pd.DataFrame, lookback: int = 100) -> MarketLevels:
        """
        Calculate key market levels

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of candles to look back for levels

        Returns:
            MarketLevels object with support/resistance and other levels
        """
        if len(df) < lookback:
            lookback = len(df)

        recent_data = df.tail(lookback)
        current_price = float(df.iloc[-1]['close'])
        atr = float(df.iloc[-1]['atr']) if 'atr' in df.columns else self._calculate_atr(recent_data)

        # Find swing highs and lows for support/resistance
        swing_highs = []
        swing_lows = []

        # Simple swing detection (2 candles on each side)
        for i in range(2, len(recent_data) - 2):
            # Swing high
            if (recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and
                recent_data.iloc[i]['high'] > recent_data.iloc[i-2]['high'] and
                recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high'] and
                recent_data.iloc[i]['high'] > recent_data.iloc[i+2]['high']):
                swing_highs.append(float(recent_data.iloc[i]['high']))

            # Swing low
            if (recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and
                recent_data.iloc[i]['low'] < recent_data.iloc[i-2]['low'] and
                recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low'] and
                recent_data.iloc[i]['low'] < recent_data.iloc[i+2]['low']):
                swing_lows.append(float(recent_data.iloc[i]['low']))

        # Filter and sort levels
        support_levels = sorted([x for x in swing_lows if x < current_price], reverse=True)[:5]
        resistance_levels = sorted([x for x in swing_highs if x > current_price])[:5]

        # Add moving averages as dynamic levels
        if 'ema_20' in df.columns:
            ema_20 = float(df.iloc[-1]['ema_20'])
            if ema_20 < current_price:
                support_levels.append(ema_20)
            else:
                resistance_levels.append(ema_20)

        if 'ema_50' in df.columns:
            ema_50 = float(df.iloc[-1]['ema_50'])
            if ema_50 < current_price:
                support_levels.append(ema_50)
            else:
                resistance_levels.append(ema_50)

        # Sort and deduplicate
        support_levels = sorted(list(set(support_levels)), reverse=True)[:5]
        resistance_levels = sorted(list(set(resistance_levels)))[:5]

        # Calculate round numbers (psychological levels)
        round_numbers = self._calculate_round_numbers(current_price)

        # Calculate Fibonacci levels from recent swing
        fibonacci_levels = self._calculate_fibonacci_levels(recent_data)

        return MarketLevels(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            current_price=current_price,
            atr=atr,
            round_numbers=round_numbers,
            fibonacci_levels=fibonacci_levels
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR manually if not in DataFrame"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(true_range.rolling(window=period, min_periods=1).mean().iloc[-1])

    def _calculate_round_numbers(self, current_price: float) -> List[float]:
        """Calculate psychological round numbers"""
        round_numbers = []

        # Determine the price scale
        if current_price < 100:
            scale = 1
        elif current_price < 1000:
            scale = 10
        elif current_price < 10000:
            scale = 100
        else:
            scale = 1000

        # Generate round numbers above and below current price
        base_round = round(current_price / scale) * scale

        for i in range(-5, 6):
            round_numbers.append(base_round + (i * scale))

        # Filter to relevant range
        return [x for x in round_numbers if abs(x - current_price) < current_price * 0.1]

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        if len(df) < 20:
            return {}

        # Find recent swing high and low
        lookback = min(100, len(df))
        recent_data = df.tail(lookback)

        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()

        diff = swing_high - swing_low

        fib_levels = {
            'swing_high': swing_high,
            'swing_low': swing_low,
            '0.0%': swing_low,
            '23.6%': swing_low + (diff * 0.236),
            '38.2%': swing_low + (diff * 0.382),
            '50.0%': swing_low + (diff * 0.5),
            '61.8%': swing_low + (diff * 0.618),
            '78.6%': swing_low + (diff * 0.786),
            '100.0%': swing_high
        }

        return fib_levels

    def calculate_entry_price(self,
                            pattern: PatternInfo,
                            levels: MarketLevels,
                            df: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate optimal entry price

        Args:
            pattern: Pattern information
            levels: Market levels
            df: Price data

        Returns:
            Tuple of (entry_price, entry_reason)
        """
        current_price = levels.current_price
        entry_reason = ""

        # Entry logic based on pattern type
        if pattern.type == "reversal":
            if pattern.direction == "bullish":
                # Wait for confirmation of reversal
                nearest_support = levels.support_levels[0] if levels.support_levels else current_price - (2 * levels.atr)
                entry_price = nearest_support + (0.5 * levels.atr)
                entry_reason = f"Entry above nearest support at {nearest_support:.2f} for bullish reversal"
            else:
                # Bearish reversal
                nearest_resistance = levels.resistance_levels[0] if levels.resistance_levels else current_price + (2 * levels.atr)
                entry_price = nearest_resistance - (0.5 * levels.atr)
                entry_reason = f"Entry below nearest resistance at {nearest_resistance:.2f} for bearish reversal"

        elif pattern.type == "continuation":
            if pattern.direction == "bullish":
                # Buy the dip in uptrend
                nearest_support = levels.support_levels[0] if levels.support_levels else current_price - (1.5 * levels.atr)
                entry_price = nearest_support + (0.3 * levels.atr)
                entry_reason = f"Buy the dip at support level {nearest_support:.2f} in uptrend"
            else:
                # Sell the rally in downtrend
                nearest_resistance = levels.resistance_levels[0] if levels.resistance_levels else current_price + (1.5 * levels.atr)
                entry_price = nearest_resistance - (0.3 * levels.atr)
                entry_reason = f"Sell the rally at resistance level {nearest_resistance:.2f} in downtrend"

        elif pattern.type == "breakout":
            if pattern.direction == "bullish":
                # Breakout above resistance
                nearest_resistance = levels.resistance_levels[0] if levels.resistance_levels else current_price + (1 * levels.atr)
                entry_price = nearest_resistance + (0.2 * levels.atr)
                entry_reason = f"Breakout entry above resistance at {nearest_resistance:.2f}"
            else:
                # Breakout below support
                nearest_support = levels.support_levels[0] if levels.support_levels else current_price - (1 * levels.atr)
                entry_price = nearest_support - (0.2 * levels.atr)
                entry_reason = f"Breakout entry below support at {nearest_support:.2f}"
        else:
            # Default to current price for neutral patterns
            entry_price = current_price
            entry_reason = "Entry at current price for neutral setup"

        # Validate entry price is reasonable
        if abs(entry_price - current_price) > current_price * 0.05:  # More than 5% away
            entry_price = current_price
            entry_reason = "Adjusted entry to current price (original too far away)"

        return entry_price, entry_reason

    def calculate_stop_loss(self,
                          entry_price: float,
                          direction: str,
                          levels: MarketLevels,
                          df: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate stop loss level

        Args:
            entry_price: Planned entry price
            direction: Trade direction (LONG/SHORT)
            levels: Market levels
            df: Price data

        Returns:
            Tuple of (stop_loss, sl_reason)
        """
        atr = levels.atr

        if direction == "LONG":
            # Stop loss below entry
            # Find nearest support below entry
            valid_supports = [s for s in levels.support_levels if s < entry_price]

            if valid_supports:
                nearest_support = valid_supports[0]
                stop_loss = min(nearest_support - (0.2 * atr), entry_price - (2 * atr))
                sl_reason = f"Stop below nearest support at {nearest_support:.2f}"
            else:
                # Use ATR-based stop
                stop_loss = entry_price - (2 * atr)
                sl_reason = f"ATR-based stop at 2x ATR ({2*atr:.2f})"

            # Ensure stop is not too far
            max_distance = entry_price * 0.03  # Max 3% risk
            if entry_price - stop_loss > max_distance:
                stop_loss = entry_price - max_distance
                sl_reason += " (adjusted to max 3% risk)"

        else:  # SHORT
            # Stop loss above entry
            # Find nearest resistance above entry
            valid_resistances = [r for r in levels.resistance_levels if r > entry_price]

            if valid_resistances:
                nearest_resistance = valid_resistances[0]
                stop_loss = max(nearest_resistance + (0.2 * atr), entry_price + (2 * atr))
                sl_reason = f"Stop above nearest resistance at {nearest_resistance:.2f}"
            else:
                # Use ATR-based stop
                stop_loss = entry_price + (2 * atr)
                sl_reason = f"ATR-based stop at 2x ATR ({2*atr:.2f})"

            # Ensure stop is not too far
            max_distance = entry_price * 0.03  # Max 3% risk
            if stop_loss - entry_price > max_distance:
                stop_loss = entry_price + max_distance
                sl_reason += " (adjusted to max 3% risk)"

        return stop_loss, sl_reason

    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss: float,
                            direction: str,
                            levels: MarketLevels,
                            target_rr: float = 2.0) -> Tuple[float, str]:
        """
        Calculate take profit level

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction (LONG/SHORT)
            levels: Market levels
            target_rr: Target risk/reward ratio

        Returns:
            Tuple of (take_profit, tp_reason)
        """
        risk = abs(entry_price - stop_loss)
        target_profit = risk * target_rr

        if direction == "LONG":
            base_tp = entry_price + target_profit

            # Look for nearest resistance above base TP
            valid_resistances = [r for r in levels.resistance_levels if r > entry_price]

            if valid_resistances:
                nearest_resistance = valid_resistances[0]
                # Take profit at nearest resistance if it's close to target
                if abs(nearest_resistance - base_tp) < risk:
                    take_profit = nearest_resistance
                    tp_reason = f"Take profit at resistance level {nearest_resistance:.2f}"
                else:
                    take_profit = base_tp
                    tp_reason = f"Take profit at {target_rr}:1 R:R ratio"
            else:
                # Check Fibonacci levels
                if levels.fibonacci_levels:
                    fib_candidates = [v for k, v in levels.fibonacci_levels.items()
                                    if v > entry_price and abs(v - base_tp) < risk * 0.5]
                    if fib_candidates:
                        take_profit = min(fib_candidates)
                        tp_reason = f"Take profit at Fibonacci level"
                    else:
                        take_profit = base_tp
                        tp_reason = f"Take profit at {target_rr}:1 R:R ratio"
                else:
                    take_profit = base_tp
                    tp_reason = f"Take profit at {target_rr}:1 R:R ratio"

        else:  # SHORT
            base_tp = entry_price - target_profit

            # Look for nearest support below base TP
            valid_supports = [s for s in levels.support_levels if s < entry_price]

            if valid_supports:
                nearest_support = valid_supports[0]
                # Take profit at nearest support if it's close to target
                if abs(nearest_support - base_tp) < risk:
                    take_profit = nearest_support
                    tp_reason = f"Take profit at support level {nearest_support:.2f}"
                else:
                    take_profit = base_tp
                    tp_reason = f"Take profit at {target_rr}:1 R:R ratio"
            else:
                # Check Fibonacci levels
                if levels.fibonacci_levels:
                    fib_candidates = [v for k, v in levels.fibonacci_levels.items()
                                    if v < entry_price and abs(v - base_tp) < risk * 0.5]
                    if fib_candidates:
                        take_profit = max(fib_candidates)
                        tp_reason = f"Take profit at Fibonacci level"
                    else:
                        take_profit = base_tp
                        tp_reason = f"Take profit at {target_rr}:1 R:R ratio"
                else:
                    take_profit = base_tp
                    tp_reason = f"Take profit at {target_rr}:1 R:R ratio"

        return take_profit, tp_reason

    def calculate_confidence_score(self,
                                 pattern: PatternInfo,
                                 levels: MarketLevels,
                                 entry_price: float,
                                 stop_loss: float,
                                 take_profit: float,
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence score

        Args:
            pattern: Pattern information
            levels: Market levels
            entry_price: Entry price
            stop_loss: Stop loss
            take_profit: Take profit
            df: Price data

        Returns:
            Dictionary with confidence breakdown
        """
        scores = {}

        # Pattern quality score (30% weight)
        scores['pattern_quality'] = pattern.quality * 30

        # Historical pattern confidence (25% weight)
        scores['pattern_confidence'] = (pattern.confidence / 100) * 25

        # Risk/Reward ratio score (20% weight)
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio >= 3.0:
            rr_score = 20
        elif rr_ratio >= 2.5:
            rr_score = 17
        elif rr_ratio >= 2.0:
            rr_score = 14
        elif rr_ratio >= 1.5:
            rr_score = 10
        elif rr_ratio >= 1.0:
            rr_score = 5
        else:
            rr_score = 0

        scores['risk_reward'] = rr_score

        # Market structure alignment (15% weight)
        structure_score = 0
        current_price = levels.current_price

        # Check if entry is at logical level
        if pattern.direction == "LONG":
            nearest_support = levels.support_levels[0] if levels.support_levels else current_price - (2 * levels.atr)
            if abs(entry_price - nearest_support) < levels.atr:
                structure_score += 10
        else:
            nearest_resistance = levels.resistance_levels[0] if levels.resistance_levels else current_price + (2 * levels.atr)
            if abs(entry_price - nearest_resistance) < levels.atr:
                structure_score += 10

        # Check if R:R is reasonable
        if 1.5 <= rr_ratio <= 4.0:
            structure_score += 5

        scores['market_structure'] = structure_score

        # Volume confirmation (10% weight)
        volume_score = 0
        if 'volume_ratio' in df.columns:
            volume_ratio = df.iloc[-1]['volume_ratio']
            if volume_ratio >= 2.0:
                volume_score = 10
            elif volume_ratio >= 1.5:
                volume_score = 7
            elif volume_ratio >= 1.0:
                volume_score = 5
            else:
                volume_score = 2
        else:
            volume_score = 5  # Neutral score

        scores['volume_confirmation'] = volume_score

        # Calculate total confidence
        total_confidence = sum(scores.values())

        # Determine strength
        if total_confidence >= 80:
            strength = "Very Strong"
        elif total_confidence >= 70:
            strength = "Strong"
        elif total_confidence >= 60:
            strength = "Moderate"
        elif total_confidence >= 50:
            strength = "Weak"
        else:
            strength = "Very Weak"

        return {
            'total_confidence': round(total_confidence, 1),
            'strength': strength,
            'breakdown': scores,
            'risk_reward_ratio': round(rr_ratio, 2),
            'risk_pct': round((risk / current_price) * 100, 2),
            'reward_pct': round((reward / current_price) * 100, 2)
        }

    def generate_reasoning(self,
                         pattern: PatternInfo,
                         levels: MarketLevels,
                         entry_price: float,
                         stop_loss: float,
                         take_profit: float,
                         confidence: Dict[str, Any],
                         entry_reason: str,
                         sl_reason: str,
                         tp_reason: str,
                         df: pd.DataFrame) -> str:
        """
        Generate detailed reasoning for the trade setup

        Returns:
            Formatted reasoning string
        """
        reasoning_parts = []

        # Pattern analysis
        reasoning_parts.append(
            f"PATTERN ANALYSIS: {pattern.name.replace('_', ' ').title()} pattern detected "
            f"({pattern.confidence}% confidence, {pattern.quality*100:.0f}% quality)"
        )

        # Market context
        current_price = levels.current_price
        price_position = "above support" if pattern.direction == "LONG" else "below resistance"
        reasoning_parts.append(
            f"MARKET CONTEXT: Price currently at {current_price:.2f}, {price_position}, "
            f"ATR at {levels.atr:.2f}"
        )

        # Entry reasoning
        reasoning_parts.append(f"ENTRY STRATEGY: {entry_reason}")

        # Risk management
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_pct = (risk / current_price) * 100
        reward_pct = (reward / current_price) * 100

        reasoning_parts.append(
            f"RISK MANAGEMENT: {sl_reason} | {tp_reason} | "
            f"Risk: {risk_pct:.2f}%, Reward: {reward_pct:.2f}%, R:R = {confidence['risk_reward_ratio']}:1"
        )

        # Confidence breakdown
        breakdown = confidence['breakdown']
        reasoning_parts.append(
            f"CONFIDENCE BREAKDOWN: Pattern Quality {breakdown['pattern_quality']:.0f}pts, "
            f"Historical Success {breakdown['pattern_confidence']:.0f}pts, "
            f"Risk/Reward {breakdown['risk_reward']:.0f}pts, "
            f"Market Structure {breakdown['market_structure']:.0f}pts, "
            f"Volume Confirmation {breakdown['volume_confirmation']:.0f}pts"
        )

        # Additional factors
        if 'volume_ratio' in df.columns:
            volume_ratio = df.iloc[-1]['volume_ratio']
            reasoning_parts.append(f"VOLUME ANALYSIS: Current volume at {volume_ratio:.1f}x average")

        if 'rsi' in df.columns:
            rsi = df.iloc[-1]['rsi']
            rsi_state = "oversold" if rsi <= 30 else "overbought" if rsi >= 70 else "neutral"
            reasoning_parts.append(f"RSI STATUS: {rsi:.0f} ({rsi_state})")

        return " | ".join(reasoning_parts)

    def generate_trade_setup(self,
                           df: pd.DataFrame,
                           pattern: PatternInfo,
                           symbol: str = "XAUUSD",
                           timeframe: str = "M15") -> Dict[str, Any]:
        """
        Generate complete trade setup

        Args:
            df: DataFrame with OHLCV and indicators
            pattern: Pattern information
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Complete trade setup dictionary
        """
        print(f"🎯 Generating trade setup for {symbol} {timeframe}...")

        # Calculate market levels
        levels = self.calculate_market_levels(df)

        # Determine direction
        if pattern.direction == "bullish":
            direction = "LONG"
        elif pattern.direction == "bearish":
            direction = "SHORT"
        else:
            direction = "HOLD"

        if direction == "HOLD":
            return {
                "direction": "HOLD",
                "recommendation": "No clear trade setup identified",
                "confidence": 0,
                "reasoning": "Pattern direction is neutral - wait for clearer setup"
            }

        # Calculate entry, stop loss, and take profit
        entry_price, entry_reason = self.calculate_entry_price(pattern, levels, df)
        stop_loss, sl_reason = self.calculate_stop_loss(entry_price, direction, levels, df)
        take_profit, tp_reason = self.calculate_take_profit(entry_price, stop_loss, direction, levels)

        # Calculate confidence
        confidence = self.calculate_confidence_score(pattern, levels, entry_price, stop_loss, take_profit, df)

        # Generate reasoning
        reasoning = self.generate_reasoning(
            pattern, levels, entry_price, stop_loss, take_profit, confidence,
            entry_reason, sl_reason, tp_reason, df
        )

        # Create trade setup
        trade_setup = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "entry_price": round(entry_price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "confidence": confidence['total_confidence'],
            "strength": confidence['strength'],
            "risk_reward_ratio": confidence['risk_reward_ratio'],
            "risk_pct": confidence['risk_pct'],
            "reward_pct": confidence['reward_pct'],
            "pattern": {
                "name": pattern.name,
                "type": pattern.type,
                "direction": pattern.direction,
                "quality": pattern.quality,
                "confidence": pattern.confidence
            },
            "market_levels": {
                "current_price": levels.current_price,
                "atr": levels.atr,
                "support_levels": levels.support_levels[:3],
                "resistance_levels": levels.resistance_levels[:3],
                "nearest_support": levels.support_levels[0] if levels.support_levels else None,
                "nearest_resistance": levels.resistance_levels[0] if levels.resistance_levels else None
            },
            "reasoning": reasoning,
            "entry_reason": entry_reason,
            "stop_loss_reason": sl_reason,
            "take_profit_reason": tp_reason,
            "confidence_breakdown": confidence['breakdown']
        }

        # Only recommend if confidence meets minimum
        if confidence['total_confidence'] < 60:
            trade_setup["direction"] = "HOLD"
            trade_setup["recommendation"] = f"Low confidence ({confidence['total_confidence']}%) - wait for better setup"
        else:
            trade_setup["recommendation"] = f"{direction} setup with {confidence['strength']} confidence"

        print(f"✅ Trade setup generated: {direction} @ {entry_price:.2f} (confidence: {confidence['total_confidence']}%)")
        return trade_setup


def main():
    """Example usage of the Trade Recommendation Engine"""

    # Create sample data
    dates = pd.date_range(start='2025-01-01', periods=200, freq='15min')
    np.random.seed(42)

    # Generate realistic price data
    price = 2000 + np.cumsum(np.random.randn(200) * 2)
    high = price + np.random.uniform(0, 5, 200)
    low = price - np.random.uniform(0, 5, 200)
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    volume = np.random.uniform(1000, 5000, 200)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    })

    # Add basic indicators
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['rsi'] = 50 + np.random.uniform(-20, 20, 200)

    # Create pattern
    pattern = PatternInfo(
        name="bullish_engulfing",
        type="reversal",
        direction="bullish",
        quality=0.85,
        confidence=80
    )

    # Generate trade setup
    engine = TradeRecommendationEngine()
    setup = engine.generate_trade_setup(df, pattern, "XAUUSD", "M15")

    print("\n" + "=" * 70)
    print("TRADE RECOMMENDATION ENGINE - EXAMPLE OUTPUT")
    print("=" * 70)
    print(json.dumps(setup, indent=2))


if __name__ == "__main__":
    main()