#!/usr/bin/env python3
"""
Trading Data Processor for RAG System
Converts MT5/Sierra Chart CSV into narrative format for vector database

Based on workflow_resume.md comprehensive implementation
Enhanced version with full pattern detection and narrative generation
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime
from tqdm import tqdm
import json
import sys
import os

class TradingDataProcessor:
    """
    Process trading data into RAG-friendly format
    Output: Narrative text documents with metadata
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.timeframe = None
        self.timeframe_minutes = None
        self.timeframe_label = None

    def _detect_timeframe(self):
        """Detect timeframe from filename or data"""
        import re

        # Timeframe mappings - ordered from longest to shortest to avoid substring matches
        timeframe_map = [
            ('M15', 15, '15-minute'),
            ('M30', 30, '30-minute'),
            ('M1', 1, '1-minute'),
            ('M5', 5, '5-minute'),
            ('H1', 60, '1-hour'),
            ('H4', 240, '4-hour'),
            ('D1', 1440, 'Daily'),
            ('W1', 10080, 'Weekly')
        ]

        # Try to extract from filename using word boundaries
        filename = os.path.basename(self.csv_path)
        for tf_code, minutes, label in timeframe_map:
            # Use word boundary to ensure exact match (e.g., M15 won't match M1)
            pattern = f'PERIOD_{tf_code}[_\\.]'
            if re.search(pattern, filename):
                self.timeframe = tf_code
                self.timeframe_minutes = minutes
                self.timeframe_label = label
                return

        # Try to extract from data if available
        if self.df is not None and 'TimeFrame' in self.df.columns:
            tf_value = self.df['TimeFrame'].iloc[0]
            for tf_code, minutes, label in timeframe_map:
                if tf_code in tf_value:
                    self.timeframe = tf_code
                    self.timeframe_minutes = minutes
                    self.timeframe_label = label
                    return

        # Default to M15
        self.timeframe = 'M15'
        self.timeframe_minutes = 15
        self.timeframe_label = '15-minute'
        print(f"Warning: Could not detect timeframe, defaulting to M15")

    def load_csv(self):
        """Load and standardize CSV format"""
        print(f"Loading: {self.csv_path}")

        # Try different encodings
        encodings = ['utf-16-le', 'utf-16', 'utf-8', 'latin1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except Exception as e:
                continue

        if df is None:
            raise ValueError(f"Could not read CSV file with any supported encoding")

        # Standardize column names
        column_mapping = {
            'DateTime': 'timestamp',
            'Timestamp': 'timestamp',
            'time': 'timestamp',
            'date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume',
            'volume': 'tick_volume',
            'vol': 'tick_volume'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items()
                                if k in df.columns})

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Keep only required columns
        df = df[required_cols]

        print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

        self.df = df

        # Detect timeframe from filename or data
        self._detect_timeframe()
        print(f"Detected timeframe: {self.timeframe_label} ({self.timeframe_minutes} minutes)")

        return df

    def calculate_indicators(self):
        """Calculate technical indicators"""
        print("Calculating indicators...")

        close = self.df['close'].values
        high = self.df['high'].values
        low = self.df['low'].values

        # Core indicators
        self.df['rsi'] = talib.RSI(close, timeperiod=14)
        self.df['ema9'] = talib.EMA(close, timeperiod=9)
        self.df['ema20'] = talib.EMA(close, timeperiod=20)
        self.df['ema50'] = talib.EMA(close, timeperiod=50)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower

        # ATR
        self.df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # Volume
        self.df['vol_sma'] = self.df['tick_volume'].rolling(20).mean()
        self.df['vol_ratio'] = self.df['tick_volume'] / self.df['vol_sma']

        # MACD
        macd, signal, hist = talib.MACD(close)
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist

        # Drop NaN rows
        self.df = self.df.dropna().reset_index(drop=True)

        print(f"Indicators calculated. {len(self.df)} bars ready.")
        return self.df

    def detect_patterns(self):
        """Detect candlestick patterns and generate narrative documents"""
        print("Detecting patterns...")

        patterns = []
        start_idx = 100  # Need history for context

        for i in tqdm(range(start_idx, len(self.df) - 20)):

            # Pattern 1: Bullish Engulfing
            if self._is_bullish_engulfing(i):
                doc = self._create_pattern_document(i, "Bullish Engulfing")
                patterns.append(doc)

            # Pattern 2: Bearish Engulfing
            if self._is_bearish_engulfing(i):
                doc = self._create_pattern_document(i, "Bearish Engulfing")
                patterns.append(doc)

            # Pattern 3: RSI Divergence
            if self._is_rsi_divergence(i, 'bullish'):
                doc = self._create_pattern_document(i, "RSI Bullish Divergence")
                patterns.append(doc)

            if self._is_rsi_divergence(i, 'bearish'):
                doc = self._create_pattern_document(i, "RSI Bearish Divergence")
                patterns.append(doc)

            # Pattern 4: Breakout
            if self._is_breakout(i):
                doc = self._create_pattern_document(i, "Breakout")
                patterns.append(doc)

            # Pattern 5: Support/Resistance Bounce
            if self._is_bounce(i, 'support'):
                doc = self._create_pattern_document(i, "Support Bounce")
                patterns.append(doc)

            if self._is_bounce(i, 'resistance'):
                doc = self._create_pattern_document(i, "Resistance Rejection")
                patterns.append(doc)

        print(f"Found {len(patterns)} patterns")
        return patterns

    # ==================== PATTERN DETECTION ====================

    def _is_bullish_engulfing(self, idx):
        """Bullish engulfing pattern"""
        curr = self.df.iloc[idx]
        prev = self.df.iloc[idx-1]

        bearish_prev = prev['close'] < prev['open']
        bullish_curr = curr['close'] > curr['open']
        engulfs = (curr['open'] <= prev['close'] and
                   curr['close'] >= prev['open'])

        return bearish_prev and bullish_curr and engulfs

    def _is_bearish_engulfing(self, idx):
        """Bearish engulfing pattern"""
        curr = self.df.iloc[idx]
        prev = self.df.iloc[idx-1]

        bullish_prev = prev['close'] > prev['open']
        bearish_curr = curr['close'] < curr['open']
        engulfs = (curr['open'] >= prev['close'] and
                   curr['close'] <= prev['open'])

        return bullish_prev and bearish_curr and engulfs

    def _is_rsi_divergence(self, idx, div_type='bullish'):
        """RSI divergence detection"""
        lookback = 20
        if idx < lookback:
            return False

        curr = self.df.iloc[idx]
        window = self.df.iloc[idx-lookback:idx]

        if div_type == 'bullish':
            # Price lower low, RSI higher low
            price_ll = curr['low'] < window['low'].min()
            rsi_hl = curr['rsi'] > window['rsi'].min()
            return price_ll and rsi_hl and curr['rsi'] < 40

        elif div_type == 'bearish':
            # Price higher high, RSI lower high
            price_hh = curr['high'] > window['high'].max()
            rsi_lh = curr['rsi'] < window['rsi'].max()
            return price_hh and rsi_lh and curr['rsi'] > 60

        return False

    def _is_breakout(self, idx):
        """Breakout pattern"""
        lookback = 20
        curr = self.df.iloc[idx]
        window = self.df.iloc[idx-lookback:idx]

        recent_high = window['high'].max()
        breakout = curr['close'] > recent_high
        volume_confirm = curr['vol_ratio'] > 1.5
        momentum = curr['rsi'] > 50

        return breakout and volume_confirm and momentum

    def _is_bounce(self, idx, level_type='support'):
        """Support/Resistance bounce"""
        lookback = 100
        if idx < lookback:
            return False

        curr = self.df.iloc[idx]
        window = self.df.iloc[idx-lookback:idx]
        tolerance = curr['atr'] * 0.5

        if level_type == 'support':
            # Find swing lows
            swing_lows = window[window['low'] == window['low'].rolling(5, center=True).min()]['low'].values
            if len(swing_lows) == 0:
                return False

            # Check if current low near any swing low
            near_support = any(abs(curr['low'] - level) < tolerance for level in swing_lows)
            bullish_close = curr['close'] > curr['open']

            return near_support and bullish_close

        elif level_type == 'resistance':
            # Find swing highs
            swing_highs = window[window['high'] == window['high'].rolling(5, center=True).max()]['high'].values
            if len(swing_highs) == 0:
                return False

            # Check if current high near any swing high
            near_resistance = any(abs(curr['high'] - level) < tolerance for level in swing_highs)
            bearish_close = curr['close'] < curr['open']

            return near_resistance and bearish_close

        return False

    # ==================== DOCUMENT CREATION ====================

    def _create_pattern_document(self, idx, pattern_name):
        """
        Create narrative document for pattern
        This is the key format for RAG
        """
        row = self.df.iloc[idx]
        prev_10 = self.df.iloc[idx-10:idx]

        # Calculate outcome (look forward 20 bars)
        outcome = self._calculate_outcome(idx)

        # Get context
        trend = self._get_trend(row)
        rsi_state = self._get_rsi_state(row)
        bb_position = self._get_bb_position(row)
        volume_state = self._get_volume_state(row)

        # Build narrative text
        text = f"""Pattern: {pattern_name}
Symbol: XAUUSD
Timeframe: {self.timeframe_label}
Date: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}

=== SETUP DESCRIPTION ===
A {pattern_name.lower()} formed at {row['close']:.2f} during {self._get_session(row['timestamp'])}.

Price Action:
- Current candle: Open {row['open']:.2f}, High {row['high']:.2f}, Low {row['low']:.2f}, Close {row['close']:.2f}
- Candle body: {abs(row['close'] - row['open']):.2f} points
- Previous 10-bar range: {prev_10['low'].min():.2f} to {prev_10['high'].max():.2f}
- Price momentum: {self._get_momentum_description(prev_10, row)}

=== TECHNICAL CONTEXT ===
Trend: {trend}
- EMA alignment: EMA9({row['ema9']:.2f}) {'>' if row['ema9'] > row['ema20'] else '<'} EMA20({row['ema20']:.2f}) {'>' if row['ema20'] > row['ema50'] else '<'} EMA50({row['ema50']:.2f})
- Price vs EMA9: {row['close'] - row['ema9']:+.2f} points

RSI: {row['rsi']:.1f} ({rsi_state})
- RSI momentum: {self._get_rsi_momentum(idx)}

Bollinger Bands: Price at {bb_position}
- BB width: {row['bb_upper'] - row['bb_lower']:.2f} points ({self._get_volatility_state(row)})

Volume: {volume_state}
- Volume ratio: {row['vol_ratio']:.2f}x average
- Volume interpretation: {self._get_volume_interpretation(row)}

MACD: {self._get_macd_state(row)}

=== TRADING SETUP ===
Entry: {row['close']:.2f}
Stop Loss: {self._calculate_stop_loss(row, pattern_name):.2f}
Take Profit 1: {self._calculate_tp1(row, pattern_name):.2f}
Take Profit 2: {self._calculate_tp2(row, pattern_name):.2f}
Risk/Reward: {self._calculate_risk_reward(row, pattern_name):.1f}:1
Position size guideline: Use {self._get_position_sizing(row)} of account risk

=== OUTCOME ===
{outcome['description']}
Result: {outcome['result']}
Duration: {outcome['bars']} bars ({outcome['time']})
P&L: {outcome['pnl']:+.2f} points
Max Favorable Excursion: {outcome['mfe']:+.2f} points
Max Adverse Excursion: {outcome['mae']:+.2f} points

=== LESSONS LEARNED ===
{self._generate_lesson(pattern_name, outcome, row)}

=== SIMILAR CONDITIONS ===
- Market regime: {self._get_market_regime(idx)}
- Time of day: {self._get_session(row['timestamp'])}
- Day of week: {row['timestamp'].strftime('%A')}
- Historical win rate for this pattern in similar conditions: {self._estimate_win_rate(pattern_name, row)}%
"""

        # Metadata for filtering
        metadata = {
            'pattern': pattern_name,
            'timestamp': row['timestamp'].isoformat(),
            'date': row['timestamp'].strftime('%Y-%m-%d'),
            'time': row['timestamp'].strftime('%H:%M'),
            'hour': row['timestamp'].hour,
            'day_of_week': row['timestamp'].strftime('%A'),
            'session': self._get_session(row['timestamp']),
            'symbol': 'XAUUSD',
            'timeframe': self.timeframe,
            'entry': float(row['close']),
            'stop': float(self._calculate_stop_loss(row, pattern_name)),
            'target': float(self._calculate_tp1(row, pattern_name)),
            'rsi': float(row['rsi']),
            'trend': trend,
            'outcome': outcome['result'],
            'pnl': float(outcome['pnl']),
            'duration_bars': int(outcome['bars']),
            'risk_reward': float(self._calculate_risk_reward(row, pattern_name)),
            'volume_ratio': float(row['vol_ratio']),
            'bb_position': bb_position,
            'market_regime': self._get_market_regime(idx)
        }

        return {
            'text': text,
            'metadata': metadata
        }

    # ==================== HELPER FUNCTIONS ====================

    def _calculate_outcome(self, idx, lookforward=20):
        """Calculate trade outcome"""
        entry_row = self.df.iloc[idx]
        entry = entry_row['close']
        atr = entry_row['atr']

        # Determine direction based on pattern context
        if entry_row['rsi'] < 50 or entry_row['close'] > entry_row['open']:
            direction = 'long'
            target = entry + (atr * 2)
            stop = entry - atr
        else:
            direction = 'short'
            target = entry - (atr * 2)
            stop = entry + atr

        future = self.df.iloc[idx+1:idx+lookforward+1]

        if len(future) == 0:
            return {
                'result': 'unknown',
                'pnl': 0,
                'bars': 0,
                'time': '0 min',
                'mfe': 0,
                'mae': 0,
                'description': 'Insufficient future data to determine outcome'
            }

        if direction == 'long':
            mfe = future['high'].max() - entry
            mae = future['low'].min() - entry

            target_hit = future[future['high'] >= target]
            stop_hit = future[future['low'] <= stop]

            if len(target_hit) > 0 and (len(stop_hit) == 0 or target_hit.index[0] < stop_hit.index[0]):
                bars = target_hit.index[0] - idx
                return {
                    'result': 'WIN',
                    'pnl': target - entry,
                    'bars': bars,
                    'time': f'{bars * self.timeframe_minutes} min',
                    'mfe': mfe,
                    'mae': mae,
                    'description': f'Target hit successfully. Price reached {target:.2f} after {bars} bars.'
                }
            elif len(stop_hit) > 0:
                bars = stop_hit.index[0] - idx
                return {
                    'result': 'LOSS',
                    'pnl': stop - entry,
                    'bars': bars,
                    'time': f'{bars * self.timeframe_minutes} min',
                    'mfe': mfe,
                    'mae': mae,
                    'description': f'Stop loss hit. Price reached {stop:.2f} after {bars} bars.'
                }

        return {
            'result': 'NEUTRAL',
            'pnl': future.iloc[-1]['close'] - entry,
            'bars': lookforward,
            'time': f'{lookforward * self.timeframe_minutes} min',
            'mfe': future['high'].max() - entry if direction == 'long' else entry - future['low'].min(),
            'mae': future['low'].min() - entry if direction == 'long' else entry - future['high'].max(),
            'description': 'Neither target nor stop was hit within observation period.'
        }

    def _get_trend(self, row):
        """Determine trend"""
        if row['ema9'] > row['ema20'] > row['ema50']:
            return "Strong Bullish"
        elif row['ema9'] > row['ema20']:
            return "Bullish"
        elif row['ema9'] < row['ema20'] < row['ema50']:
            return "Strong Bearish"
        elif row['ema9'] < row['ema20']:
            return "Bearish"
        else:
            return "Neutral/Choppy"

    def _get_rsi_state(self, row):
        """RSI state description"""
        rsi = row['rsi']
        if rsi > 80:
            return "Extremely Overbought"
        elif rsi > 70:
            return "Overbought"
        elif rsi > 60:
            return "Bullish"
        elif rsi > 40:
            return "Neutral"
        elif rsi > 30:
            return "Bearish"
        elif rsi > 20:
            return "Oversold"
        else:
            return "Extremely Oversold"

    def _get_bb_position(self, row):
        """Bollinger Band position"""
        bb_range = row['bb_upper'] - row['bb_lower']
        if bb_range == 0:
            return "middle"

        position = (row['close'] - row['bb_lower']) / bb_range

        if position > 1.0:
            return "above upper band (extreme overbought)"
        elif position > 0.8:
            return "upper band"
        elif position > 0.6:
            return "upper third"
        elif position > 0.4:
            return "middle"
        elif position > 0.2:
            return "lower third"
        elif position > 0:
            return "lower band"
        else:
            return "below lower band (extreme oversold)"

    def _get_volume_state(self, row):
        """Volume description"""
        ratio = row['vol_ratio']
        if ratio > 2.5:
            return "Extremely High (climax)"
        elif ratio > 2.0:
            return "Very High"
        elif ratio > 1.5:
            return "High"
        elif ratio > 0.8:
            return "Average"
        elif ratio > 0.5:
            return "Below Average"
        else:
            return "Very Low (low conviction)"

    def _get_session(self, timestamp):
        """Trading session"""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return "Asia"
        elif 8 <= hour < 16:
            return "Europe"
        else:
            return "New York"

    def _get_momentum_description(self, prev_bars, current):
        """Price momentum description"""
        prev_avg = prev_bars['close'].mean()
        change = ((current['close'] - prev_avg) / prev_avg) * 100

        if change > 0.5:
            return f"Strong bullish momentum (+{change:.2f}%)"
        elif change > 0.2:
            return f"Bullish momentum (+{change:.2f}%)"
        elif change > -0.2:
            return "Neutral momentum"
        elif change > -0.5:
            return f"Bearish momentum ({change:.2f}%)"
        else:
            return f"Strong bearish momentum ({change:.2f}%)"

    def _get_rsi_momentum(self, idx):
        """RSI momentum"""
        if idx < 5:
            return "insufficient data"

        curr_rsi = self.df.iloc[idx]['rsi']
        prev_rsi = self.df.iloc[idx-5]['rsi']
        diff = curr_rsi - prev_rsi

        if diff > 10:
            return "strongly rising"
        elif diff > 5:
            return "rising"
        elif diff > -5:
            return "stable"
        elif diff > -10:
            return "falling"
        else:
            return "strongly falling"

    def _get_volatility_state(self, row):
        """Volatility state"""
        bb_width_pct = ((row['bb_upper'] - row['bb_lower']) / row['close']) * 100

        if bb_width_pct > 3.0:
            return "very high volatility - trending conditions"
        elif bb_width_pct > 2.0:
            return "high volatility"
        elif bb_width_pct > 1.0:
            return "normal volatility"
        else:
            return "low volatility - potential breakout pending"

    def _get_volume_interpretation(self, row):
        """Volume interpretation"""
        if row['vol_ratio'] > 2.0:
            if row['close'] > row['open']:
                return "Strong buying pressure with high conviction"
            else:
                return "Strong selling pressure with high conviction"
        elif row['vol_ratio'] < 0.7:
            return "Low conviction move, be cautious"
        else:
            return "Normal market participation"

    def _get_macd_state(self, row):
        """MACD state"""
        if pd.isna(row['macd']) or pd.isna(row['macd_signal']):
            return "MACD data not available"

        if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
            return "Bullish (above signal line)"
        elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
            return "Bearish (below signal line)"
        else:
            return "Neutral (near signal line)"

    def _calculate_stop_loss(self, row, pattern):
        """Calculate stop loss"""
        atr = row['atr']

        if 'Bullish' in pattern or 'Support' in pattern:
            return row['close'] - atr
        else:
            return row['close'] + atr

    def _calculate_tp1(self, row, pattern):
        """Calculate first target"""
        atr = row['atr']

        if 'Bullish' in pattern or 'Support' in pattern:
            return row['close'] + (atr * 2)
        else:
            return row['close'] - (atr * 2)

    def _calculate_tp2(self, row, pattern):
        """Calculate second target"""
        atr = row['atr']

        if 'Bullish' in pattern or 'Support' in pattern:
            return row['close'] + (atr * 3)
        else:
            return row['close'] - (atr * 3)

    def _calculate_risk_reward(self, row, pattern):
        """Calculate R:R ratio"""
        return 2.0  # Default 1:2

    def _get_position_sizing(self, row):
        """Position sizing recommendation"""
        if row['vol_ratio'] > 2.0:
            return "0.5-1%"  # High volatility
        else:
            return "1-2%"    # Normal

    def _generate_lesson(self, pattern, outcome, row):
        """Generate lesson learned"""
        if outcome['result'] == 'WIN':
            return f"Pattern worked as expected. Key factors: {self._get_trend(row)} trend, RSI at {row['rsi']:.1f}, volume {row['vol_ratio']:.1f}x. This combination has high probability."
        elif outcome['result'] == 'LOSS':
            return f"Pattern failed. Possible reasons: {self._analyze_failure(row, outcome)}. Avoid similar setups."
        else:
            return "Outcome inconclusive. Consider waiting for stronger confirmation signals."

    def _analyze_failure(self, row, outcome):
        """Analyze why pattern failed"""
        reasons = []

        if row['vol_ratio'] < 0.8:
            reasons.append("weak volume confirmation")

        if abs(row['rsi'] - 50) < 10:
            reasons.append("RSI in neutral zone (no momentum)")

        if self._get_trend(row) == "Neutral/Choppy":
            reasons.append("choppy market conditions")

        if outcome['mae'] < -row['atr'] * 0.5:
            reasons.append("immediate adverse move (false signal)")

        return ", ".join(reasons) if reasons else "unclear - market conditions changed"

    def _get_market_regime(self, idx):
        """Determine market regime"""
        lookback = 50
        if idx < lookback:
            return "Unknown"

        window = self.df.iloc[idx-lookback:idx]

        # Calculate metrics
        avg_atr = window['atr'].mean()
        current_atr = self.df.iloc[idx]['atr']
        volatility_ratio = current_atr / avg_atr

        # Trend strength
        ema_spread = abs(window['ema9'] - window['ema50']).mean()

        if volatility_ratio > 1.5 and ema_spread > avg_atr * 2:
            return "Strong Trending High Volatility"
        elif volatility_ratio > 1.5:
            return "Range-bound High Volatility"
        elif ema_spread > avg_atr * 2:
            return "Strong Trending Low Volatility"
        else:
            return "Range-bound Low Volatility"

    def _estimate_win_rate(self, pattern, row):
        """Estimate historical win rate (simplified)"""
        # In production, this would query historical data
        # For now, return heuristic based on conditions

        base_rate = 50

        # Adjust for trend alignment
        trend = self._get_trend(row)
        if 'Bullish' in pattern and 'Bullish' in trend:
            base_rate += 10
        elif 'Bearish' in pattern and 'Bearish' in trend:
            base_rate += 10

        # Adjust for volume
        if row['vol_ratio'] > 1.5:
            base_rate += 5
        elif row['vol_ratio'] < 0.8:
            base_rate -= 10

        # Adjust for RSI
        if 'Bullish' in pattern and row['rsi'] < 40:
            base_rate += 5
        elif 'Bearish' in pattern and row['rsi'] > 60:
            base_rate += 5

        return max(30, min(75, base_rate))

    # ==================== SUPPORT/RESISTANCE LEVELS ====================

    def find_support_resistance(self):
        """Find key support and resistance levels"""
        print("Finding support/resistance levels...")

        levels = []

        # Find swing highs and lows
        for i in range(20, len(self.df)-20):
            row = self.df.iloc[i]

            # Swing High (resistance)
            if (row['high'] == self.df.iloc[i-10:i+11]['high'].max() and
                row['high'] > self.df.iloc[i-20:i-10]['high'].max()):

                text = f"""Support/Resistance Level: {row['high']:.2f}
Type: Resistance (Swing High)
Date formed: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}

Level Details:
- Price: {row['high']:.2f}
- Formation: Swing high with {10}-bar confirmation
- Context: Formed during {self._get_session(row['timestamp'])} session
- Market regime: {self._get_market_regime(i)}
- Volume at formation: {row['vol_ratio']:.2f}x average

Historical Behavior:
- This level formed when RSI was at {row['rsi']:.1f}
- Trend at formation: {self._get_trend(row)}
- Expected reaction: Price likely to face resistance here

Trading Strategy:
- Look for rejection signals (bearish candles, volume increase)
- Consider shorts if price reaches this level with confirmation
- Place stops above this level if going short
- Alternatively, wait for breakout above for long entries
"""

                levels.append({
                    'text': text,
                    'metadata': {
                        'type': 'resistance',
                        'level': float(row['high']),
                        'date': row['timestamp'].isoformat(),
                        'symbol': 'XAUUSD',
                        'timeframe': self.timeframe,
                        'strength': 'moderate'
                    }
                })

            # Swing Low (support)
            if (row['low'] == self.df.iloc[i-10:i+11]['low'].min() and
                row['low'] < self.df.iloc[i-20:i-10]['low'].min()):

                text = f"""Support/Resistance Level: {row['low']:.2f}
Type: Support (Swing Low)
Date formed: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}

Level Details:
- Price: {row['low']:.2f}
- Formation: Swing low with {10}-bar confirmation
- Context: Formed during {self._get_session(row['timestamp'])} session
- Market regime: {self._get_market_regime(i)}
- Volume at formation: {row['vol_ratio']:.2f}x average

Historical Behavior:
- This level formed when RSI was at {row['rsi']:.1f}
- Trend at formation: {self._get_trend(row)}
- Expected reaction: Price likely to find support here

Trading Strategy:
- Look for bounce signals (bullish candles, volume increase)
- Consider longs if price reaches this level with confirmation
- Place stops below this level if going long
- Alternatively, wait for breakdown below for short entries
"""

                levels.append({
                    'text': text,
                    'metadata': {
                        'type': 'support',
                        'level': float(row['low']),
                        'date': row['timestamp'].isoformat(),
                        'symbol': 'XAUUSD',
                        'timeframe': self.timeframe,
                        'strength': 'moderate'
                    }
                })

        print(f"Found {len(levels)} key levels")
        return levels

    # ==================== MAIN PROCESSING ====================

    def process_all(self):
        """Run complete pipeline"""
        print("\n" + "="*60)
        print("TRADING DATA PROCESSING PIPELINE")
        print("="*60 + "\n")

        # Step 1: Load
        self.load_csv()

        # Step 2: Calculate indicators
        self.calculate_indicators()

        # Step 3: Detect patterns
        patterns = self.detect_patterns()

        # Step 4: Find levels
        levels = self.find_support_resistance()

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Patterns: {len(patterns)}")
        print(f"Levels: {len(levels)}")
        print(f"Total documents: {len(patterns) + len(levels)}")

        return {
            'patterns': patterns,
            'levels': levels
        }

# ==================== USAGE ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_processor.py <csv_file>")
        print("Example: python data_processor.py ../data/XAUUSD-2025.10.21.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Process
    processor = TradingDataProcessor(csv_file)
    results = processor.process_all()

    # Save output
    output_file = csv_file.replace('.csv', '_processed.json')

    with open(output_file, 'w') as f:
        json.dump({
            'patterns': results['patterns'],
            'levels': results['levels']
        }, f, indent=2)

    print(f"\nOutput saved to: {output_file}")
    print("\nSample pattern document:")
    print("-" * 60)
    if results['patterns']:
        print(results['patterns'][0]['text'][:500] + "...")
