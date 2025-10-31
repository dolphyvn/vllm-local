#!/usr/bin/env python3
"""
MT5 to Structured JSON Converter
Optimized for small LLM + ChromaDB RAG system

Converts raw MT5 CSV to clean structured JSON with:
- OHLCV data
- Technical indicators
- Summary field (optimized for embeddings)
- Proper metadata for filtering

Usage:
    python scripts/mt5_to_structured_json.py \
        --input data/XAUUSD_PERIOD_M15_200.csv \
        --output data/structured/XAUUSD_M15_structured.json \
        --symbol XAUUSD --timeframe M15
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class MT5ToStructuredJSON:
    """Convert MT5 CSV to structured JSON format"""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M15"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df: Optional[pd.DataFrame] = None

    def load_csv(self, csv_path: str) -> bool:
        """
        Load CSV with robust encoding detection

        Args:
            csv_path: Path to MT5 CSV file

        Returns:
            True if successful, False otherwise
        """
        print(f"📂 Loading: {csv_path}")

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

        # Check if we have only one column (corrupted format)
        if len(self.df.columns) == 1:
            print(f"⚠️  Single column detected, likely encoding issue. Trying alternative parsing...")
            # This is a corrupted file - skip it
            print(f"   Current column: {repr(self.df.columns[0][:50])}...")
            return False

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
        """Calculate technical indicators"""
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
            self.df['rsi'] = self.df['rsi'].fillna(50)  # Neutral default

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

            # VWAP (Volume Weighted Average Price) - Cumulative
            typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
            self.df['vwap'] = (typical_price * pd.Series(volume)).cumsum() / pd.Series(volume).cumsum()

            # === MARKET PROFILE INDICATORS ===

            # Calculate session for each candle
            self.df['session'] = self.df['timestamp'].apply(self._get_session)

            # Session-based VWAP (resets per session)
            self.df['session_vwap'] = self._calculate_session_vwap(typical_price, pd.Series(volume))

            # Market Profile: POC, VAH, VAL per session
            market_profile = self._calculate_market_profile()
            self.df['poc'] = market_profile['poc']
            self.df['vah'] = market_profile['vah']
            self.df['val'] = market_profile['val']
            self.df['value_area_pct'] = market_profile['va_pct']  # % of price within value area

            # Initial Balance (first hour range of session)
            ib_data = self._calculate_initial_balance()
            self.df['ib_high'] = ib_data['ib_high']
            self.df['ib_low'] = ib_data['ib_low']
            self.df['ib_range'] = ib_data['ib_range']

            # === PRICE ACTION ===

            # Price changes
            self.df['price_change_pct'] = pd.Series(close).pct_change() * 100
            self.df['price_change_abs'] = pd.Series(close).diff()

            # Candle body and wicks
            self.df['body_size'] = np.abs(self.df['close'] - self.df['open'])
            self.df['upper_wick'] = pd.Series(high) - pd.DataFrame({'open': self.df['open'], 'close': self.df['close']}).max(axis=1)
            self.df['lower_wick'] = pd.DataFrame({'open': self.df['open'], 'close': self.df['close']}).min(axis=1) - pd.Series(low)
            self.df['hl_range'] = pd.Series(high) - pd.Series(low)

            # Fill any NaN values
            self.df = self.df.ffill().bfill()

            print(f"✅ Calculated {len([c for c in self.df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
            return True

        except Exception as e:
            print(f"❌ Indicator calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_trend_description(self, row: pd.Series) -> str:
        """Get simple trend description"""
        if row['ema_9'] > row['ema_20'] > row['ema_50']:
            return "strong_bullish"
        elif row['ema_9'] > row['ema_20']:
            return "bullish"
        elif row['ema_9'] < row['ema_20'] < row['ema_50']:
            return "strong_bearish"
        elif row['ema_9'] < row['ema_20']:
            return "bearish"
        else:
            return "neutral"

    def _get_rsi_state(self, rsi: float) -> str:
        """Get RSI state"""
        if rsi >= 70:
            return "overbought"
        elif rsi <= 30:
            return "oversold"
        elif rsi > 55:
            return "bullish"
        elif rsi < 45:
            return "bearish"
        else:
            return "neutral"

    def _get_volume_state(self, vol_ratio: float) -> str:
        """Get volume state"""
        if vol_ratio >= 2.0:
            return "very_high"
        elif vol_ratio >= 1.5:
            return "high"
        elif vol_ratio >= 0.8:
            return "normal"
        else:
            return "low"

    def _get_session(self, timestamp: datetime) -> str:
        """Get trading session"""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 17:
            return "newyork"
        elif 17 <= hour < 22:
            return "us_late"
        else:
            return "pacific"

    def _calculate_session_vwap(self, typical_price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate VWAP that resets at each trading session start

        Args:
            typical_price: (High + Low + Close) / 3
            volume: Volume series

        Returns:
            Session-based VWAP series
        """
        session_vwap = pd.Series(index=self.df.index, dtype=float)

        # Group by date and session
        self.df['date'] = self.df['timestamp'].dt.date

        for (date, session), group in self.df.groupby(['date', 'session']):
            indices = group.index

            # Calculate cumulative VWAP within session
            tp_session = typical_price.loc[indices]
            vol_session = volume.loc[indices]

            cumulative_tp_vol = (tp_session * vol_session).cumsum()
            cumulative_vol = vol_session.cumsum()

            session_vwap.loc[indices] = cumulative_tp_vol / cumulative_vol.replace(0, 1)

        return session_vwap

    def _calculate_market_profile(self) -> Dict[str, pd.Series]:
        """
        Calculate Market Profile indicators: POC, VAH, VAL per session

        Returns:
            Dictionary with poc, vah, val, and value area percentage series
        """
        poc = pd.Series(index=self.df.index, dtype=float)
        vah = pd.Series(index=self.df.index, dtype=float)
        val = pd.Series(index=self.df.index, dtype=float)
        va_pct = pd.Series(index=self.df.index, dtype=float)

        # Process each session separately
        for (date, session), group in self.df.groupby(['date', 'session']):
            indices = group.index

            if len(indices) == 0:
                continue

            # Get price and volume data for this session
            session_high = group['high'].values
            session_low = group['low'].values
            session_close = group['close'].values
            session_volume = group['volume'].values

            # Create price levels (tick size = 0.01 for most instruments)
            price_min = session_low.min()
            price_max = session_high.max()
            tick_size = 0.01

            # Create volume profile bins
            num_bins = min(int((price_max - price_min) / tick_size) + 1, 500)  # Cap at 500 bins
            if num_bins < 5:
                num_bins = 5

            price_levels = np.linspace(price_min, price_max, num_bins)
            volume_at_price = np.zeros(len(price_levels))

            # Distribute volume across price levels
            for i in range(len(indices)):
                low_price = session_low[i]
                high_price = session_high[i]
                vol = session_volume[i]

                # Find which bins this candle covers
                bins_covered = np.where((price_levels >= low_price) & (price_levels <= high_price))[0]

                if len(bins_covered) > 0:
                    # Distribute volume evenly across covered bins
                    volume_at_price[bins_covered] += vol / len(bins_covered)

            # Find POC (Point of Control) - price with highest volume
            if volume_at_price.sum() > 0:
                poc_idx = np.argmax(volume_at_price)
                session_poc = price_levels[poc_idx]

                # Calculate Value Area (70% of total volume)
                total_volume = volume_at_price.sum()
                target_volume = total_volume * 0.70

                # Start from POC and expand outward to capture 70% volume
                va_indices = [poc_idx]
                accumulated_volume = volume_at_price[poc_idx]

                left_idx = poc_idx - 1
                right_idx = poc_idx + 1

                while accumulated_volume < target_volume:
                    left_vol = volume_at_price[left_idx] if left_idx >= 0 else 0
                    right_vol = volume_at_price[right_idx] if right_idx < len(volume_at_price) else 0

                    if left_vol == 0 and right_vol == 0:
                        break

                    # Expand to side with more volume
                    if left_vol > right_vol and left_idx >= 0:
                        va_indices.append(left_idx)
                        accumulated_volume += left_vol
                        left_idx -= 1
                    elif right_idx < len(volume_at_price):
                        va_indices.append(right_idx)
                        accumulated_volume += right_vol
                        right_idx += 1
                    else:
                        break

                # Value Area High and Low
                va_indices = sorted(va_indices)
                session_vah = price_levels[va_indices[-1]]
                session_val = price_levels[va_indices[0]]
                session_va_pct = accumulated_volume / total_volume * 100
            else:
                # Fallback if no volume data
                session_poc = session_close.mean()
                session_vah = session_high.max()
                session_val = session_low.min()
                session_va_pct = 0

            # Assign to all candles in this session
            poc.loc[indices] = session_poc
            vah.loc[indices] = session_vah
            val.loc[indices] = session_val
            va_pct.loc[indices] = session_va_pct

        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'va_pct': va_pct
        }

    def _calculate_initial_balance(self) -> Dict[str, pd.Series]:
        """
        Calculate Initial Balance (first hour range of each session)

        Returns:
            Dictionary with ib_high, ib_low, and ib_range series
        """
        ib_high = pd.Series(index=self.df.index, dtype=float)
        ib_low = pd.Series(index=self.df.index, dtype=float)
        ib_range = pd.Series(index=self.df.index, dtype=float)

        # Process each session
        for (date, session), group in self.df.groupby(['date', 'session']):
            indices = group.index

            if len(indices) == 0:
                continue

            # Sort by timestamp to get first hour
            session_data = group.sort_values('timestamp')

            # Determine first hour based on timeframe
            # For M15: 4 candles = 1 hour
            # For M5: 12 candles = 1 hour
            # For M1: 60 candles = 1 hour
            # For H1: 1 candle = 1 hour

            timeframe_map = {
                'M1': 60, 'M5': 12, 'M15': 4, 'M30': 2,
                'H1': 1, 'H4': 1, 'D1': 1
            }
            first_hour_candles = timeframe_map.get(self.timeframe, 4)  # Default to M15

            # Get first hour data
            first_hour = session_data.iloc[:min(first_hour_candles, len(session_data))]

            # Calculate IB high/low
            session_ib_high = first_hour['high'].max()
            session_ib_low = first_hour['low'].min()
            session_ib_range = session_ib_high - session_ib_low

            # Assign to all candles in session
            ib_high.loc[indices] = session_ib_high
            ib_low.loc[indices] = session_ib_low
            ib_range.loc[indices] = session_ib_range

        return {
            'ib_high': ib_high,
            'ib_low': ib_low,
            'ib_range': ib_range
        }

    def to_structured_json(self) -> Dict[str, Any]:
        """
        Convert DataFrame to structured JSON format

        Returns:
            Structured dictionary ready for JSON serialization
        """
        print(f"📦 Converting to structured JSON...")

        if self.df is None:
            return {}

        # Prepare metadata
        metadata = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_candles": len(self.df),
            "date_range": {
                "start": self.df['timestamp'].iloc[0].isoformat(),
                "end": self.df['timestamp'].iloc[-1].isoformat(),
                "duration_days": (self.df['timestamp'].iloc[-1] - self.df['timestamp'].iloc[0]).days
            },
            "indicators": [
                "rsi", "macd", "stoch", "ema_9", "ema_20", "ema_50", "ema_200",
                "sma_20", "sma_50", "bb_bands", "atr", "vwap", "volume_ratio",
                "price_change", "candle_structure",
                "market_profile_poc", "market_profile_vah_val", "session_vwap", "initial_balance"
            ],
            "processed_at": datetime.now().isoformat(),
            "format_version": "2.1_market_profile"
        }

        # Process each candle
        candles = []
        for idx, row in self.df.iterrows():
            try:
                # Get contextual info
                trend = self._get_trend_description(row)
                rsi_state = self._get_rsi_state(row['rsi'])
                vol_state = self._get_volume_state(row['volume_ratio'])
                session = self._get_session(row['timestamp'])

                # Create summary (optimized for embeddings)
                summary = (
                    f"{self.symbol} {self.timeframe} | "
                    f"Close {row['close']:.2f} | "
                    f"RSI {row['rsi']:.0f} ({rsi_state}) | "
                    f"Trend {trend} | "
                    f"Vol {row['volume_ratio']:.1f}x ({vol_state}) | "
                    f"Session {session} | "
                    f"MACD {'positive' if row['macd'] > 0 else 'negative'}"
                )

                candle = {
                    "timestamp": row['timestamp'].isoformat(),
                    "candle_index": int(idx),

                    # OHLCV
                    "ohlc": {
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close'])
                    },
                    "volume": float(row['volume']),

                    # Technical Indicators
                    "indicators": {
                        "momentum": {
                            "rsi": float(row['rsi']),
                            "macd": float(row['macd']),
                            "macd_signal": float(row['macd_signal']),
                            "macd_histogram": float(row['macd_histogram']),
                            "stoch_k": float(row['stoch_k']),
                            "stoch_d": float(row['stoch_d'])
                        },
                        "trend": {
                            "ema_9": float(row['ema_9']),
                            "ema_20": float(row['ema_20']),
                            "ema_50": float(row['ema_50']),
                            "ema_200": float(row['ema_200']),
                            "sma_20": float(row['sma_20']),
                            "sma_50": float(row['sma_50'])
                        },
                        "volatility": {
                            "bb_upper": float(row['bb_upper']),
                            "bb_middle": float(row['bb_middle']),
                            "bb_lower": float(row['bb_lower']),
                            "bb_width": float(row['bb_width']),
                            "atr": float(row['atr'])
                        },
                        "volume": {
                            "volume_sma": float(row['volume_sma']),
                            "volume_ratio": float(row['volume_ratio']),
                            "vwap": float(row['vwap'])
                        },
                        "market_profile": {
                            "session_vwap": float(row['session_vwap']),
                            "poc": float(row['poc']),
                            "vah": float(row['vah']),
                            "val": float(row['val']),
                            "value_area_pct": float(row['value_area_pct']),
                            "ib_high": float(row['ib_high']),
                            "ib_low": float(row['ib_low']),
                            "ib_range": float(row['ib_range'])
                        },
                        "price_action": {
                            "price_change_pct": float(row['price_change_pct']),
                            "price_change_abs": float(row['price_change_abs']),
                            "body_size": float(row['body_size']),
                            "upper_wick": float(row['upper_wick']),
                            "lower_wick": float(row['lower_wick']),
                            "hl_range": float(row['hl_range'])
                        }
                    },

                    # Context (for filtering)
                    "context": {
                        "trend": trend,
                        "rsi_state": rsi_state,
                        "volume_state": vol_state,
                        "session": session,
                        "day_of_week": row['timestamp'].strftime('%A'),
                        "hour": row['timestamp'].hour
                    },

                    # Summary (optimized for embeddings)
                    "summary": summary
                }

                candles.append(candle)

            except Exception as e:
                print(f"⚠️  Error processing candle at index {idx}: {e}")
                continue

        print(f"✅ Converted {len(candles)} candles to structured format")

        return {
            "metadata": metadata,
            "candles": candles
        }

    def save_json(self, output_path: str, data: Dict[str, Any]) -> bool:
        """Save structured data to JSON file"""
        print(f"💾 Saving to: {output_path}")

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


def main():
    parser = argparse.ArgumentParser(
        description='Convert MT5 CSV to structured JSON for LLM+RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full history file
  python scripts/mt5_to_structured_json.py \\
      --input data/XAUUSD_PERIOD_M15_0.csv \\
      --output data/structured/XAUUSD_M15_full.json \\
      --symbol XAUUSD --timeframe M15

  # Process live data (200 candles)
  python scripts/mt5_to_structured_json.py \\
      --input data/XAUUSD_PERIOD_M15_200.csv \\
      --output data/structured/XAUUSD_M15_live.json \\
      --symbol XAUUSD --timeframe M15
        """
    )

    parser.add_argument('--input', required=True, help='Input MT5 CSV file')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol (default: XAUUSD)')
    parser.add_argument('--timeframe', default='M15', help='Timeframe (default: M15)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1

    print("=" * 70)
    print("MT5 → STRUCTURED JSON CONVERTER")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Symbol: {args.symbol}")
    print(f"TF:     {args.timeframe}")
    print("=" * 70)

    # Process
    converter = MT5ToStructuredJSON(args.symbol, args.timeframe)

    # Step 1: Load CSV
    if not converter.load_csv(args.input):
        return 1

    # Step 2: Normalize columns
    if not converter.normalize_columns():
        return 1

    # Step 3: Parse timestamps
    if not converter.parse_timestamps():
        return 1

    # Step 4: Calculate indicators
    if not converter.calculate_indicators():
        return 1

    # Step 5: Convert to structured JSON
    structured_data = converter.to_structured_json()
    if not structured_data:
        print("❌ Conversion failed")
        return 1

    # Step 6: Save
    if not converter.save_json(args.output, structured_data):
        return 1

    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"📊 Total candles: {structured_data['metadata']['total_candles']}")
    print(f"📅 Date range: {structured_data['metadata']['date_range']['start']}")
    print(f"            to {structured_data['metadata']['date_range']['end']}")
    print(f"📈 Indicators: {len(structured_data['metadata']['indicators'])}")
    print(f"💾 Output: {args.output}")
    print("\n💡 Next step:")
    print(f"   python scripts/pattern_detector.py --input {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
