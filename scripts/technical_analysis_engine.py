#!/usr/bin/env python3
"""
Advanced Technical Analysis Engine

Shared technical analysis functions for comprehensive market analysis.
Includes advanced Market Profile, VWAP, volume analysis, and trading patterns.
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with intraday sessions and market microstructure"""

    def __init__(self):
        self.sessions = {
            'sydney': {'start': 21, 'end': 6, 'timezone': 'UTC'},  # 21:00-06:00 UTC
            'tokyo': {'start': 23, 'end': 8, 'timezone': 'UTC'},    # 23:00-08:00 UTC
            'london': {'start': 8, 'end': 17, 'timezone': 'UTC'},    # 08:00-17:00 UTC
            'new_york': {'start': 13, 'end': 22, 'timezone': 'UTC'}, # 13:00-22:00 UTC
            'asian': {'start': 0, 'end': 9, 'timezone': 'UTC'},      # Combined Asian session
            'european': {'start': 7, 'end': 16, 'timezone': 'UTC'}, # Combined European
            'american': {'start': 12, 'end': 21, 'timezone': 'UTC'}, # Combined American
        }

    def calculate_intraday_market_profiles(self, df, symbol=None):
        """Calculate Market Profile for each trading session"""
        profiles = {}

        if 'datetime' not in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df.index)
            except:
                # If index conversion fails, create a synthetic datetime index
                df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        for session_name, session_config in self.sessions.items():
            session_data = self._get_session_data(df, session_config)

            if len(session_data) > 10:  # Minimum data for meaningful profile
                profiles[session_name] = self._calculate_session_market_profile(session_data, session_name)

                # Add session-specific VWAP
                profiles[session_name]['session_vwap'] = self._calculate_session_vwap(session_data)

                # Add session volume analysis
                profiles[session_name]['session_volume_analysis'] = self._calculate_session_volume_analysis(session_data)

        return profiles

    def _get_session_data(self, df, session_config):
        """Extract data for specific trading session"""
        start_hour = session_config['start']
        end_hour = session_config['end']

        if start_hour > end_hour:  # Session spans midnight
            session_mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
        else:  # Same day session
            session_mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)

        return df[session_mask].copy()

    def _calculate_session_market_profile(self, session_data, session_name):
        """Calculate Market Profile for specific session"""
        if len(session_data) < 10:
            return self._empty_profile()

        # Create price bins (more bins for intraday precision)
        price_min = session_data['low'].min()
        price_max = session_data['high'].max()
        num_bins = min(100, max(20, len(session_data) // 5))  # Dynamic bin count
        price_bins = np.linspace(price_min, price_max, num_bins)

        # Calculate volume at each price level
        volume_at_price = []
        tpo_at_price = []
        buy_volume_at_price = []
        sell_volume_at_price = []

        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]

            # Find candles that traded in this price range
            in_range = ((session_data['low'] <= bin_high) & (session_data['high'] >= bin_low))

            if in_range.any():
                # Volume calculation
                volume_in_range = 0
                buy_volume = 0
                sell_volume = 0

                for _, candle in session_data[in_range].iterrows():
                    # Distribute volume proportionally within the range
                    if candle['high'] != candle['low']:
                        overlap = min(candle['high'], bin_high) - max(candle['low'], bin_low)
                        proportion = overlap / (candle['high'] - candle['low'])
                        candle_volume = candle['volume'] * proportion
                    else:
                        candle_volume = candle['volume']

                    volume_in_range += candle_volume

                    # Classify as buy/sell volume based on candle direction
                    if candle['close'] > candle['open']:
                        buy_volume += candle_volume
                    elif candle['close'] < candle['open']:
                        sell_volume += candle_volume
                    else:
                        # Doji - split equally
                        buy_volume += candle_volume / 2
                        sell_volume += candle_volume / 2

                volume_at_price.append(volume_in_range)
                tpo_at_price.append(in_range.sum())
                buy_volume_at_price.append(buy_volume)
                sell_volume_at_price.append(sell_volume)
            else:
                volume_at_price.append(0)
                tpo_at_price.append(0)
                buy_volume_at_price.append(0)
                sell_volume_at_price.append(0)

        # Find POC (price with highest volume)
        if max(volume_at_price) > 0:
            poc_index = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

            # Calculate Value Area (68% of total volume)
            total_volume = sum(volume_at_price)
            target_volume = total_volume * 0.68

            value_area_low = poc_price
            value_area_high = poc_price
            accumulated_volume = volume_at_price[poc_index]

            lower_index = poc_index
            upper_index = poc_index

            while accumulated_volume < target_volume and (lower_index > 0 or upper_index < len(volume_at_price) - 1):
                # Add next highest volume level
                if lower_index > 0 and (upper_index >= len(volume_at_price) - 1 or
                    volume_at_price[lower_index - 1] >= volume_at_price[upper_index + 1]):
                    lower_index -= 1
                    accumulated_volume += volume_at_price[lower_index]
                    value_area_low = price_bins[lower_index]
                elif upper_index < len(volume_at_price) - 1:
                    upper_index += 1
                    accumulated_volume += volume_at_price[upper_index]
                    value_area_high = price_bins[upper_index + 1]
                else:
                    break

            # Calculate High Volume Nodes (HVNs) and Low Volume Nodes (LVNs)
            hvns, lvns = self._find_volume_nodes(volume_at_price, price_bins)

            # Calculate Volume Weighted Average Price
            vwap = sum((price_bins[i] + price_bins[i+1])/2 * volume_at_price[i]
                       for i in range(len(volume_at_price))) / total_volume if total_volume > 0 else poc_price

            current_price = session_data['close'].iloc[-1]

            profile = {
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_width': value_area_high - value_area_low,
                'value_area_volume_pct': (accumulated_volume / total_volume) * 100,
                'current_price_in_va': value_area_low <= current_price <= value_area_high,
                'vwap': vwap,
                'current_price_vs_vwap': (current_price - vwap) / vwap * 100 if vwap != 0 else 0,
                'volume_distribution': self._classify_volume_distribution(volume_at_price),
                'profile_type': self._classify_session_profile_type(session_data, poc_price, value_area_low, value_area_high),
                'high_volume_nodes': hvns,
                'low_volume_nodes': lvns,
                'buy_sell_ratio': sum(buy_volume_at_price) / (sum(sell_volume_at_price) + 0.001),
                'session_development': self._analyze_session_development(session_data),
            }

        else:
            profile = self._empty_profile()

        return profile

    def _calculate_session_vwap(self, session_data):
        """Calculate VWAP for the session"""
        if len(session_data) == 0:
            return {}

        typical_price = (session_data['high'] + session_data['low'] + session_data['close']) / 3
        cumulative_volume = session_data['volume'].cumsum()
        cumulative_tpv = (typical_price * session_data['volume']).cumsum()
        vwap = cumulative_tpv / cumulative_volume

        # VWAP bands
        returns = typical_price.pct_change()
        rolling_std = returns.rolling(min(20, len(returns))).std() if len(returns) > 1 else pd.Series([0])

        vwap_upper = vwap + rolling_std * np.sqrt(cumulative_volume.index + 1)
        vwap_lower = vwap - rolling_std * np.sqrt(cumulative_volume.index + 1)

        return {
            'vwap': vwap.iloc[-1] if len(vwap) > 0 else session_data['close'].iloc[-1],
            'vwap_upper_1std': vwap_upper.iloc[-1] if len(vwap_upper) > 0 else session_data['high'].iloc[-1],
            'vwap_lower_1std': vwap_lower.iloc[-1] if len(vwap_lower) > 0 else session_data['low'].iloc[-1],
            'vwap_trend': self._calculate_vwap_trend(vwap),
            'price_above_vwap': session_data['close'].iloc[-1] > vwap.iloc[-1] if len(vwap) > 0 else False,
            'vwap_distance': (session_data['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100 if len(vwap) > 0 and vwap.iloc[-1] != 0 else 0,
        }

    def _calculate_session_volume_analysis(self, session_data):
        """Analyze volume patterns within the session"""
        if len(session_data) < 5:
            return {}

        volume_analysis = {
            'total_volume': session_data['volume'].sum(),
            'avg_volume': session_data['volume'].mean(),
            'volume_std': session_data['volume'].std(),
            'max_volume': session_data['volume'].max(),
            'min_volume': session_data['volume'].min(),
            'volume_trend': self._calculate_volume_trend(session_data['volume']),
            'volume_spike': session_data['volume'].iloc[-1] > session_data['volume'].mean() + 2 * session_data['volume'].std(),
            'volume_accumulation': self._calculate_volume_accumulation(session_data),
            'buying_pressure': self._calculate_buying_pressure(session_data),
            'selling_pressure': self._calculate_selling_pressure(session_data),
            'volume_profile_trend': self._analyze_volume_profile_trend(session_data),
        }

        return volume_analysis

    def calculate_fixed_range_volume_profile(self, df, range_type='session'):
        """Calculate Fixed Range Volume Profile"""
        if len(df) < 10:
            return {}

        if range_type == 'session':
            # Use current day's range
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            current_date = df['date'].iloc[-1]
            range_data = df[df['date'] == current_date]
        elif range_type == 'week':
            # Use current week's range
            df['week'] = pd.to_datetime(df['datetime']).dt.isocalendar().week
            current_week = df['week'].iloc[-1]
            range_data = df[df['week'] == current_week]
        else:  # 'range' - use visible range on chart
            range_data = df.tail(100)  # Last 100 candles

        if len(range_data) < 10:
            return {}

        # Calculate volume profile for the fixed range
        price_min = range_data['low'].min()
        price_max = range_data['high'].max()
        num_bins = min(50, max(20, len(range_data) // 3))
        price_bins = np.linspace(price_min, price_max, num_bins)

        volume_at_price = []
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]

            in_range = ((range_data['low'] <= bin_high) & (range_data['high'] >= bin_low))
            volume_in_range = 0

            for _, candle in range_data[in_range].iterrows():
                if candle['high'] != candle['low']:
                    overlap = min(candle['high'], bin_high) - max(candle['low'], bin_low)
                    proportion = overlap / (candle['high'] - candle['low'])
                    volume_in_range += candle['volume'] * proportion
                else:
                    volume_in_range += candle['volume']

            volume_at_price.append(volume_in_range)

        # Find POC and Value Area for the fixed range
        total_volume = sum(volume_at_price)
        if total_volume > 0:
            poc_index = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

            # Value Area calculation
            target_volume = total_volume * 0.70  # 70% for fixed range
            value_area_low = poc_price
            value_area_high = poc_price
            accumulated_volume = volume_at_price[poc_index]

            lower_index = poc_index
            upper_index = poc_index

            while accumulated_volume < target_volume and (lower_index > 0 or upper_index < len(volume_at_price) - 1):
                if lower_index > 0 and (upper_index >= len(volume_at_price) - 1 or
                    volume_at_price[lower_index - 1] >= volume_at_price[upper_index + 1]):
                    lower_index -= 1
                    accumulated_volume += volume_at_price[lower_index]
                    value_area_low = price_bins[lower_index]
                elif upper_index < len(volume_at_price) - 1:
                    upper_index += 1
                    accumulated_volume += volume_at_price[upper_index]
                    value_area_high = price_bins[upper_index + 1]
                else:
                    break

            return {
                'range_type': range_type,
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_width': value_area_high - value_area_low,
                'total_range_volume': total_volume,
                'poc_volume_pct': (volume_at_price[poc_index] / total_volume) * 100,
                'value_area_volume_pct': (accumulated_volume / total_volume) * 100,
                'volume_nodes': self._find_volume_nodes(volume_at_price, price_bins),
                'range_high': price_max,
                'range_low': price_min,
                'range_width': price_max - price_min,
            }

        return {}

    def detect_gaps(self, df):
        """Detect price gaps and gap patterns"""
        if len(df) < 2:
            return {}

        gaps = []
        gap_analysis = {
            'total_gaps': 0,
            'gap_up_count': 0,
            'gap_down_count': 0,
            'gap_fill_rate': 0,
            'average_gap_size': 0,
            'largest_gap': 0,
            'recent_gaps': [],
            'gap_statistics': {}
        }

        for i in range(1, len(df)):
            prev_high = df['high'].iloc[i-1]
            prev_low = df['low'].iloc[i-1]
            curr_open = df['open'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_time = df['datetime'].iloc[i] if 'datetime' in df.columns else i

            # Gap Up detection
            if curr_open > prev_high:
                gap_size = curr_open - prev_high
                gap_pct = (gap_size / prev_high) * 100

                gap = {
                    'type': 'gap_up',
                    'size': gap_size,
                    'size_pct': gap_pct,
                    'open': curr_open,
                    'prev_high': prev_high,
                    'time': curr_time,
                    'filled': curr_close <= prev_high,  # Gap filled in same period
                    'fill_time': None,  # Would need future data to determine
                }
                gaps.append(gap)
                gap_analysis['gap_up_count'] += 1

            # Gap Down detection
            elif curr_open < prev_low:
                gap_size = prev_low - curr_open
                gap_pct = (gap_size / prev_low) * 100

                gap = {
                    'type': 'gap_down',
                    'size': gap_size,
                    'size_pct': gap_pct,
                    'open': curr_open,
                    'prev_low': prev_low,
                    'time': curr_time,
                    'filled': curr_close >= prev_low,  # Gap filled in same period
                    'fill_time': None,
                }
                gaps.append(gap)
                gap_analysis['gap_down_count'] += 1

        gap_analysis['total_gaps'] = len(gaps)

        if gaps:
            gap_sizes = [gap['size_pct'] for gap in gaps]
            gap_analysis['average_gap_size'] = np.mean(gap_sizes)
            gap_analysis['largest_gap'] = max(gap_sizes)
            gap_analysis['gap_fill_rate'] = sum(1 for gap in gaps if gap['filled']) / len(gaps)
            gap_analysis['recent_gaps'] = gaps[-5:]  # Last 5 gaps

        return gap_analysis

    def detect_order_blocks(self, df, lookback=20):
        """Detect order blocks (institutional buying/selling zones)"""
        if len(df) < lookback + 10:
            return {'order_blocks': [], 'statistics': {}}

        order_blocks = []

        for i in range(lookback, len(df)):
            # Look for strong momentum candles (potential order block creation)
            current_candle = df.iloc[i]

            # Bullish order block criteria
            if (current_candle['close'] > current_candle['open'] and  # Bullish candle
                (current_candle['close'] - current_candle['open']) > (current_candle['high'] - current_candle['low']) * 0.6 and  # Large body
                current_candle['volume'] > df['volume'].iloc[i-lookback:i].mean()):  # High volume

                # Find the preceding down move
                for j in range(i-1, max(i-lookback-1, i-10), -1):
                    if df['close'].iloc[j] < df['close'].iloc[j-1]:  # Down candle
                        order_block = {
                            'type': 'bullish_order_block',
                            'high': df['high'].iloc[j],
                            'low': df['low'].iloc[j],
                            'close': df['close'].iloc[j],
                            'open': df['open'].iloc[j],
                            'volume': df['volume'].iloc[j],
                            'time': df['datetime'].iloc[j] if 'datetime' in df.columns else j,
                            'strength': self._calculate_order_block_strength(df, j, 'bullish'),
                            'tested': df['low'].iloc[i:].min() <= df['low'].iloc[j],  # Tested later
                            'reaction': self._analyze_order_block_reaction(df, j, 'bullish'),
                        }
                        order_blocks.append(order_block)
                        break

            # Bearish order block criteria
            elif (current_candle['close'] < current_candle['open'] and  # Bearish candle
                  (current_candle['open'] - current_candle['close']) > (current_candle['high'] - current_candle['low']) * 0.6 and  # Large body
                  current_candle['volume'] > df['volume'].iloc[i-lookback:i].mean()):  # High volume

                # Find the preceding up move
                for j in range(i-1, max(i-lookback-1, i-10), -1):
                    if df['close'].iloc[j] > df['close'].iloc[j-1]:  # Up candle
                        order_block = {
                            'type': 'bearish_order_block',
                            'high': df['high'].iloc[j],
                            'low': df['low'].iloc[j],
                            'close': df['close'].iloc[j],
                            'open': df['open'].iloc[j],
                            'volume': df['volume'].iloc[j],
                            'time': df['datetime'].iloc[j] if 'datetime' in df.columns else j,
                            'strength': self._calculate_order_block_strength(df, j, 'bearish'),
                            'tested': df['high'].iloc[i:].max() >= df['high'].iloc[j],  # Tested later
                            'reaction': self._analyze_order_block_reaction(df, j, 'bearish'),
                        }
                        order_blocks.append(order_block)
                        break

        # Analyze order block statistics
        if order_blocks:
            bullish_blocks = [ob for ob in order_blocks if ob['type'] == 'bullish_order_block']
            bearish_blocks = [ob for ob in order_blocks if ob['type'] == 'bearish_order_block']

            statistics = {
                'total_order_blocks': len(order_blocks),
                'bullish_count': len(bullish_blocks),
                'bearish_count': len(bearish_blocks),
                'avg_strength': np.mean([ob['strength'] for ob in order_blocks]),
                'test_rate': sum(1 for ob in order_blocks if ob['tested']) / len(order_blocks),
                'recent_blocks': order_blocks[-3:],  # Last 3 order blocks
            }
        else:
            statistics = {
                'total_order_blocks': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_strength': 0,
                'test_rate': 0,
                'recent_blocks': [],
            }

        return {
            'order_blocks': order_blocks,
            'statistics': statistics
        }

    def calculate_fibonacci_retracements(self, df, swing_length=100):
        """Calculate Fibonacci retracement levels from significant swings"""
        if len(df) < swing_length:
            return {}

        # Find significant swing high and low
        recent_data = df.tail(swing_length)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()

        # Get the time of these swings
        swing_high_time = recent_data[recent_data['high'] == swing_high]['datetime'].iloc[0] if 'datetime' in recent_data.columns else None
        swing_low_time = recent_data[recent_data['low'] == swing_low]['datetime'].iloc[0] if 'datetime' in recent_data.columns else None

        # Determine swing direction
        current_price = df['close'].iloc[-1]

        if swing_high_time and swing_low_time:
            if swing_high_time > swing_low_time:  # Up swing
                swing_direction = 'upward'
                diff = swing_high - swing_low
            else:  # Down swing
                swing_direction = 'downward'
                diff = swing_low - swing_high
        else:
            swing_direction = 'unknown'
            diff = swing_high - swing_low

        # Calculate Fibonacci levels
        fib_levels = {
            'swing_direction': swing_direction,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_high_time': swing_high_time,
            'swing_low_time': swing_low_time,
            'current_price': current_price,
            'retracement_levels': {},
            'extension_levels': {},
        }

        if swing_direction == 'upward':
            # Retracement levels (measuring pullbacks)
            fib_levels['retracement_levels'] = {
                '0.0%': swing_high,
                '23.6%': swing_high - diff * 0.236,
                '38.2%': swing_high - diff * 0.382,
                '50.0%': swing_high - diff * 0.500,
                '61.8%': swing_high - diff * 0.618,
                '78.6%': swing_high - diff * 0.786,
                '100.0%': swing_low,
            }

            # Extension levels (measuring projections beyond swing)
            fib_levels['extension_levels'] = {
                '127.2%': swing_high + diff * 0.272,
                '161.8%': swing_high + diff * 0.618,
                '200.0%': swing_high + diff * 1.000,
                '261.8%': swing_high + diff * 1.618,
            }

        elif swing_direction == 'downward':
            # Retracement levels (measuring pullbacks)
            fib_levels['retracement_levels'] = {
                '0.0%': swing_low,
                '23.6%': swing_low + diff * 0.236,
                '38.2%': swing_low + diff * 0.382,
                '50.0%': swing_low + diff * 0.500,
                '61.8%': swing_low + diff * 0.618,
                '78.6%': swing_low + diff * 0.786,
                '100.0%': swing_high,
            }

            # Extension levels
            fib_levels['extension_levels'] = {
                '127.2%': swing_low - diff * 0.272,
                '161.8%': swing_low - diff * 0.618,
                '200.0%': swing_low - diff * 1.000,
                '261.8%': swing_low - diff * 1.618,
            }

        # Find which level is closest to current price
        if fib_levels['retracement_levels']:
            closest_level = min(fib_levels['retracement_levels'].items(),
                              key=lambda x: abs(x[1] - current_price))
            fib_levels['closest_level'] = {
                'name': closest_level[0],
                'price': closest_level[1],
                'distance': abs(closest_level[1] - current_price),
                'distance_pct': abs(closest_level[1] - current_price) / current_price * 100,
            }

        return fib_levels

    def calculate_ichimoku_cloud(self, df):
        """Calculate Ichimoku Cloud components"""
        if len(df) < 52:  # Minimum for Senkou Span B
            return {}

        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']

        # Tenkan-sen (Conversion Line): 9-period high+low average
        tenkan_sen = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2

        # Kijun-sen (Base Line): 26-period high+low average
        kijun_sen = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2

        # Senkou Span A (Leading Span A): Tenkan+Kijun average, plotted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): 52-period high+low average, plotted 26 periods ahead
        senkou_span_b = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)

        # Chikou Span (Lagging Span): Close plotted 26 periods behind
        chikou_span = close_prices.shift(-26)

        current_values = {
            'tenkan_sen': tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else None,
            'kijun_sen': kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else None,
            'senkou_span_a': senkou_span_a.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) else None,
            'senkou_span_b': senkou_span_b.iloc[-1] if not pd.isna(senkou_span_b.iloc[-1]) else None,
            'chikou_span': chikou_span.iloc[-1] if not pd.isna(chikou_span.iloc[-1]) else None,
        }

        # Cloud analysis
        if all(current_values.values()):
            cloud_top = max(current_values['senkou_span_a'], current_values['senkou_span_b'])
            cloud_bottom = min(current_values['senkou_span_a'], current_values['senkou_span_b'])
            current_price = close_prices.iloc[-1]

            current_values.update({
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom,
                'cloud_thickness': cloud_top - cloud_bottom,
                'price_above_cloud': current_price > cloud_top,
                'price_below_cloud': current_price < cloud_bottom,
                'price_in_cloud': cloud_bottom <= current_price <= cloud_top,
                'cloud_color': 'green' if current_values['senkou_span_a'] > current_values['senkou_span_b'] else 'red',
                'tk_cross_bullish': current_values['tenkan_sen'] > current_values['kijun_sen'],
                'tk_cross_bearish': current_values['tenkan_sen'] < current_values['kijun_sen'],
                'future_cloud_bullish': senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) and not pd.isna(senkou_span_b.iloc[-1]) else False,
            })

        return {
            'current_values': current_values,
            'historical_data': {
                'tenkan_sen': tenkan_sen.dropna().tolist()[-20:],  # Last 20 values
                'kijun_sen': kijun_sen.dropna().tolist()[-20:],
                'senkou_span_a': senkou_span_a.dropna().tolist()[-20:],
                'senkou_span_b': senkou_span_b.dropna().tolist()[-20:],
                'chikou_span': chikou_span.dropna().tolist()[-20:],
            }
        }

    def calculate_market_microstructure(self, df):
        """Analyze market microstructure and liquidity patterns"""
        if len(df) < 20:
            return {}

        microstructure = {
            'liquidity_analysis': self._analyze_liquidity(df),
            'order_flow': self._analyze_order_flow(df),
            'market_impact': self._calculate_market_impact(df),
            'tick_analysis': self._analyze_tick_patterns(df),
            'spread_analysis': self._analyze_spread_patterns(df),
            'depth_analysis': self._analyze_market_depth(df),
        }

        return microstructure

    # Helper methods
    def _empty_profile(self):
        """Return empty market profile structure"""
        return {
            'poc': None,
            'value_area_high': None,
            'value_area_low': None,
            'value_area_width': 0,
            'value_area_volume_pct': 0,
            'current_price_in_va': False,
            'vwap': None,
            'current_price_vs_vwap': 0,
            'volume_distribution': 'UNKNOWN',
            'profile_type': 'UNKNOWN',
            'high_volume_nodes': [],
            'low_volume_nodes': [],
            'buy_sell_ratio': 1,
            'session_development': 'UNKNOWN',
        }

    def _find_volume_nodes(self, volume_at_price, price_bins):
        """Find High Volume Nodes (HVNs) and Low Volume Nodes (LVNs)"""
        if len(volume_at_price) < 5:
            return {'hvns': [], 'lvns': []}

        volume_array = np.array(volume_at_price)
        mean_volume = np.mean(volume_array)
        std_volume = np.std(volume_array)

        # HVNs: Volume > mean + 0.5*std
        hvn_indices = np.where(volume_array > mean_volume + 0.5 * std_volume)[0]
        hvns = []
        for idx in hvn_indices:
            hvns.append({
                'price': (price_bins[idx] + price_bins[idx + 1]) / 2,
                'volume': volume_array[idx],
                'volume_pct': (volume_array[idx] / sum(volume_array)) * 100,
            })

        # LVNs: Volume < mean - 0.5*std
        lvn_indices = np.where(volume_array < mean_volume - 0.5 * std_volume)[0]
        lvns = []
        for idx in lvn_indices:
            lvns.append({
                'price': (price_bins[idx] + price_bins[idx + 1]) / 2,
                'volume': volume_array[idx],
                'volume_pct': (volume_array[idx] / sum(volume_array)) * 100,
            })

        return {'hvns': sorted(hvns, key=lambda x: x['volume'], reverse=True)[:5],  # Top 5 HVNs
                'lvns': sorted(lvns, key=lambda x: x['volume'])[:5]}  # Bottom 5 LVNs

    def _classify_volume_distribution(self, volume_at_price):
        """Classify the shape of volume distribution"""
        if len(volume_at_price) < 5:
            return 'INSUFFICIENT_DATA'

        volume_array = np.array(volume_at_price)
        max_idx = np.argmax(volume_array)
        total_volume = sum(volume_array)

        # Calculate volume above and below POC
        volume_above = sum(volume_array[max_idx+1:])
        volume_below = sum(volume_array[:max_idx])

        if volume_above > volume_below * 1.2:
            return 'UPWARD_BIASED'
        elif volume_below > volume_above * 1.2:
            return 'DOWNWARD_BIASED'
        else:
            return 'BALANCED'

    def _classify_session_profile_type(self, session_data, poc, va_low, va_high):
        """Classify the session profile development"""
        if len(session_data) < 10:
            return 'INSUFFICIENT_DATA'

        current_price = session_data['close'].iloc[-1]

        # Check for trend profile
        price_range = session_data['high'].max() - session_data['low'].min()
        value_area_width = va_high - va_low

        if current_price > va_high:
            return 'TREND_UP'
        elif current_price < va_low:
            return 'TREND_DOWN'
        elif value_area_width < price_range * 0.3:
            return 'BALANCED'
        else:
            return 'ROTATION'

    def _analyze_session_development(self, session_data):
        """Analyze how the session developed over time"""
        if len(session_data) < 20:
            return 'INSUFFICIENT_DATA'

        # Divide session into thirds and analyze development
        third = len(session_data) // 3

        first_third = session_data.iloc[:third]
        second_third = session_data.iloc[third:2*third]
        last_third = session_data.iloc[2*third:]

        # Compare ranges in each third
        first_range = first_third['high'].max() - first_third['low'].min()
        second_range = second_third['high'].max() - second_third['low'].min()
        last_range = last_third['high'].max() - last_third['low'].min()

        if first_range > second_range > last_range:
            return 'RANGE_CONTRACTION'
        elif first_range < second_range < last_range:
            return 'RANGE_EXPANSION'
        elif second_range < first_range and second_range < last_range:
            return 'BALANCED_DEVELOPMENT'
        else:
            return 'IRREGULAR_DEVELOPMENT'

    def _calculate_vwap_trend(self, vwap_series):
        """Calculate VWAP trend direction"""
        if len(vwap_series) < 5:
            return 'UNKNOWN'

        recent_vwap = vwap_series.tail(5)
        if recent_vwap.is_monotonic_increasing:
            return 'RISING'
        elif recent_vwap.is_monotonic_decreasing:
            return 'FALLING'
        else:
            return 'SIDEWAYS'

    def _calculate_volume_trend(self, volume_series):
        """Calculate volume trend"""
        if len(volume_series) < 5:
            return 'UNKNOWN'

        recent_volume = volume_series.tail(5)
        avg_volume = recent_volume.mean()
        current_volume = recent_volume.iloc[-1]

        if current_volume > avg_volume * 1.2:
            return 'INCREASING'
        elif current_volume < avg_volume * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'

    def _calculate_volume_accumulation(self, session_data):
        """Calculate volume accumulation/distribution pattern"""
        if len(session_data) < 5:
            return {}

        buying_volume = session_data[session_data['close'] > session_data['open']]['volume'].sum()
        selling_volume = session_data[session_data['close'] < session_data['open']]['volume'].sum()
        total_volume = session_data['volume'].sum()

        return {
            'buying_volume': buying_volume,
            'selling_volume': selling_volume,
            'buying_pct': (buying_volume / total_volume) * 100 if total_volume > 0 else 50,
            'selling_pct': (selling_volume / total_volume) * 100 if total_volume > 0 else 50,
            'accumulation_trend': 'ACCUMULATION' if buying_volume > selling_volume else 'DISTRIBUTION',
        }

    def _calculate_buying_pressure(self, df):
        """Calculate buying pressure indicator"""
        if len(df) < 2:
            return 50

        # Money Flow Index-like calculation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow[df['close'] > df['close'].shift(1)].sum()
        negative_flow = money_flow[df['close'] < df['close'].shift(1)].sum()
        total_flow = positive_flow + negative_flow

        if total_flow > 0:
            return (positive_flow / total_flow) * 100
        else:
            return 50

    def _calculate_selling_pressure(self, df):
        """Calculate selling pressure indicator"""
        buying_pressure = self._calculate_buying_pressure(df)
        return 100 - buying_pressure

    def _analyze_volume_profile_trend(self, session_data):
        """Analyze how volume profile developed during session"""
        if len(session_data) < 10:
            return 'UNKNOWN'

        # Compare early vs late volume distribution
        midpoint = len(session_data) // 2
        early_data = session_data.iloc[:midpoint]
        late_data = session_data.iloc[midpoint:]

        early_volume = early_data['volume'].sum()
        late_volume = late_data['volume'].sum()

        if late_volume > early_volume * 1.3:
            return 'VOLUME_INCREASING'
        elif late_volume < early_volume * 0.7:
            return 'VOLUME_DECREASING'
        else:
            return 'VOLUME_STABLE'

    def _calculate_order_block_strength(self, df, index, block_type):
        """Calculate the strength of an order block"""
        if index < 3 or index >= len(df) - 3:
            return 0.5

        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['volume'].iloc[index-10:index+10].mean()

        # Price strength (size of the move)
        if block_type == 'bullish':
            price_strength = (df['close'].iloc[index] - df['open'].iloc[index]) / (df['high'].iloc[index] - df['low'].iloc[index])
        else:
            price_strength = (df['open'].iloc[index] - df['close'].iloc[index]) / (df['high'].iloc[index] - df['low'].iloc[index])

        # Momentum strength (preceding move)
        if block_type == 'bullish':
            momentum = -(df['close'].iloc[index-3:index].pct_change().mean())
        else:
            momentum = df['close'].iloc[index-3:index].pct_change().mean()

        # Combine strengths
        combined_strength = (volume_strength * 0.4 + price_strength * 0.4 + abs(momentum) * 0.2)
        return min(1.0, max(0.0, combined_strength))

    def _analyze_order_block_reaction(self, df, index, block_type):
        """Analyze how price reacted when testing the order block"""
        if index >= len(df) - 5:
            return 'NOT_TESTED'

        # Look at the next 5 candles after the order block
        future_data = df.iloc[index+1:index+6]

        if block_type == 'bullish':
            # Check if price found support at the order block
            min_price = future_data['low'].min()
            order_block_low = df['low'].iloc[index]

            if min_price <= order_block_low:
                # Check the reaction
                next_candle = future_data.iloc[0]
                if next_candle['close'] > next_candle['open']:
                    return 'BULLISH_REJECTION'
                else:
                    return 'BEARISH_BREAK'
            else:
                return 'NOT_TESTED_YET'
        else:
            # Check if price found resistance at the order block
            max_price = future_data['high'].max()
            order_block_high = df['high'].iloc[index]

            if max_price >= order_block_high:
                # Check the reaction
                next_candle = future_data.iloc[0]
                if next_candle['close'] < next_candle['open']:
                    return 'BEARISH_REJECTION'
                else:
                    return 'BULLISH_BREAK'
            else:
                return 'NOT_TESTED_YET'

    def _analyze_liquidity(self, df):
        """Analyze liquidity patterns"""
        return {
            'liquidity_level': 'NORMAL',  # Would need order book data for proper analysis
            'liquidity_trend': self._calculate_volume_trend(df['volume']),
        }

    def _analyze_order_flow(self, df):
        """Analyze order flow patterns"""
        return {
            'flow_direction': 'BALANCED',
            'flow_strength': 0.5,
        }

    def _calculate_market_impact(self, df):
        """Calculate market impact of trades"""
        if len(df) < 10:
            return {}

        price_changes = df['close'].pct_change().abs()
        volumes = df['volume']

        impact = price_changes * volumes

        return {
            'avg_impact': impact.mean(),
            'max_impact': impact.max(),
            'impact_trend': self._calculate_trend(impact.tail(10)),
        }

    def _analyze_tick_patterns(self, df):
        """Analyze tick patterns (simplified for OHLC data)"""
        return {
            'tick_intensity': 'NORMAL',
            'tick_pattern': 'IRREGULAR',
        }

    def _analyze_spread_patterns(self, df):
        """Analyze spread patterns (simplified - would need bid/ask data)"""
        # Use high-low as proxy for spread
        spreads = df['high'] - df['low']

        return {
            'avg_spread': spreads.mean(),
            'spread_trend': self._calculate_trend(spreads.tail(10)),
            'spread_volatility': spreads.std(),
        }

    def _analyze_market_depth(self, df):
        """Analyze market depth (simplified - would need order book data)"""
        return {
            'depth_level': 'NORMAL',
            'depth_trend': 'STABLE',
        }

    def _calculate_trend(self, series):
        """Simple trend calculation"""
        if len(series) < 2:
            return 0

        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]

        if slope > 0.01:
            return 1
        elif slope < -0.01:
            return -1
        else:
            return 0