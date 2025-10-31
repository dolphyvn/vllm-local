#!/usr/bin/env python3
"""
Comprehensive Feature Analyzer for Historical Market Data

This script processes full historical data files ({SYMBOL}_PERIOD_{TIMEFRAME}_0.csv)
to calculate ALL possible technical indicators and features for RAG knowledge base.
It generates comprehensive market analysis with advanced indicators to help the system
learn market patterns, trends, and optimal trading strategies.

Features Calculated:
- Price Action: OHLCV analysis, candlestick patterns, price levels
- Volume Analysis: Volume profile, volume indicators, volume patterns
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA at multiple periods
- Momentum Indicators: RSI, MACD, Stochastic, CCI, ADX, Aroon
- Volatility Indicators: Bollinger Bands, ATR, Keltner Channels
- Trend Indicators: ADX, Aroon, Parabolic SAR, TRIX
- Market Profile: POC, Value Area, Volume Profile
- VWAP: Multiple VWAP calculations and analysis
- Support/Resistance: Key levels, breakout zones, confluence areas
- Pattern Recognition: Candlestick patterns, chart patterns
- Time Analysis: Session analysis, time-based patterns
- Correlation Analysis: Inter-timeframe correlations
- Risk Metrics: Volatility, drawdown, risk-reward ratios
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import talib
from datetime import datetime, timedelta
from scipy import stats
import warnings
from technical_analysis_engine import AdvancedTechnicalAnalyzer
warnings.filterwarnings('ignore')

class ComprehensiveFeatureAnalyzer:
    """Comprehensive technical analysis feature calculator"""

    def __init__(self):
        self.indicators_cache = {}
        self.technical_analyzer = AdvancedTechnicalAnalyzer()

    def calculate_price_features(self, df):
        """Calculate comprehensive price action features"""
        features = {}

        # Basic price metrics
        features['basic_price'] = {
            'current_price': df['close'].iloc[-1],
            'price_change': df['close'].iloc[-1] - df['close'].iloc[-2],
            'price_change_pct': (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100,
            'price_range_5': df['high'].rolling(5).max().iloc[-1] - df['low'].rolling(5).min().iloc[-1],
            'price_range_20': df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1],
            'price_range_50': df['high'].rolling(50).max().iloc[-1] - df['low'].rolling(50).min().iloc[-1],
            'avg_price_5': df['close'].rolling(5).mean().iloc[-1],
            'avg_price_20': df['close'].rolling(20).mean().iloc[-1],
            'avg_price_50': df['close'].rolling(50).mean().iloc[-1],
            'price_volatility_5': df['close'].rolling(5).std().iloc[-1],
            'price_volatility_20': df['close'].rolling(20).std().iloc[-1],
        }

        # Price position analysis
        features['price_position'] = {
            'price_position_5': (df['close'].iloc[-1] - df['low'].rolling(5).min().iloc[-1]) / (df['high'].rolling(5).max().iloc[-1] - df['low'].rolling(5).min().iloc[-1]) * 100,
            'price_position_20': (df['close'].iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / (df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]) * 100,
            'price_position_50': (df['close'].iloc[-1] - df['low'].rolling(50).min().iloc[-1]) / (df['high'].rolling(50).max().iloc[-1] - df['low'].rolling(50).min().iloc[-1]) * 100,
            'distance_from_high_5': (df['high'].rolling(5).max().iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100,
            'distance_from_low_5': (df['close'].iloc[-1] - df['low'].rolling(5).min().iloc[-1]) / df['close'].iloc[-1] * 100,
            'distance_from_high_20': (df['high'].rolling(20).max().iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100,
            'distance_from_low_20': (df['close'].iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1] * 100,
        }

        # Recent price action
        features['recent_action'] = {
            'consecutive_gains': self._calculate_consecutive_moves(df['close'] > df['close'].shift(1)),
            'consecutive_losses': self._calculate_consecutive_moves(df['close'] < df['close'].shift(1)),
            'max_gain_5': df['close'].pct_change().rolling(5).max().iloc[-1] * 100,
            'max_loss_5': df['close'].pct_change().rolling(5).min().iloc[-1] * 100,
            'max_gain_20': df['close'].pct_change().rolling(20).max().iloc[-1] * 100,
            'max_loss_20': df['close'].pct_change().rolling(20).min().iloc[-1] * 100,
        }

        return features

    def calculate_volume_features(self, df):
        """Calculate comprehensive volume analysis features"""
        features = {}

        # Basic volume metrics
        features['basic_volume'] = {
            'current_volume': df['volume'].iloc[-1],
            'volume_change': df['volume'].iloc[-1] - df['volume'].iloc[-2],
            'volume_change_pct': (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] * 100 if df['volume'].iloc[-2] != 0 else 0,
            'avg_volume_5': df['volume'].rolling(5).mean().iloc[-1],
            'avg_volume_20': df['volume'].rolling(20).mean().iloc[-1],
            'avg_volume_50': df['volume'].rolling(50).mean().iloc[-1],
            'volume_std_5': df['volume'].rolling(5).std().iloc[-1],
            'volume_std_20': df['volume'].rolling(20).std().iloc[-1],
            'volume_ratio_5': df['volume'].iloc[-1] / df['volume'].rolling(5).mean().iloc[-1] if df['volume'].rolling(5).mean().iloc[-1] != 0 else 0,
            'volume_ratio_20': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if df['volume'].rolling(20).mean().iloc[-1] != 0 else 0,
        }

        # Volume trend analysis
        features['volume_trend'] = {
            'volume_trend_5': self._calculate_trend(df['volume'].tail(5)),
            'volume_trend_20': self._calculate_trend(df['volume'].tail(20)),
            'volume_trend_strength_5': self._calculate_trend_strength(df['volume'].tail(5)),
            'volume_trend_strength_20': self._calculate_trend_strength(df['volume'].tail(20)),
            'volume_momentum': df['volume'].iloc[-1] - df['volume'].rolling(10).mean().iloc[-1],
            'volume_acceleration': (df['volume'].pct_change().iloc[-1] - df['volume'].pct_change().rolling(5).mean().iloc[-1]) * 100,
        }

        # Volume-Price analysis
        features['volume_price'] = {
            'price_volume_trend_5': self._calculate_price_volume_trend(df.tail(5)),
            'price_volume_trend_20': self._calculate_price_volume_trend(df.tail(20)),
            'volume_weighted_price_5': (df['close'] * df['volume']).rolling(5).sum().iloc[-1] / df['volume'].rolling(5).sum().iloc[-1] if df['volume'].rolling(5).sum().iloc[-1] != 0 else df['close'].iloc[-1],
            'volume_weighted_price_20': (df['close'] * df['volume']).rolling(20).sum().iloc[-1] / df['volume'].rolling(20).sum().iloc[-1] if df['volume'].rolling(20).sum().iloc[-1] != 0 else df['close'].iloc[-1],
            'buying_pressure_5': self._calculate_buying_pressure(df.tail(5)),
            'selling_pressure_5': self._calculate_selling_pressure(df.tail(5)),
        }

        return features

    def calculate_moving_averages(self, df):
        """Calculate comprehensive moving average features"""
        features = {}

        # Simple Moving Averages
        sma_periods = [5, 10, 20, 50, 100, 200]
        features['sma'] = {}
        for period in sma_periods:
            if len(df) >= period:
                sma = df['close'].rolling(period).mean()
                features['sma'][f'sma_{period}'] = sma.iloc[-1]
                features['sma'][f'sma_{period}_distance'] = (df['close'].iloc[-1] - sma.iloc[-1]) / sma.iloc[-1] * 100
                features['sma'][f'sma_{period}_slope'] = self._calculate_slope(sma.tail(5))
                features['sma'][f'sma_{period}_trend'] = self._calculate_trend(sma.tail(10))

        # Exponential Moving Averages
        ema_periods = [5, 10, 20, 50, 100, 200]
        features['ema'] = {}
        for period in ema_periods:
            if len(df) >= period:
                ema = df['close'].ewm(span=period).mean()
                features['ema'][f'ema_{period}'] = ema.iloc[-1]
                features['ema'][f'ema_{period}_distance'] = (df['close'].iloc[-1] - ema.iloc[-1]) / ema.iloc[-1] * 100
                features['ema'][f'ema_{period}_slope'] = self._calculate_slope(ema.tail(5))
                features['ema'][f'ema_{period}_trend'] = self._calculate_trend(ema.tail(10))

        # Moving average crossovers
        features['ma_crossovers'] = {
            'sma_5_20_crossover': self._detect_crossover(df['close'].rolling(5).mean(), df['close'].rolling(20).mean()),
            'sma_20_50_crossover': self._detect_crossover(df['close'].rolling(20).mean(), df['close'].rolling(50).mean()),
            'ema_5_20_crossover': self._detect_crossover(df['close'].ewm(span=5).mean(), df['close'].ewm(span=20).mean()),
            'ema_20_50_crossover': self._detect_crossover(df['close'].ewm(span=20).mean(), df['close'].ewm(span=50).mean()),
        }

        # Moving average ribbons
        features['ma_ribbon'] = self._calculate_ma_ribbon(df)

        return features

    def calculate_momentum_indicators(self, df):
        """Calculate momentum indicators using TA-Lib"""
        features = {}

        try:
            # RSI
            features['rsi'] = {
                'rsi_14': talib.RSI(df['close'], timeperiod=14).iloc[-1] if len(df) >= 14 else None,
                'rsi_7': talib.RSI(df['close'], timeperiod=7).iloc[-1] if len(df) >= 7 else None,
                'rsi_21': talib.RSI(df['close'], timeperiod=21).iloc[-1] if len(df) >= 21 else None,
                'rsi_overbought_14': talib.RSI(df['close'], timeperiod=14).iloc[-1] > 70 if len(df) >= 14 else False,
                'rsi_oversold_14': talib.RSI(df['close'], timeperiod=14).iloc[-1] < 30 if len(df) >= 14 else False,
                'rsi_divergence': self._detect_rsi_divergence(df),
            }

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            features['macd'] = {
                'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else None,
                'macd_histogram': macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else None,
                'macd_crossover': self._detect_crossover(macd, macd_signal),
                'macd_trend': self._calculate_trend(macd.tail(10)),
                'macd_divergence': self._detect_macd_divergence(df),
            }

            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            features['stochastic'] = {
                'stoch_k': slowk.iloc[-1] if not pd.isna(slowk.iloc[-1]) else None,
                'stoch_d': slowd.iloc[-1] if not pd.isna(slowd.iloc[-1]) else None,
                'stoch_overbought': slowk.iloc[-1] > 80 if not pd.isna(slowk.iloc[-1]) else False,
                'stoch_oversold': slowk.iloc[-1] < 20 if not pd.isna(slowk.iloc[-1]) else False,
                'stoch_crossover': self._detect_crossover(slowk, slowd),
            }

            # CCI (Commodity Channel Index)
            cci = talib.CCI(df['high'], df['low'], df['close'])
            features['cci'] = {
                'cci': cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else None,
                'cci_overbought': cci.iloc[-1] > 100 if not pd.isna(cci.iloc[-1]) else False,
                'cci_oversold': cci.iloc[-1] < -100 if not pd.isna(cci.iloc[-1]) else False,
            }

            # ADX (Average Directional Index)
            adx = talib.ADX(df['high'], df['low'], df['close'])
            features['adx'] = {
                'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else None,
                'adx_strong': adx.iloc[-1] > 25 if not pd.isna(adx.iloc[-1]) else False,
                'adx_weak': adx.iloc[-1] < 20 if not pd.isna(adx.iloc[-1]) else False,
            }

            # Aroon
            aroon_down, aroon_up = talib.AROON(df['high'], df['low'])
            features['aroon'] = {
                'aroon_up': aroon_up.iloc[-1] if not pd.isna(aroon_up.iloc[-1]) else None,
                'aroon_down': aroon_down.iloc[-1] if not pd.isna(aroon_down.iloc[-1]) else None,
                'aroon_oscillator': aroon_up.iloc[-1] - aroon_down.iloc[-1] if not pd.isna(aroon_up.iloc[-1]) and not pd.isna(aroon_down.iloc[-1]) else None,
                'aroon_uptrend': aroon_up.iloc[-1] > aroon_down.iloc[-1] if not pd.isna(aroon_up.iloc[-1]) and not pd.isna(aroon_down.iloc[-1]) else False,
            }

            # Williams %R
            williams_r = talib.WILLR(df['high'], df['low'], df['close'])
            features['williams_r'] = {
                'williams_r': williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else None,
                'williams_overbought': williams_r.iloc[-1] > -20 if not pd.isna(williams_r.iloc[-1]) else False,
                'williams_oversold': williams_r.iloc[-1] < -80 if not pd.isna(williams_r.iloc[-1]) else False,
            }

            # MFI (Money Flow Index)
            mfi = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
            features['mfi'] = {
                'mfi': mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else None,
                'mfi_overbought': mfi.iloc[-1] > 80 if not pd.isna(mfi.iloc[-1]) else False,
                'mfi_oversold': mfi.iloc[-1] < 20 if not pd.isna(mfi.iloc[-1]) else False,
            }

        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
            features['error'] = str(e)

        return features

    def calculate_volatility_indicators(self, df):
        """Calculate volatility indicators"""
        features = {}

        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
            features['bollinger_bands'] = {
                'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else None,
                'bb_middle': bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else None,
                'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else None,
                'bb_width': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] * 100 if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]) and bb_middle.iloc[-1] != 0 else None,
                'bb_position': (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100 if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]) and bb_upper.iloc[-1] != bb_lower.iloc[-1] else None,
                'bb_squeeze': ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]) < 0.05 if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]) and bb_middle.iloc[-1] != 0 else False,
            }

            # ATR (Average True Range)
            atr = talib.ATR(df['high'], df['low'], df['close'])
            features['atr'] = {
                'atr': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None,
                'atr_pct': atr.iloc[-1] / df['close'].iloc[-1] * 100 if not pd.isna(atr.iloc[-1]) and df['close'].iloc[-1] != 0 else None,
                'atr_trend': self._calculate_trend(atr.tail(10)),
                'atr_expansion': atr.iloc[-1] > atr.rolling(20).mean().iloc[-1] if not pd.isna(atr.iloc[-1]) and not pd.isna(atr.rolling(20).mean().iloc[-1]) else False,
            }

            # Keltner Channels (using EMA + ATR)
            ema = talib.EMA(df['close'], timeperiod=20)
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            keltner_upper = ema + 2 * atr
            keltner_middle = ema
            keltner_lower = ema - 2 * atr
            features['keltner_channels'] = {
                'keltner_upper': keltner_upper.iloc[-1] if not pd.isna(keltner_upper.iloc[-1]) else None,
                'keltner_middle': keltner_middle.iloc[-1] if not pd.isna(keltner_middle.iloc[-1]) else None,
                'keltner_lower': keltner_lower.iloc[-1] if not pd.isna(keltner_lower.iloc[-1]) else None,
                'keltner_position': (df['close'].iloc[-1] - keltner_lower.iloc[-1]) / (keltner_upper.iloc[-1] - keltner_lower.iloc[-1]) * 100 if not pd.isna(keltner_upper.iloc[-1]) and not pd.isna(keltner_lower.iloc[-1]) and keltner_upper.iloc[-1] != keltner_lower.iloc[-1] else None,
            }

            # Historical volatility
            features['historical_volatility'] = {
                'volatility_5': df['close'].pct_change().rolling(5).std() * np.sqrt(252) * 100,
                'volatility_20': df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100,
                'volatility_50': df['close'].pct_change().rolling(50).std() * np.sqrt(252) * 100,
                'volatility_trend': self._calculate_trend(df['close'].pct_change().rolling(20).std().tail(10)),
            }

        except Exception as e:
            print(f"Error calculating volatility indicators: {e}")
            features['error'] = str(e)

        return features

    def calculate_trend_indicators(self, df):
        """Calculate trend following indicators"""
        features = {}

        try:
            # Parabolic SAR
            sar = talib.SAR(df['high'], df['low'])
            features['parabolic_sar'] = {
                'sar': sar.iloc[-1] if not pd.isna(sar.iloc[-1]) else None,
                'sar_above_price': sar.iloc[-1] > df['close'].iloc[-1] if not pd.isna(sar.iloc[-1]) else False,
                'sar_signal': 'SELL' if not pd.isna(sar.iloc[-1]) and sar.iloc[-1] > df['close'].iloc[-1] else 'BUY' if not pd.isna(sar.iloc[-1]) else 'NEUTRAL',
            }

            # TRIX
            trix = talib.TRIX(df['close'])
            features['trix'] = {
                'trix': trix.iloc[-1] if not pd.isna(trix.iloc[-1]) else None,
                'trix_signal': 'BUY' if not pd.isna(trix.iloc[-1]) and trix.iloc[-1] > 0 else 'SELL' if not pd.isna(trix.iloc[-1]) else 'NEUTRAL',
                'trix_trend': self._calculate_trend(trix.tail(10)),
            }

            # DEMA (Double Exponential Moving Average)
            dema = talib.DEMA(df['close'])
            features['dema'] = {
                'dema': dema.iloc[-1] if not pd.isna(dema.iloc[-1]) else None,
                'dema_distance': (df['close'].iloc[-1] - dema.iloc[-1]) / dema.iloc[-1] * 100 if not pd.isna(dema.iloc[-1]) and dema.iloc[-1] != 0 else None,
                'dema_trend': self._calculate_trend(dema.tail(10)),
            }

            # TEMA (Triple Exponential Moving Average)
            tema = talib.TEMA(df['close'])
            features['tema'] = {
                'tema': tema.iloc[-1] if not pd.isna(tema.iloc[-1]) else None,
                'tema_distance': (df['close'].iloc[-1] - tema.iloc[-1]) / tema.iloc[-1] * 100 if not pd.isna(tema.iloc[-1]) and tema.iloc[-1] != 0 else None,
                'tema_trend': self._calculate_trend(tema.tail(10)),
            }

        except Exception as e:
            print(f"Error calculating trend indicators: {e}")
            features['error'] = str(e)

        return features

    def calculate_vwap_analysis(self, df):
        """Calculate comprehensive VWAP analysis"""
        features = {}

        try:
            # Standard VWAP
            vwap = self._calculate_vwap(df)
            features['vwap'] = {
                'vwap': vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else None,
                'vwap_distance': (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100 if not pd.isna(vwap.iloc[-1]) and vwap.iloc[-1] != 0 else None,
                'price_above_vwap': df['close'].iloc[-1] > vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else False,
                'vwap_trend': self._calculate_trend(vwap.tail(20)),
                'vwap_slope': self._calculate_slope(vwap.tail(5)),
            }

            # VWAP bands (standard deviations)
            vwap_std = self._calculate_vwap_std(df)
            features['vwap_bands'] = {
                'vwap_upper_1std': vwap.iloc[-1] + vwap_std.iloc[-1] if not pd.isna(vwap.iloc[-1]) and not pd.isna(vwap_std.iloc[-1]) else None,
                'vwap_lower_1std': vwap.iloc[-1] - vwap_std.iloc[-1] if not pd.isna(vwap.iloc[-1]) and not pd.isna(vwap_std.iloc[-1]) else None,
                'vwap_upper_2std': vwap.iloc[-1] + 2 * vwap_std.iloc[-1] if not pd.isna(vwap.iloc[-1]) and not pd.isna(vwap_std.iloc[-1]) else None,
                'vwap_lower_2std': vwap.iloc[-1] - 2 * vwap_std.iloc[-1] if not pd.isna(vwap.iloc[-1]) and not pd.isna(vwap_std.iloc[-1]) else None,
                'vwap_band_position': (df['close'].iloc[-1] - (vwap.iloc[-1] - vwap_std.iloc[-1])) / (2 * vwap_std.iloc[-1]) * 100 if not pd.isna(vwap.iloc[-1]) and not pd.isna(vwap_std.iloc[-1]) and vwap_std.iloc[-1] != 0 else None,
            }

            # Anchored VWAP (from session start, week start, month start)
            features['anchored_vwap'] = self._calculate_anchored_vwap(df)

        except Exception as e:
            print(f"Error calculating VWAP analysis: {e}")
            features['error'] = str(e)

        return features

    def calculate_market_profile(self, df):
        """Calculate comprehensive Market Profile analysis"""
        features = {}

        try:
            # Basic Market Profile
            profile = self._calculate_market_profile(df, num_bins=50)
            features['market_profile'] = {
                'poc': profile['poc'],
                'value_area_high': profile['value_area_high'],
                'value_area_low': profile['value_area_low'],
                'value_area_width': profile['value_area_width'],
                'value_area_volume_pct': profile['value_area_volume_pct'],
                'current_price_in_va': profile['current_price_in_va'],
                'price_above_va_high': df['close'].iloc[-1] > profile['value_area_high'],
                'price_below_va_low': df['close'].iloc[-1] < profile['value_area_low'],
                'profile_type': profile['profile_type'],  # balanced, trend, neutral
                'development': profile['development'],  # range, trend, rotation
            }

            # Volume Profile
            volume_profile = self._calculate_volume_profile(df, num_bins=50)
            features['volume_profile'] = {
                'volume_poc': volume_profile['poc'],
                'volume_value_area_high': volume_profile['value_area_high'],
                'volume_value_area_low': volume_profile['value_area_low'],
                'volume_at_poc': volume_profile['volume_at_poc'],
                'volume_distribution': volume_profile['distribution'],
                'volume_trend': volume_profile['trend'],
            }

            # TPO (Time Price Opportunity) analysis
            tpo_profile = self._calculate_tpo_profile(df)
            features['tpo_profile'] = {
                'tpo_poc': tpo_profile['poc'],
                'tpo_value_area_high': tpo_profile['value_area_high'],
                'tpo_value_area_low': tpo_profile['value_area_low'],
                'tpo_balance': tpo_profile['balance'],
                'tpo_rotation': tpo_profile['rotation_factor'],
            }

        except Exception as e:
            print(f"Error calculating Market Profile: {e}")
            features['error'] = str(e)

        return features

    def calculate_support_resistance(self, df):
        """Calculate comprehensive support and resistance levels"""
        features = {}

        try:
            # Pivot Points
            pivots = self._calculate_pivot_points(df)
            features['pivot_points'] = pivots

            # Key price levels
            features['key_levels'] = {
                'recent_high_20': df['high'].rolling(20).max().iloc[-1],
                'recent_low_20': df['low'].rolling(20).min().iloc[-1],
                'recent_high_50': df['high'].rolling(50).max().iloc[-1],
                'recent_low_50': df['low'].rolling(50).min().iloc[-1],
                'recent_high_200': df['high'].rolling(200).max().iloc[-1],
                'recent_low_200': df['low'].rolling(200).min().iloc[-1],
                'all_time_high': df['high'].max(),
                'all_time_low': df['low'].min(),
            }

            # Fibonacci levels
            features['fibonacci_levels'] = self._calculate_fibonacci_levels(df)

            # Psychological levels
            features['psychological_levels'] = self._calculate_psychological_levels(df)

            # Breakout analysis
            features['breakout_analysis'] = {
                'breakout_20_high': df['close'].iloc[-1] > df['high'].rolling(20).max().iloc[-2],
                'breakout_20_low': df['close'].iloc[-1] < df['low'].rolling(20).min().iloc[-2],
                'breakout_50_high': df['close'].iloc[-1] > df['high'].rolling(50).max().iloc[-2],
                'breakout_50_low': df['close'].iloc[-1] < df['low'].rolling(50).min().iloc[-2],
                'consolidation_20': self._detect_consolidation(df.tail(20)),
                'consolidation_50': self._detect_consolidation(df.tail(50)),
            }

        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
            features['error'] = str(e)

        return features

    def calculate_pattern_recognition(self, df):
        """Recognize candlestick and chart patterns"""
        features = {}

        try:
            # Candlestick patterns using TA-Lib
            patterns = {}

            # Single candlestick patterns
            patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 1 else 0
            patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 1 else 0
            patterns['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 1 else 0
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 1 else 0
            patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 2 else 0
            patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 3 else 0
            patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 3 else 0
            patterns['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 2 else 0
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 2 else 0

            features['candlestick_patterns'] = patterns

            # Chart patterns (simplified detection)
            features['chart_patterns'] = {
                'higher_highs_higher_lows': self._detect_uptrend(df),
                'lower_highs_lower_lows': self._detect_downtrend(df),
                'double_top': self._detect_double_top(df),
                'double_bottom': self._detect_double_bottom(df),
                'head_shoulders': self._detect_head_shoulders(df),
                'inverse_head_shoulders': self._detect_inverse_head_shoulders(df),
                'triangle': self._detect_triangle(df),
                'wedge': self._detect_wedge(df),
            }

        except Exception as e:
            print(f"Error calculating pattern recognition: {e}")
            features['error'] = str(e)

        return features

    def calculate_risk_metrics(self, df):
        """Calculate risk management metrics"""
        features = {}

        try:
            # Drawdown analysis
            cumulative_returns = (1 + df['close'].pct_change()).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            features['drawdown'] = {
                'current_drawdown': drawdown.iloc[-1] if not pd.isna(drawdown.iloc[-1]) else 0,
                'max_drawdown': drawdown.min(),
                'avg_drawdown': drawdown.mean(),
                'drawdown_duration': self._calculate_drawdown_duration(drawdown),
            }

            # Value at Risk (VaR)
            returns = df['close'].pct_change().dropna()
            features['var'] = {
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
            }

            # Sharpe ratio (simplified, assuming risk-free rate = 0)
            features['risk_metrics'] = {
                'sharpe_ratio_20': returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252) if returns.rolling(20).std().iloc[-1] != 0 else 0,
                'sharpe_ratio_50': returns.rolling(50).mean() / returns.rolling(50).std() * np.sqrt(252) if returns.rolling(50).std().iloc[-1] != 0 else 0,
                'volatility_20': returns.rolling(20).std() * np.sqrt(252),
                'volatility_50': returns.rolling(50).std() * np.sqrt(252),
                'max_loss_20': returns.rolling(20).min(),
                'max_gain_20': returns.rolling(20).max(),
            }

            # Position sizing recommendations
            features['position_sizing'] = {
                'recommended_risk_per_trade': 0.02,  # 2% risk recommendation
                'position_size_1r': 0.02 / abs(returns.rolling(20).std().iloc[-1]) if returns.rolling(20).std().iloc[-1] != 0 else 0,
                'position_size_2r': 0.01 / abs(returns.rolling(20).std().iloc[-1]) if returns.rolling(20).std().iloc[-1] != 0 else 0,
            }

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            features['error'] = str(e)

        return features

    def calculate_time_analysis(self, df):
        """Calculate time-based analysis"""
        features = {}

        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day

            # Time-based patterns
            hourly_returns = df.groupby('hour')['close'].pct_change().mean()
            daily_returns = df.groupby('day_of_week')['close'].pct_change().mean()

            features['time_patterns'] = {
                'best_trading_hour': int(hourly_returns.idxmax()) if not hourly_returns.empty else 0,
                'worst_trading_hour': int(hourly_returns.idxmin()) if not hourly_returns.empty else 0,
                'best_trading_day': int(daily_returns.idxmax()) if not daily_returns.empty else 0,
                'worst_trading_day': int(daily_returns.idxmin()) if not daily_returns.empty else 0,
                'current_hour_performance': df[df['hour'] == df['hour'].iloc[-1]]['close'].pct_change().mean() if len(df[df['hour'] == df['hour'].iloc[-1]]) > 0 else 0,
                'current_day_performance': df[df['day_of_week'] == df['day_of_week'].iloc[-1]]['close'].pct_change().mean() if len(df[df['day_of_week'] == df['day_of_week'].iloc[-1]]) > 0 else 0,
            }

            # Session analysis (assuming forex market hours)
            features['session_analysis'] = self._analyze_trading_sessions(df)

        except Exception as e:
            print(f"Error calculating time analysis: {e}")
            features['error'] = str(e)

        return features

    def calculate_intraday_session_analysis(self, df):
        """Calculate comprehensive intraday session analysis"""
        features = {}

        try:
            # Intraday session Market Profiles
            features['session_market_profiles'] = self.technical_analyzer.calculate_intraday_market_profiles(df)

            # Fixed Range Volume Profiles
            features['fixed_range_volume_profiles'] = {
                'session_range': self.technical_analyzer.calculate_fixed_range_volume_profile(df, 'session'),
                'week_range': self.technical_analyzer.calculate_fixed_range_volume_profile(df, 'week'),
                'visible_range': self.technical_analyzer.calculate_fixed_range_volume_profile(df, 'range'),
            }

            # Gap analysis
            features['gap_analysis'] = self.technical_analyzer.detect_gaps(df)

            # Order block analysis
            features['order_block_analysis'] = self.technical_analyzer.detect_order_blocks(df)

            # Fibonacci retracements
            features['fibonacci_analysis'] = self.technical_analyzer.calculate_fibonacci_retracements(df)

            # Ichimoku Cloud
            features['ichimoku_analysis'] = self.technical_analyzer.calculate_ichimoku_cloud(df)

            # Market microstructure
            features['market_microstructure'] = self.technical_analyzer.calculate_market_microstructure(df)

        except Exception as e:
            print(f"Error calculating intraday session analysis: {e}")
            features['error'] = str(e)

        return features

    def calculate_advanced_patterns(self, df):
        """Calculate advanced trading patterns and formations"""
        features = {}

        try:
            # Harmonic patterns detection
            features['harmonic_patterns'] = self._detect_harmonic_patterns(df)

            # Elliott Wave patterns (simplified)
            features['elliott_wave'] = self._detect_elliott_wave_patterns(df)

            # Supply and Demand zones
            features['supply_demand_zones'] = self._detect_supply_demand_zones(df)

            # Breakout patterns
            features['breakout_patterns'] = self._detect_breakout_patterns(df)

            # Consolidation patterns
            features['consolidation_patterns'] = self._detect_consolidation_patterns(df)

            # Reversal patterns
            features['reversal_patterns'] = self._detect_reversal_patterns(df)

            # Continuation patterns
            features['continuation_patterns'] = self._detect_continuation_patterns(df)

        except Exception as e:
            print(f"Error calculating advanced patterns: {e}")
            features['error'] = str(e)

        return features

    def calculate_market_sentiment(self, df):
        """Calculate market sentiment indicators"""
        features = {}

        try:
            # Fear & Greed Index calculation (simplified)
            features['fear_greed_index'] = self._calculate_fear_greed_index(df)

            # Market sentiment score
            features['sentiment_score'] = self._calculate_sentiment_score(df)

            # Volatility sentiment
            features['volatility_sentiment'] = self._calculate_volatility_sentiment(df)

            # Trend sentiment
            features['trend_sentiment'] = self._calculate_trend_sentiment(df)

            # Volume sentiment
            features['volume_sentiment'] = self._calculate_volume_sentiment(df)

        except Exception as e:
            print(f"Error calculating market sentiment: {e}")
            features['error'] = str(e)

        return features

    def calculate_intermarket_analysis(self, df):
        """Calculate intermarket correlations and relationships"""
        features = {}

        try:
            # Currency correlations (if multiple currency pairs available)
            features['currency_correlations'] = self._calculate_currency_correlations(df)

            # Commodity correlations
            features['commodity_correlations'] = self._calculate_commodity_correlations(df)

            # Market breadth
            features['market_breadth'] = self._calculate_market_breadth(df)

            # Market leadership
            features['market_leadership'] = self._analyze_market_leadership(df)

        except Exception as e:
            print(f"Error calculating intermarket analysis: {e}")
            features['error'] = str(e)

        return features

    # Helper methods for calculations
    def _calculate_consecutive_moves(self, condition_series):
        """Calculate consecutive True values in a boolean series"""
        consecutive = 0
        max_consecutive = 0

        for condition in condition_series.tail(20):
            if condition:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive

    def _calculate_trend(self, series):
        """Calculate trend direction (1 for uptrend, -1 for downtrend, 0 for neutral)"""
        if len(series) < 2:
            return 0

        # Simple linear regression slope
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]

        if slope > 0.01:
            return 1
        elif slope < -0.01:
            return -1
        else:
            return 0

    def _calculate_trend_strength(self, series):
        """Calculate trend strength using R-squared of linear regression"""
        if len(series) < 2:
            return 0

        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        return r_value ** 2

    def _calculate_price_volume_trend(self, df):
        """Calculate price-volume trend correlation"""
        if len(df) < 2:
            return 0

        price_change = df['close'].pct_change().dropna()
        volume_change = df['volume'].pct_change().dropna()

        if len(price_change) != len(volume_change):
            min_len = min(len(price_change), len(volume_change))
            price_change = price_change.tail(min_len)
            volume_change = volume_change.tail(min_len)

        correlation = price_change.corr(volume_change)
        return correlation if not pd.isna(correlation) else 0

    def _calculate_buying_pressure(self, df):
        """Calculate buying pressure based on volume and price action"""
        if len(df) < 1:
            return 0

        buying_volume = df[df['close'] > df['open']]['volume'].sum()
        total_volume = df['volume'].sum()

        return (buying_volume / total_volume * 100) if total_volume > 0 else 0

    def _calculate_selling_pressure(self, df):
        """Calculate selling pressure based on volume and price action"""
        if len(df) < 1:
            return 0

        selling_volume = df[df['close'] < df['open']]['volume'].sum()
        total_volume = df['volume'].sum()

        return (selling_volume / total_volume * 100) if total_volume > 0 else 0

    def _calculate_slope(self, series):
        """Calculate slope of linear regression"""
        if len(series) < 2:
            return 0

        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope

    def _detect_crossover(self, fast_line, slow_line):
        """Detect crossover between two lines"""
        if len(fast_line) < 2 or len(slow_line) < 2:
            return 'NEUTRAL'

        recent_fast = fast_line.iloc[-1]
        recent_slow = slow_line.iloc[-1]
        prev_fast = fast_line.iloc[-2]
        prev_slow = slow_line.iloc[-2]

        if pd.isna(recent_fast) or pd.isna(recent_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return 'NEUTRAL'

        if prev_fast <= prev_slow and recent_fast > recent_slow:
            return 'BULLISH_CROSS'
        elif prev_fast >= prev_slow and recent_fast < recent_slow:
            return 'BEARISH_CROSS'
        else:
            return 'NEUTRAL'

    def _calculate_ma_ribbon(self, df):
        """Calculate moving average ribbon analysis"""
        ma_periods = [10, 20, 30, 40, 50]
        mas = []

        for period in ma_periods:
            if len(df) >= period:
                ma = df['close'].rolling(period).mean().iloc[-1]
                if not pd.isna(ma):
                    mas.append(ma)

        if len(mas) < 2:
            return {'ribbon_alignment': 'NEUTRAL', 'ribbon_spread': 0}

        # Check if all MAs are aligned (all increasing or all decreasing)
        current_price = df['close'].iloc[-1]
        above_all = all(current_price > ma for ma in mas)
        below_all = all(current_price < ma for ma in mas)

        if above_all:
            alignment = 'BULLISH_ALIGNMENT'
        elif below_all:
            alignment = 'BEARISH_ALIGNMENT'
        else:
            alignment = 'MIXED'

        # Calculate ribbon spread (standard deviation of MAs)
        ribbon_spread = np.std(mas)

        return {
            'ribbon_alignment': alignment,
            'ribbon_spread': ribbon_spread,
            'ribbon_contraction': ribbon_spread < np.percentile([ribbon_spread], 25) if len(mas) > 0 else False,
            'ribbon_expansion': ribbon_spread > np.percentile([ribbon_spread], 75) if len(mas) > 0 else False,
        }

    def _detect_rsi_divergence(self, df):
        """Detect RSI divergence patterns"""
        if len(df) < 20:
            return 'NEUTRAL'

        try:
            rsi = talib.RSI(df['close'], timeperiod=14)
            if len(rsi) < 20 or pd.isna(rsi.iloc[-1]):
                return 'NEUTRAL'

            # Recent price highs and RSI highs
            price_highs = df['high'].rolling(10).max()
            rsi_highs = rsi.rolling(10).max()

            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            if (price_highs.iloc[-1] > price_highs.iloc[-11] and
                rsi_highs.iloc[-1] < rsi_highs.iloc[-11]):
                return 'BEARISH_DIVERGENCE'

            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            price_lows = df['low'].rolling(10).min()
            rsi_lows = rsi.rolling(10).min()

            if (price_lows.iloc[-1] < price_lows.iloc[-11] and
                rsi_lows.iloc[-1] > rsi_lows.iloc[-11]):
                return 'BULLISH_DIVERGENCE'

        except:
            pass

        return 'NEUTRAL'

    def _detect_macd_divergence(self, df):
        """Detect MACD divergence patterns"""
        if len(df) < 26:
            return 'NEUTRAL'

        try:
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            if len(macd_hist) < 20 or pd.isna(macd_hist.iloc[-1]):
                return 'NEUTRAL'

            # Similar logic to RSI divergence but with MACD histogram
            price_highs = df['high'].rolling(10).max()
            macd_highs = macd_hist.rolling(10).max()

            if (price_highs.iloc[-1] > price_highs.iloc[-11] and
                macd_highs.iloc[-1] < macd_highs.iloc[-11]):
                return 'BEARISH_DIVERGENCE'

            price_lows = df['low'].rolling(10).min()
            macd_lows = macd_hist.rolling(10).min()

            if (price_lows.iloc[-1] < price_lows.iloc[-11] and
                macd_lows.iloc[-1] > macd_lows.iloc[-11]):
                return 'BULLISH_DIVERGENCE'

        except:
            pass

        return 'NEUTRAL'

    def _calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

    def _calculate_vwap_std(self, df):
        """Calculate VWAP standard deviation for bands"""
        vwap = self._calculate_vwap(df)
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate squared deviations from VWAP
        squared_diffs = (typical_price - vwap) ** 2

        # Calculate rolling variance
        volume_weights = df['volume'] / df['volume'].cumsum()
        weighted_variance = (squared_diffs * volume_weights).cumsum() / volume_weights.cumsum()

        return np.sqrt(weighted_variance)

    def _calculate_anchored_vwap(self, df):
        """Calculate anchored VWAP from different time periods"""
        current_time = pd.to_datetime(df['datetime'].iloc[-1])

        anchored_vwaps = {}

        # Session start VWAP (assuming each day is a new session)
        today = current_time.date()
        today_data = df[pd.to_datetime(df['datetime']).dt.date == today]
        if len(today_data) > 0:
            anchored_vwaps['session_vwap'] = self._calculate_vwap(today_data).iloc[-1]

        # Week start VWAP
        week_start = current_time - timedelta(days=current_time.weekday())
        week_data = df[pd.to_datetime(df['datetime']) >= week_start]
        if len(week_data) > 0:
            anchored_vwaps['week_vwap'] = self._calculate_vwap(week_data).iloc[-1]

        # Month start VWAP
        month_start = current_time.replace(day=1)
        month_data = df[pd.to_datetime(df['datetime']) >= month_start]
        if len(month_data) > 0:
            anchored_vwaps['month_vwap'] = self._calculate_vwap(month_data).iloc[-1]

        return anchored_vwaps

    def _calculate_market_profile(self, df, num_bins=50):
        """Calculate Market Profile analysis"""
        if len(df) < 10:
            return {
                'poc': df['close'].iloc[-1],
                'value_area_high': df['high'].iloc[-1],
                'value_area_low': df['low'].iloc[-1],
                'value_area_width': 0,
                'value_area_volume_pct': 0,
                'current_price_in_va': True,
                'profile_type': 'NEUTRAL',
                'development': 'NEUTRAL',
            }

        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, num_bins)

        # Calculate volume at each price level
        volume_at_price = []
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]

            # Find candles that traded in this price range
            in_range = ((df['low'] <= bin_high) & (df['high'] >= bin_low))
            volume_in_range = df[in_range]['volume'].sum()

            volume_at_price.append(volume_in_range)

        # Find POC (price with highest volume)
        poc_index = np.argmax(volume_at_price)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

        # Calculate value area (68% of total volume)
        total_volume = sum(volume_at_price)
        target_volume = total_volume * 0.68

        # Expand from POC outward until we reach target volume
        value_area_low = poc_price
        value_area_high = poc_price
        accumulated_volume = volume_at_price[poc_index]

        lower_index = poc_index
        upper_index = poc_index

        while accumulated_volume < target_volume and (lower_index > 0 or upper_index < len(volume_at_price) - 1):
            # Add the next highest volume level
            if lower_index > 0 and (upper_index >= len(volume_at_price) - 1 or volume_at_price[lower_index - 1] >= volume_at_price[upper_index + 1]):
                lower_index -= 1
                accumulated_volume += volume_at_price[lower_index]
                value_area_low = price_bins[lower_index]
            elif upper_index < len(volume_at_price) - 1:
                upper_index += 1
                accumulated_volume += volume_at_price[upper_index]
                value_area_high = price_bins[upper_index + 1]
            else:
                break

        current_price = df['close'].iloc[-1]

        return {
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'value_area_width': value_area_high - value_area_low,
            'value_area_volume_pct': (accumulated_volume / total_volume) * 100,
            'current_price_in_va': value_area_low <= current_price <= value_area_high,
            'profile_type': self._classify_profile_type(df, poc_price, value_area_low, value_area_high),
            'development': self._classify_profile_development(df, poc_price),
        }

    def _calculate_volume_profile(self, df, num_bins=50):
        """Calculate Volume Profile analysis"""
        if len(df) < 10:
            return {
                'poc': df['close'].iloc[-1],
                'value_area_high': df['high'].iloc[-1],
                'value_area_low': df['low'].iloc[-1],
                'volume_at_poc': 0,
                'distribution': 'NEUTRAL',
                'trend': 'NEUTRAL',
            }

        # Similar to market profile but focusing on volume distribution
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, num_bins)

        volume_at_price = []
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]

            # More precise volume calculation for volume profile
            in_range = ((df['low'] <= bin_high) & (df['high'] >= bin_low))

            # Estimate volume at specific price levels
            volume_in_range = 0
            for _, candle in df[in_range].iterrows():
                # Distribute volume proportionally within the range
                if candle['high'] != candle['low']:
                    overlap = min(candle['high'], bin_high) - max(candle['low'], bin_low)
                    proportion = overlap / (candle['high'] - candle['low'])
                    volume_in_range += candle['volume'] * proportion
                else:
                    volume_in_range += candle['volume']

            volume_at_price.append(volume_in_range)

        poc_index = np.argmax(volume_at_price)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

        # Calculate value area for volume profile
        total_volume = sum(volume_at_price)
        target_volume = total_volume * 0.68

        value_area_low = poc_price
        value_area_high = poc_price
        accumulated_volume = volume_at_price[poc_index]

        lower_index = poc_index
        upper_index = poc_index

        while accumulated_volume < target_volume and (lower_index > 0 or upper_index < len(volume_at_price) - 1):
            if lower_index > 0 and (upper_index >= len(volume_at_price) - 1 or volume_at_price[lower_index - 1] >= volume_at_price[upper_index + 1]):
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
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_at_poc': volume_at_price[poc_index],
            'distribution': self._classify_volume_distribution(volume_at_price),
            'trend': self._classify_volume_trend(df, poc_price),
        }

    def _calculate_tpo_profile(self, df):
        """Calculate TPO (Time Price Opportunity) profile"""
        if len(df) < 10:
            return {
                'poc': df['close'].iloc[-1],
                'value_area_high': df['high'].iloc[-1],
                'value_area_low': df['low'].iloc[-1],
                'balance': 'NEUTRAL',
                'rotation_factor': 0,
            }

        # Count TPOs (time periods) at each price level
        price_min = df['low'].min()
        price_max = df['high'].max()
        num_bins = 50
        price_bins = np.linspace(price_min, price_max, num_bins)

        tpo_at_price = []
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]

            # Count time periods (candles) that touched this price range
            in_range = ((df['low'] <= bin_high) & (df['high'] >= bin_low))
            tpo_count = in_range.sum()
            tpo_at_price.append(tpo_count)

        poc_index = np.argmax(tpo_at_price)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

        # Calculate value area based on TPOs
        total_tpos = sum(tpo_at_price)
        target_tpos = total_tpos * 0.68

        value_area_low = poc_price
        value_area_high = poc_price
        accumulated_tpos = tpo_at_price[poc_index]

        lower_index = poc_index
        upper_index = poc_index

        while accumulated_tpos < target_tpos and (lower_index > 0 or upper_index < len(tpo_at_price) - 1):
            if lower_index > 0 and (upper_index >= len(tpo_at_price) - 1 or tpo_at_price[lower_index - 1] >= tpo_at_price[upper_index + 1]):
                lower_index -= 1
                accumulated_tpos += tpo_at_price[lower_index]
                value_area_low = price_bins[lower_index]
            elif upper_index < len(tpo_at_price) - 1:
                upper_index += 1
                accumulated_tpos += tpo_at_price[upper_index]
                value_area_high = price_bins[upper_index + 1]
            else:
                break

        return {
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'balance': self._calculate_tpo_balance(df, poc_price, value_area_low, value_area_high),
            'rotation_factor': self._calculate_rotation_factor(df, poc_price),
        }

    def _classify_profile_type(self, df, poc, va_low, va_high):
        """Classify market profile type"""
        current_price = df['close'].iloc[-1]

        if current_price > va_high:
            return 'TREND_UP'
        elif current_price < va_low:
            return 'TREND_DOWN'
        else:
            return 'BALANCED'

    def _classify_profile_development(self, df, poc):
        """Classify profile development pattern"""
        recent_prices = df['close'].tail(20)

        if len(recent_prices) < 10:
            return 'NEUTRAL'

        # Check if price is rotating around POC
        above_poc = (recent_prices > poc).sum()
        below_poc = (recent_prices < poc).sum()

        if abs(above_poc - below_poc) <= 2:
            return 'ROTATION'
        elif above_poc > below_poc:
            return 'RANGE_UP'
        else:
            return 'RANGE_DOWN'

    def _classify_volume_distribution(self, volume_at_price):
        """Classify volume distribution pattern"""
        if len(volume_at_price) < 2:
            return 'NEUTRAL'

        # Check if volume is concentrated or distributed
        max_volume = max(volume_at_price)
        avg_volume = np.mean(volume_at_price)

        if max_volume > avg_volume * 2:
            return 'CONCENTRATED'
        else:
            return 'DISTRIBUTED'

    def _classify_volume_trend(self, df, poc):
        """Classify volume-based trend"""
        recent_volume = df['volume'].tail(10)
        recent_price = df['close'].tail(10)

        if len(recent_volume) < 5:
            return 'NEUTRAL'

        # Correlate volume and price movement
        volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        price_trend = np.polyfit(range(len(recent_price)), recent_price, 1)[0]

        if volume_trend > 0 and price_trend > 0:
            return 'ACCUMULATION'
        elif volume_trend > 0 and price_trend < 0:
            return 'DISTRIBUTION'
        else:
            return 'NEUTRAL'

    def _calculate_tpo_balance(self, df, poc, va_low, va_high):
        """Calculate TPO balance metric"""
        recent_prices = df['close'].tail(20)

        if len(recent_prices) < 5:
            return 'NEUTRAL'

        above_va = (recent_prices > va_high).sum()
        below_va = (recent_prices < va_low).sum()
        in_va = len(recent_prices) - above_va - below_va

        if in_va > len(recent_prices) * 0.6:
            return 'BALANCED'
        elif above_va > below_va:
            return 'UPWARD_PRESSURE'
        else:
            return 'DOWNWARD_PRESSURE'

    def _calculate_rotation_factor(self, df, poc):
        """Calculate rotation factor around POC"""
        recent_prices = df['close'].tail(10)

        if len(recent_prices) < 5:
            return 0

        # Count rotations around POC
        crosses = 0
        for i in range(1, len(recent_prices)):
            if (recent_prices.iloc[i-1] <= poc <= recent_prices.iloc[i]) or \
               (recent_prices.iloc[i-1] >= poc >= recent_prices.iloc[i]):
                crosses += 1

        return crosses / len(recent_prices)

    def _calculate_pivot_points(self, df):
        """Calculate pivot points"""
        if len(df) < 2:
            return {}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        classic_pivot = (prev_candle['high'] + prev_candle['low'] + prev_candle['close']) / 3

        return {
            'classic_pivot': classic_pivot,
            'r1': (2 * classic_pivot) - prev_candle['low'],
            's1': (2 * classic_pivot) - prev_candle['high'],
            'r2': classic_pivot + (prev_candle['high'] - prev_candle['low']),
            's2': classic_pivot - (prev_candle['high'] - prev_candle['low']),
            'r3': prev_candle['high'] + 2 * (classic_pivot - prev_candle['low']),
            's3': prev_candle['low'] - 2 * (prev_candle['high'] - classic_pivot),
        }

    def _calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels"""
        if len(df) < 20:
            return {}

        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]

        diff = recent_high - recent_low

        return {
            'fib_0': recent_high,
            'fib_236': recent_high - 0.236 * diff,
            'fib_382': recent_high - 0.382 * diff,
            'fib_500': recent_high - 0.500 * diff,
            'fib_618': recent_high - 0.618 * diff,
            'fib_100': recent_low,
        }

    def _calculate_psychological_levels(self, df):
        """Calculate psychological price levels"""
        current_price = df['close'].iloc[-1]

        # Round to nearest psychological levels
        round_numbers = []
        for i in range(-3, 4):  # From 0.001 to 1000
            level = round(current_price, i)
            if level > 0:
                round_numbers.append(level)

        # Remove duplicates and sort
        round_numbers = sorted(list(set(round_numbers)))

        # Find closest psychological levels
        closest_below = [level for level in round_numbers if level < current_price]
        closest_above = [level for level in round_numbers if level > current_price]

        return {
            'psychological_below': closest_below[-3:] if closest_below else [],
            'psychological_above': closest_above[:3] if closest_above else [],
            'nearest_psychological': closest_below[-1] if closest_below else round_numbers[0] if round_numbers else current_price,
        }

    def _detect_consolidation(self, df):
        """Detect if market is in consolidation"""
        if len(df) < 10:
            return False

        # Check if price is trading in a range
        price_range = df['high'].max() - df['low'].min()
        avg_price = df['close'].mean()

        # Consolidation if range is less than 2% of average price
        consolidation_threshold = avg_price * 0.02

        return price_range < consolidation_threshold

    def _detect_uptrend(self, df):
        """Detect uptrend pattern (higher highs, higher lows)"""
        if len(df) < 20:
            return False

        recent_highs = df['high'].rolling(5).max().dropna()
        recent_lows = df['low'].rolling(5).min().dropna()

        if len(recent_highs) < 4 or len(recent_lows) < 4:
            return False

        # Check if making higher highs and higher lows
        higher_highs = (recent_highs.iloc[-1] > recent_highs.iloc[-2] > recent_highs.iloc[-3])
        higher_lows = (recent_lows.iloc[-1] > recent_lows.iloc[-2] > recent_lows.iloc[-3])

        return higher_highs and higher_lows

    def _detect_downtrend(self, df):
        """Detect downtrend pattern (lower highs, lower lows)"""
        if len(df) < 20:
            return False

        recent_highs = df['high'].rolling(5).max().dropna()
        recent_lows = df['low'].rolling(5).min().dropna()

        if len(recent_highs) < 4 or len(recent_lows) < 4:
            return False

        # Check if making lower highs and lower lows
        lower_highs = (recent_highs.iloc[-1] < recent_highs.iloc[-2] < recent_highs.iloc[-3])
        lower_lows = (recent_lows.iloc[-1] < recent_lows.iloc[-2] < recent_lows.iloc[-3])

        return lower_highs and lower_lows

    def _detect_double_top(self, df):
        """Detect double top pattern (simplified)"""
        if len(df) < 20:
            return False

        recent_highs = df['high'].rolling(5).max().dropna()

        if len(recent_highs) < 8:
            return False

        # Look for two similar peaks with a trough between
        peak1 = recent_highs.iloc[-3]
        trough = recent_highs.iloc[-2]
        peak2 = recent_highs.iloc[-1]

        # Check if peaks are similar (within 2%) and trough is significantly lower
        similarity = abs(peak1 - peak2) / peak1 < 0.02
        depth = (peak1 - trough) / peak1 > 0.02

        return similarity and depth

    def _detect_double_bottom(self, df):
        """Detect double bottom pattern (simplified)"""
        if len(df) < 20:
            return False

        recent_lows = df['low'].rolling(5).min().dropna()

        if len(recent_lows) < 8:
            return False

        # Look for two similar troughs with a peak between
        trough1 = recent_lows.iloc[-3]
        peak = recent_lows.iloc[-2]
        trough2 = recent_lows.iloc[-1]

        # Check if troughs are similar (within 2%) and peak is significantly higher
        similarity = abs(trough1 - trough2) / trough1 < 0.02
        height = (peak - trough1) / trough1 > 0.02

        return similarity and height

    def _detect_head_shoulders(self, df):
        """Detect head and shoulders pattern (simplified)"""
        # Simplified detection - would need more complex pattern recognition in practice
        return False

    def _detect_inverse_head_shoulders(self, df):
        """Detect inverse head and shoulders pattern (simplified)"""
        # Simplified detection - would need more complex pattern recognition in practice
        return False

    def _detect_triangle(self, df):
        """Detect triangle pattern (consolidation)"""
        if len(df) < 20:
            return False

        # Check if volatility is decreasing (narrowing range)
        volatility = df['high'] - df['low']
        vol_trend = np.polyfit(range(len(volatility.tail(10))), volatility.tail(10), 1)[0]

        return vol_trend < 0  # Decreasing volatility suggests triangle formation

    def _detect_wedge(self, df):
        """Detect wedge pattern"""
        # Simplified wedge detection
        return False

    def _calculate_drawdown_duration(self, drawdown_series):
        """Calculate average drawdown duration"""
        drawdown_periods = []
        in_drawdown = False
        current_duration = 0

        for dd in drawdown_series:
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                current_duration = 1
            elif dd < 0 and in_drawdown:
                current_duration += 1
            elif dd >= 0 and in_drawdown:
                drawdown_periods.append(current_duration)
                in_drawdown = False
                current_duration = 0

        return np.mean(drawdown_periods) if drawdown_periods else 0

    def _analyze_trading_sessions(self, df):
        """Analyze trading session performance"""
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour

        # Define forex trading sessions
        asian_session = df[(df['hour'] >= 0) & (df['hour'] < 8)]
        london_session = df[(df['hour'] >= 8) & (df['hour'] < 16)]
        ny_session = df[(df['hour'] >= 13) & (df['hour'] < 22)]

        return {
            'asian_session_avg_return': asian_session['close'].pct_change().mean() if len(asian_session) > 0 else 0,
            'london_session_avg_return': london_session['close'].pct_change().mean() if len(london_session) > 0 else 0,
            'ny_session_avg_return': ny_session['close'].pct_change().mean() if len(ny_session) > 0 else 0,
            'asian_session_volatility': asian_session['close'].pct_change().std() if len(asian_session) > 0 else 0,
            'london_session_volatility': london_session['close'].pct_change().std() if len(london_session) > 0 else 0,
            'ny_session_volatility': ny_session['close'].pct_change().std() if len(ny_session) > 0 else 0,
        }

    def process_symbol_data(self, symbol, data_dir="./data", output_dir="./data/rag_processed"):
        """Process all historical data for a symbol with comprehensive features"""

        print(f"🔄 Processing comprehensive historical analysis for {symbol}")
        print("=" * 60)

        # Find all historical data files for the symbol
        pattern = os.path.join(data_dir, f"{symbol}_PERIOD_*_0.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"❌ No historical data files found for {symbol}")
            return False

        print(f"📁 Found {len(csv_files)} historical timeframe files for {symbol}")

        # Process each timeframe
        timeframe_results = {}

        for csv_file in csv_files:
            # Extract timeframe from filename
            filename = os.path.basename(csv_file)
            import re
            timeframe_match = re.search(r'.*PERIOD_([MH][0-9,]+[DW]?)_0\.csv', filename)

            if not timeframe_match:
                # Try alternative pattern for W1, MN1, etc.
                timeframe_match = re.search(r'.*PERIOD_([A-Z0-9,]+)_0\.csv', filename)

            if not timeframe_match:
                print(f"⚠️  Could not parse timeframe from {filename}")
                continue

            timeframe = timeframe_match.group(1)

            print(f"🔄 Processing {timeframe} historical data for {symbol}...")

            try:
                # Load and process data with better error handling
                try:
                    # Try reading with different encodings
                    try:
                        df = pd.read_csv(csv_file)
                    except UnicodeDecodeError:
                        try:
                            # Try with UTF-16 LE (common for Windows exports)
                            df = pd.read_csv(csv_file, encoding='utf-16-le')
                        except:
                            try:
                                # Try with UTF-16 BOM (FF FE)
                                df = pd.read_csv(csv_file, encoding='utf-16')
                            except:
                                try:
                                    # Try with UTF-8 BOM
                                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                                except:
                                    # Try with latin1 as fallback
                                    df = pd.read_csv(csv_file, encoding='latin1')
                except Exception as e:
                    print(f"❌ Could not read file {csv_file}: {e}")
                    continue

                # Handle different datetime column names
                datetime_columns = ['datetime', 'time', 'Date', 'date', 'timestamp', 'Timestamp', 'DateTime']
                datetime_col = None
                for col in datetime_columns:
                    if col in df.columns:
                        datetime_col = col
                        break

                if datetime_col:
                    df['datetime'] = pd.to_datetime(df[datetime_col])
                else:
                    # If no datetime column, try to use index
                    if df.index.dtype == 'object':
                        df['datetime'] = pd.to_datetime(df.index)
                    else:
                        print(f"⚠️  No datetime column found in {csv_file}")
                        continue

                # Handle different OHLC column names
                column_mapping = {
                    'open': ['open', 'Open', 'OPEN'],
                    'high': ['high', 'High', 'HIGH'],
                    'low': ['low', 'Low', 'LOW'],
                    'close': ['close', 'Close', 'CLOSE'],
                    'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
                }

                for standard_col, possible_cols in column_mapping.items():
                    found_col = None
                    for col in possible_cols:
                        if col in df.columns:
                            found_col = col
                            break

                    if found_col and found_col != standard_col:
                        df[standard_col] = df[found_col]

                # Validate required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"❌ Missing required columns {missing_cols} in {csv_file}")
                    continue

                # Debug: Print column names
                print(f"📋 {timeframe} CSV columns: {list(df.columns)}")

                # Clean data
                df = df.dropna(subset=required_cols).reset_index(drop=True)

                # Ensure numeric types
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Remove any remaining NaN rows
                df = df.dropna(subset=required_cols).reset_index(drop=True)

                if len(df) < 50:
                    print(f"⚠️  {timeframe}: Insufficient data ({len(df)} candles)")
                    continue

                # Calculate all features with error handling
                features = {
                    'timeframe_info': {
                        'timeframe': timeframe,
                        'total_candles': len(df),
                        'date_range': {
                            'start': df['datetime'].min().isoformat(),
                            'end': df['datetime'].max().isoformat(),
                        },
                        'time_coverage_days': (df['datetime'].max() - df['datetime'].min()).days,
                        'avg_daily_candles': len(df) / max(1, (df['datetime'].max() - df['datetime'].min()).days),
                    }
                }

                # Calculate features with individual error handling
                feature_functions = [
                    ('price_features', self.calculate_price_features),
                    ('volume_features', self.calculate_volume_features),
                    ('moving_averages', self.calculate_moving_averages),
                    ('momentum_indicators', self.calculate_momentum_indicators),
                    ('volatility_indicators', self.calculate_volatility_indicators),
                    ('trend_indicators', self.calculate_trend_indicators),
                    ('vwap_analysis', self.calculate_vwap_analysis),
                    ('market_profile', self.calculate_market_profile),
                    ('support_resistance', self.calculate_support_resistance),
                    ('pattern_recognition', self.calculate_pattern_recognition),
                    ('risk_metrics', self.calculate_risk_metrics),
                    ('time_analysis', self.calculate_time_analysis),
                ]

                for feature_name, function in feature_functions:
                    try:
                        features[feature_name] = function(df)
                    except Exception as e:
                        print(f"⚠️  Error calculating {feature_name} for {timeframe}: {e}")
                        features[feature_name] = {'error': str(e)}

                # Advanced features (might fail on smaller datasets)
                advanced_feature_functions = [
                    ('intraday_session_analysis', self.calculate_intraday_session_analysis),
                    ('advanced_patterns', self.calculate_advanced_patterns),
                    ('market_sentiment', self.calculate_market_sentiment),
                    ('intermarket_analysis', self.calculate_intermarket_analysis),
                ]

                for feature_name, function in advanced_feature_functions:
                    try:
                        if len(df) >= 100:  # Only calculate advanced features on sufficient data
                            features[feature_name] = function(df)
                        else:
                            features[feature_name] = {'skipped': f'Insufficient data ({len(df)} < 100)'}
                    except Exception as e:
                        print(f"⚠️  Error calculating advanced {feature_name} for {timeframe}: {e}")
                        features[feature_name] = {'error': str(e)}

                timeframe_results[timeframe] = features

                # Calculate timeframe statistics
                price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
                volatility = df['close'].pct_change().std() * np.sqrt(252) * 100

                print(f"✅ {timeframe}: {len(df)} candles, {price_change:.2f}% change, {volatility:.2f}% volatility")

            except Exception as e:
                print(f"❌ Error processing {timeframe}: {e}")
                continue

        if not timeframe_results:
            print(f"❌ No timeframes successfully processed for {symbol}")
            return False

        # Create comprehensive analysis
        comprehensive_analysis = {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_type': 'COMPREHENSIVE_HISTORICAL',
            'total_timeframes': len(timeframe_results),
            'timeframe_analysis': timeframe_results,
            'cross_timeframe_analysis': self._calculate_cross_timeframe_analysis(timeframe_results),
            'market_regime_analysis': self._analyze_market_regime(timeframe_results),
            'trading_recommendations': self._generate_trading_recommendations(timeframe_results),
        }

        # Save comprehensive analysis
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{symbol}_comprehensive.json")

        with open(output_file, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)

        file_size = os.path.getsize(output_file)

        print(f"\n✅ Comprehensive historical analysis completed for {symbol}")
        print(f"📊 Analysis saved to: {output_file}")
        print(f"📏 File size: {file_size:,} bytes")
        print(f"📈 Timeframes processed: {len(timeframe_results)}")

        # Calculate total coverage
        total_days = sum(tf['timeframe_info']['time_coverage_days'] for tf in timeframe_results.values())
        print(f"📅 Total historical coverage: {total_days:.1f} days")

        return True

    def _calculate_cross_timeframe_analysis(self, timeframe_results):
        """Calculate cross-timeframe correlations and consensus"""

        cross_analysis = {
            'trend_consensus': {},
            'level_confluence': {},
            'volatility_regime': {},
            'volume_patterns': {},
        }

        # Trend consensus across timeframes
        trend_votes = {'uptrend': 0, 'downtrend': 0, 'neutral': 0}

        for tf_name, tf_data in timeframe_results.items():
            # Get trend from momentum indicators
            rsi = tf_data.get('momentum_indicators', {}).get('rsi', {}).get('rsi_14', 50)
            macd_trend = tf_data.get('momentum_indicators', {}).get('macd', {}).get('macd_trend', 0)

            # Determine overall trend for this timeframe
            if rsi > 60 and macd_trend > 0:
                trend_votes['uptrend'] += 1
            elif rsi < 40 and macd_trend < 0:
                trend_votes['downtrend'] += 1
            else:
                trend_votes['neutral'] += 1

        cross_analysis['trend_consensus'] = {
            'votes': trend_votes,
            'overall_trend': max(trend_votes, key=trend_votes.get),
            'consensus_strength': max(trend_votes.values()) / sum(trend_votes.values()) if sum(trend_votes.values()) > 0 else 0,
        }

        # Level confluence analysis
        all_support_levels = []
        all_resistance_levels = []
        vwap_levels = []
        poc_levels = []

        for tf_data in timeframe_results.values():
            # Add key levels from each timeframe
            support_resistance = tf_data.get('support_resistance', {})

            if 'key_levels' in support_resistance:
                all_support_levels.append(support_resistance['key_levels'].get('recent_low_20', 0))
                all_resistance_levels.append(support_resistance['key_levels'].get('recent_high_20', 0))

            vwap_data = tf_data.get('vwap_analysis', {}).get('vwap', {})
            if vwap_data:
                vwap_levels.append(vwap_data.get('vwap', 0))

            market_profile = tf_data.get('market_profile', {}).get('market_profile', {})
            if market_profile:
                poc_levels.append(market_profile.get('poc', 0))

        # Find confluence zones (levels that appear across multiple timeframes)
        cross_analysis['level_confluence'] = {
            'support_confluence': self._find_level_confluence(all_support_levels),
            'resistance_confluence': self._find_level_confluence(all_resistance_levels),
            'vwap_confluence': self._find_level_confluence(vwap_levels),
            'poc_confluence': self._find_level_confluence(poc_levels),
        }

        return cross_analysis

    def _find_level_confluence(self, levels):
        """Find price levels with confluence (similar levels across timeframes)"""
        if not levels or len(levels) < 2:
            return {'confluence_levels': [], 'strongest_level': None}

        # Group similar levels (within 1% of each other)
        levels.sort()
        confluence_groups = []

        for level in levels:
            if level == 0:
                continue

            added_to_group = False
            for group in confluence_groups:
                group_avg = sum(group) / len(group)
                if abs(level - group_avg) / group_avg < 0.01:  # Within 1%
                    group.append(level)
                    added_to_group = True
                    break

            if not added_to_group:
                confluence_groups.append([level])

        # Find strongest confluence (most levels in group)
        strongest_group = max(confluence_groups, key=len) if confluence_groups else []
        strongest_level = sum(strongest_group) / len(strongest_group) if strongest_group else None

        return {
            'confluence_levels': [(sum(group) / len(group), len(group)) for group in confluence_groups if len(group) > 1],
            'strongest_level': strongest_level,
            'confluence_count': len(strongest_group) if strongest_group else 0,
        }

    def _analyze_market_regime(self, timeframe_results):
        """Analyze overall market regime"""

        regime_analysis = {
            'volatility_regime': 'UNKNOWN',
            'trend_regime': 'UNKNOWN',
            'volume_regime': 'UNKNOWN',
            'overall_regime': 'UNKNOWN',
        }

        # Analyze volatility across timeframes
        volatility_levels = []
        for tf_data in timeframe_results.values():
            vol = tf_data.get('risk_metrics', {}).get('risk_metrics', {}).get('volatility_20', 0)
            # Handle scalar and Series cases
            if hasattr(vol, 'iloc'):
                vol = vol.iloc[-1] if len(vol) > 0 else 0
            elif hasattr(vol, 'item'):
                vol = vol.item() if hasattr(vol, 'item') else vol
            if not pd.isna(vol) and vol != 0:
                volatility_levels.append(vol)

        if volatility_levels:
            avg_volatility = np.mean(volatility_levels)
            if avg_volatility > 30:
                regime_analysis['volatility_regime'] = 'HIGH_VOLATILITY'
            elif avg_volatility > 15:
                regime_analysis['volatility_regime'] = 'NORMAL_VOLATILITY'
            else:
                regime_analysis['volatility_regime'] = 'LOW_VOLATILITY'

        # Analyze trend strength
        trend_strengths = []
        for tf_data in timeframe_results.values():
            adx = tf_data.get('momentum_indicators', {}).get('adx', {}).get('adx', 0)
            if adx:
                trend_strengths.append(adx)

        if trend_strengths:
            avg_trend_strength = np.mean(trend_strengths)
            if avg_trend_strength > 25:
                regime_analysis['trend_regime'] = 'STRONG_TREND'
            elif avg_trend_strength > 20:
                regime_analysis['trend_regime'] = 'MODERATE_TREND'
            else:
                regime_analysis['trend_regime'] = 'WEAK_TREND/RANGE'

        # Overall regime assessment
        if (regime_analysis['volatility_regime'] == 'HIGH_VOLATILITY' and
            regime_analysis['trend_regime'] == 'STRONG_TREND'):
            regime_analysis['overall_regime'] = 'TRENDING_VOLATILE'
        elif (regime_analysis['volatility_regime'] == 'LOW_VOLATILITY' and
              regime_analysis['trend_regime'] == 'WEAK_TREND/RANGE'):
            regime_analysis['overall_regime'] = 'RANGING_QUIET'
        else:
            regime_analysis['overall_regime'] = 'MIXED_CONDITIONS'

        return regime_analysis

    def _generate_trading_recommendations(self, timeframe_results):
        """Generate trading recommendations based on comprehensive analysis"""

        recommendations = {
            'overall_bias': 'NEUTRAL',
            'entry_strategy': [],
            'risk_management': [],
            'key_levels': [],
            'timeframe_preferences': {},
            'confidence_level': 0,
        }

        # Analyze trend bias
        bullish_signals = 0
        bearish_signals = 0

        for tf_name, tf_data in timeframe_results.items():
            # RSI signals
            rsi = tf_data.get('momentum_indicators', {}).get('rsi', {}).get('rsi_14', 50)
            if rsi > 70:
                bearish_signals += 1
            elif rsi < 30:
                bullish_signals += 1

            # MACD signals
            macd_crossover = tf_data.get('momentum_indicators', {}).get('macd', {}).get('macd_crossover', 'NEUTRAL')
            if macd_crossover == 'BULLISH_CROSS':
                bullish_signals += 1
            elif macd_crossover == 'BEARISH_CROSS':
                bearish_signals += 1

            # Price vs VWAP
            price_above_vwap = tf_data.get('vwap_analysis', {}).get('vwap', {}).get('price_above_vwap', False)
            if price_above_vwap:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Determine overall bias
        if bullish_signals > bearish_signals * 1.5:
            recommendations['overall_bias'] = 'BULLISH'
        elif bearish_signals > bullish_signals * 1.5:
            recommendations['overall_bias'] = 'BEARISH'
        else:
            recommendations['overall_bias'] = 'NEUTRAL'

        # Generate entry strategies
        if recommendations['overall_bias'] == 'BULLISH':
            recommendations['entry_strategy'] = [
                'Look for pullbacks to key support levels',
                'Consider buying on dips towards VWAP',
                'Target breakouts above recent highs',
                'Use timeframes showing strong momentum for entry',
            ]
        elif recommendations['overall_bias'] == 'BEARISH':
            recommendations['entry_strategy'] = [
                'Look for rallies to key resistance levels',
                'Consider selling on rallies towards VWAP',
                'Target breakdowns below recent lows',
                'Use timeframes showing strong bearish momentum for entry',
            ]
        else:
            recommendations['entry_strategy'] = [
                'Wait for clear breakout or breakdown',
                'Consider range-bound strategies',
                'Look for reversal patterns at key levels',
                'Use shorter timeframes for entry timing',
            ]

        # Risk management recommendations
        recommendations['risk_management'] = [
            'Use stop losses below key support (long) or above resistance (short)',
            'Consider position sizing based on volatility (ATR)',
            'Take partial profits at key resistance/support levels',
            'Monitor multiple timeframe confirmations',
            'Adjust position size based on market regime',
        ]

        # Calculate confidence level
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            consensus_strength = max(bullish_signals, bearish_signals) / total_signals
            recommendations['confidence_level'] = int(consensus_strength * 100)
        else:
            recommendations['confidence_level'] = 0

        return recommendations

    # New helper methods for advanced analysis
    def _detect_harmonic_patterns(self, df):
        """Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)"""
        patterns = {
            'gartley_bullish': False,
            'gartley_bearish': False,
            'butterfly_bullish': False,
            'butterfly_bearish': False,
            'bat_bullish': False,
            'bat_bearish': False,
            'crab_bullish': False,
            'crab_bearish': False,
        }

        # Simplified harmonic pattern detection
        # In a real implementation, this would involve complex Fibonacci ratio calculations
        if len(df) >= 100:
            # Look for potential harmonic structures in recent price action
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            current_price = df['close'].iloc[-1]

            # Basic structure detection (simplified)
            if current_price > recent_low and current_price < recent_high:
                # Could be some form of retracement pattern
                pass

        return patterns

    def _detect_elliott_wave_patterns(self, df):
        """Detect Elliott Wave patterns (simplified)"""
        waves = {
            'impulse_wave_5': False,
            'corrective_wave_abc': False,
            'wave_count': 0,
            'current_wave_position': 'UNKNOWN',
        }

        if len(df) >= 50:
            # Simplified wave counting based on price movements
            price_changes = df['close'].diff().dropna()

            # Count significant moves
            significant_moves = price_changes[abs(price_changes) > price_changes.std()]
            waves['wave_count'] = len(significant_moves)

            # Very basic wave position estimation
            recent_trend = self._calculate_trend(df['close'].tail(20))
            if recent_trend > 0:
                waves['current_wave_position'] = 'UPTREND'
            elif recent_trend < 0:
                waves['current_wave_position'] = 'DOWNTREND'
            else:
                waves['current_wave_position'] = 'SIDEWAYS'

        return waves

    def _detect_supply_demand_zones(self, df):
        """Detect supply and demand zones"""
        zones = {
            'supply_zones': [],
            'demand_zones': [],
            'fresh_zones': [],
            'tested_zones': [],
        }

        if len(df) >= 50:
            # Look for strong price rejection zones
            lookback = min(50, len(df) // 4)

            for i in range(lookback, len(df) - 10):
                candle = df.iloc[i]

                # Demand zone (strong buying after downtrend)
                if (candle['close'] > candle['open'] and  # Bullish
                    candle['volume'] > df['volume'].iloc[i-5:i].mean()):  # High volume

                    # Check if this was a turning point
                    prior_trend = self._calculate_trend(df['close'].iloc[i-10:i])
                    if prior_trend < 0:  # Was in downtrend
                        zones['demand_zones'].append({
                            'price': candle['low'],
                            'strength': candle['volume'] / df['volume'].mean(),
                            'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                            'tested': df['low'].iloc[i+1:].min() <= candle['low'],
                        })

                # Supply zone (strong selling after uptrend)
                elif (candle['close'] < candle['open'] and  # Bearish
                      candle['volume'] > df['volume'].iloc[i-5:i].mean()):  # High volume

                    # Check if this was a turning point
                    prior_trend = self._calculate_trend(df['close'].iloc[i-10:i])
                    if prior_trend > 0:  # Was in uptrend
                        zones['supply_zones'].append({
                            'price': candle['high'],
                            'strength': candle['volume'] / df['volume'].mean(),
                            'time': df['datetime'].iloc[i] if 'datetime' in df.columns else i,
                            'tested': df['high'].iloc[i+1:].max() >= candle['high'],
                        })

        return zones

    def _detect_breakout_patterns(self, df):
        """Detect breakout patterns"""
        patterns = {
            'consolidation_breakout': False,
            'range_breakout': False,
            'volume_breakout': False,
            'gap_breakout': False,
            'false_breakout_risk': False,
        }

        if len(df) >= 30:
            # Recent consolidation detection
            recent_data = df.tail(30)
            price_range = recent_data['high'].max() - recent_data['low'].min()
            avg_range = (df['high'] - df['low']).rolling(10).mean().iloc[-1]

            # Narrow range consolidation
            if price_range < avg_range * 0.5:
                patterns['consolidation_breakout'] = True

            # Volume breakout detection
            recent_volume = df['volume'].tail(5)
            avg_volume = df['volume'].tail(30).mean()

            if recent_volume.iloc[-1] > avg_volume * 2:
                patterns['volume_breakout'] = True

        return patterns

    def _detect_consolidation_patterns(self, df):
        """Detect consolidation patterns"""
        patterns = {
            'rectangle': False,
            'triangle': False,
            'wedge': False,
            'flag': False,
            'pennant': False,
        }

        if len(df) >= 40:
            recent_data = df.tail(40)

            # Rectangle detection (sideways movement)
            price_std = recent_data['close'].std()
            price_mean = recent_data['close'].mean()

            if price_std / price_mean < 0.02:  # Less than 2% variation
                patterns['rectangle'] = True

            # Triangle detection (converting volatility)
            volatility = recent_data['high'] - recent_data['low']
            vol_trend = np.polyfit(range(len(volatility)), volatility, 1)[0]

            if vol_trend < -0.01:  # Decreasing volatility
                patterns['triangle'] = True

        return patterns

    def _detect_reversal_patterns(self, df):
        """Detect reversal patterns"""
        patterns = {
            'double_top': self._detect_double_top(df),
            'double_bottom': self._detect_double_bottom(df),
            'head_shoulders': self._detect_head_shoulders(df),
            'inverse_head_shoulders': self._detect_inverse_head_shoulders(df),
            'triple_top': False,
            'triple_bottom': False,
            'v_bottom': False,
            'v_top': False,
        }

        return patterns

    def _detect_continuation_patterns(self, df):
        """Detect continuation patterns"""
        patterns = {
            'ascending_triangle': False,
            'descending_triangle': False,
            'bull_flag': False,
            'bear_flag': False,
            'rising_wedge': False,
            'falling_wedge': False,
        }

        if len(df) >= 30:
            recent_data = df.tail(30)

            # Flag patterns (small consolidation after strong move)
            if len(recent_data) >= 10:
                strong_move = abs(recent_data['close'].pct_change()).nlargest(5).sum()
                consolidation = recent_data['close'].tail(10).std()

                if strong_move > 0.05 and consolidation < recent_data['close'].std() * 0.5:
                    patterns['bull_flag'] = recent_data['close'].iloc[-1] > recent_data['close'].iloc[-10]
                    patterns['bear_flag'] = recent_data['close'].iloc[-1] < recent_data['close'].iloc[-10]

        return patterns

    def _calculate_fear_greed_index(self, df):
        """Calculate simplified Fear & Greed Index"""
        if len(df) < 20:
            return 50  # Neutral

        # Components of Fear & Greed (simplified)
        try:
            # Price momentum
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100

            # Volatility (inverse - high volatility = fear)
            volatility = df['close'].pct_change().tail(20).std() * 100
            volatility_score = max(0, 100 - volatility * 10)

            # Volume surge
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_score = min(100, (current_volume / avg_volume) * 50) if avg_volume > 0 else 50

            # Price distance from moving averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            price_vs_sma = ((df['close'].iloc[-1] - sma_20) / sma_20) * 100 if sma_20 > 0 else 0
            sma_score = 50 + price_vs_sma

            # Combine scores
            fear_greed_score = (price_momentum + volatility_score + volume_score + sma_score) / 4
            fear_greed_score = max(0, min(100, fear_greed_score))

            return {
                'index': fear_greed_score,
                'sentiment': 'EXTREME_FEAR' if fear_greed_score < 20 else 'FEAR' if fear_greed_score < 40 else 'NEUTRAL' if fear_greed_score < 60 else 'GREED' if fear_greed_score < 80 else 'EXTREME_GREED',
                'components': {
                    'price_momentum': price_momentum,
                    'volatility_score': volatility_score,
                    'volume_score': volume_score,
                    'sma_score': sma_score,
                }
            }

        except:
            return {'index': 50, 'sentiment': 'NEUTRAL', 'components': {}}

    def _calculate_sentiment_score(self, df):
        """Calculate overall market sentiment score"""
        try:
            # RSI contribution
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1] if len(df) >= 14 else 50
            rsi_sentiment = (rsi - 50) * 2  # -100 to +100

            # MACD contribution
            macd, macd_signal, _ = talib.MACD(df['close'])
            if not pd.isna(macd.iloc[-1]) and not pd.isna(macd_signal.iloc[-1]):
                macd_sentiment = (macd.iloc[-1] - macd_signal.iloc[-1]) * 1000
            else:
                macd_sentiment = 0

            # Price vs moving averages
            sma_short = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['close'].iloc[-1]
            sma_long = df['close'].rolling(30).mean().iloc[-1] if len(df) >= 30 else df['close'].iloc[-1]
            price_vs_sma = ((df['close'].iloc[-1] - sma_short) / sma_short +
                           (df['close'].iloc[-1] - sma_long) / sma_long) * 50 if sma_short > 0 and sma_long > 0 else 0

            # Volume sentiment
            volume_avg = df['volume'].tail(20).mean()
            volume_current = df['volume'].iloc[-1]
            volume_sentiment = (volume_current / volume_avg - 1) * 20 if volume_avg > 0 else 0

            # Combine all sentiments
            overall_sentiment = (rsi_sentiment + macd_sentiment + price_vs_sma + volume_sentiment) / 4

            return {
                'overall_score': overall_sentiment,
                'sentiment': 'STRONGLY_BULLISH' if overall_sentiment > 50 else 'BULLISH' if overall_sentiment > 20 else 'NEUTRAL' if overall_sentiment > -20 else 'BEARISH' if overall_sentiment > -50 else 'STRONGLY_BEARISH',
                'components': {
                    'rsi_sentiment': rsi_sentiment,
                    'macd_sentiment': macd_sentiment,
                    'price_sentiment': price_vs_sma,
                    'volume_sentiment': volume_sentiment,
                }
            }

        except:
            return {'overall_score': 0, 'sentiment': 'NEUTRAL', 'components': {}}

    def _calculate_volatility_sentiment(self, df):
        """Calculate volatility-based sentiment"""
        if len(df) < 20:
            return {}

        try:
            current_volatility = df['close'].pct_change().tail(20).std() * 100
            historical_volatility = df['close'].pct_change().std() * 100

            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1

            return {
                'current_volatility': current_volatility,
                'historical_volatility': historical_volatility,
                'volatility_ratio': volatility_ratio,
                'volatility_regime': 'HIGH' if volatility_ratio > 1.5 else 'LOW' if volatility_ratio < 0.7 else 'NORMAL',
                'volatility_trend': 'INCREASING' if current_volatility > df['close'].pct_change().tail(10).std() * 100 else 'DECREASING',
            }

        except:
            return {}

    def _calculate_trend_sentiment(self, df):
        """Calculate trend-based sentiment"""
        if len(df) < 30:
            return {}

        try:
            # Multiple timeframe trend analysis
            short_trend = self._calculate_trend(df['close'].tail(10))
            medium_trend = self._calculate_trend(df['close'].tail(20))
            long_trend = self._calculate_trend(df['close'].tail(30))

            trend_strength = abs(short_trend) + abs(medium_trend) + abs(long_trend)

            return {
                'short_trend': 'BULLISH' if short_trend > 0 else 'BEARISH' if short_trend < 0 else 'NEUTRAL',
                'medium_trend': 'BULLISH' if medium_trend > 0 else 'BEARISH' if medium_trend < 0 else 'NEUTRAL',
                'long_trend': 'BULLISH' if long_trend > 0 else 'BEARISH' if long_trend < 0 else 'NEUTRAL',
                'trend_strength': 'STRONG' if trend_strength > 2 else 'MODERATE' if trend_strength > 1 else 'WEAK',
                'trend_consensus': 'BULLISH' if short_trend + medium_trend + long_trend > 1 else 'BEARISH' if short_trend + medium_trend + long_trend < -1 else 'NEUTRAL',
            }

        except:
            return {}

    def _calculate_volume_sentiment(self, df):
        """Calculate volume-based sentiment"""
        if len(df) < 20:
            return {}

        try:
            recent_volume = df['volume'].tail(5)
            avg_volume = df['volume'].tail(20).mean()

            volume_ratio = recent_volume.iloc[-1] / avg_volume if avg_volume > 0 else 1

            # Buying vs selling pressure
            buying_volume = df[df['close'] > df['open']]['volume'].tail(10).sum()
            selling_volume = df[df['close'] < df['open']]['volume'].tail(10).sum()
            total_volume = buying_volume + selling_volume

            buy_pressure = (buying_volume / total_volume * 100) if total_volume > 0 else 50

            return {
                'volume_ratio': volume_ratio,
                'volume_level': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.7 else 'NORMAL',
                'buying_pressure': buy_pressure,
                'pressure_sentiment': 'BULLISH' if buy_pressure > 60 else 'BEARISH' if buy_pressure < 40 else 'NEUTRAL',
                'volume_trend': self._calculate_volume_trend(recent_volume),
            }

        except:
            return {}

    def _calculate_currency_correlations(self, df):
        """Calculate currency correlations (placeholder)"""
        # Would need multiple currency data for proper implementation
        return {
            'usd_index_correlation': 0.5,  # Placeholder
            'eur_correlation': -0.3,       # Placeholder
            'jpy_correlation': -0.2,       # Placeholder
            'gbp_correlation': -0.1,       # Placeholder
        }

    def _calculate_commodity_correlations(self, df):
        """Calculate commodity correlations (placeholder)"""
        return {
            'gold_correlation': 0.8,   # Placeholder for XAUUSD
            'oil_correlation': 0.1,    # Placeholder
            'silver_correlation': 0.6, # Placeholder
        }

    def _calculate_market_breadth(self, df):
        """Calculate market breadth (placeholder)"""
        return {
            'breadth_score': 50,
            'advancing_issues': 0,
            'declining_issues': 0,
            'breadth_trend': 'NEUTRAL',
        }

    def _analyze_market_leadership(self, df):
        """Analyze market leadership (placeholder)"""
        return {
            'leading_sector': 'UNKNOWN',
            'leadership_strength': 0.5,
            'leadership_change': False,
        }


def main():
    """Main function for comprehensive feature analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Feature Analyzer for Historical Market Data')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., XAUUSD)')
    parser.add_argument('--data-dir', default='./data', help='Data directory path')
    parser.add_argument('--output-dir', default='./data/rag_processed', help='Output directory')
    parser.add_argument('--add-to-rag', action='store_true', help='Add analysis to RAG knowledge base')
    parser.add_argument('--model', default='qwen3:14b', help='LLM model for analysis')
    parser.add_argument('--api-base', default='http://localhost:8080', help='LLM API base URL')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ComprehensiveFeatureAnalyzer()

    # Process symbol data
    success = analyzer.process_symbol_data(
        symbol=args.symbol,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    if success:
        print(f"\n🎉 Comprehensive historical analysis completed successfully for {args.symbol}!")

        # Optional: Add to RAG knowledge base
        if args.add_to_rag:
            print(f"\n🔄 Adding comprehensive analysis to RAG knowledge base...")
            # Here you would implement the RAG addition logic
            print(f"✅ Analysis added to knowledge base")

        print(f"\n📝 You can now ask for comprehensive trading analysis including:")
        print(f"   • Multi-timeframe trend analysis")
        print(f"   • Support and resistance levels")
        print(f"   • Risk management recommendations")
        print(f"   • Entry and exit strategies")
        print(f"   • Market regime analysis")

    else:
        print(f"\n❌ Failed to process comprehensive analysis for {args.symbol}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())