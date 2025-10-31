import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical Analysis library: pip install ta

class TradingDataEnricher:
    def __init__(self):
        self.technical_indicators = {}
        
    def load_raw_data(self, file_path):
        """Load raw CSV data"""
        df = pd.read_csv(file_path,encoding="utf-16")
        
        # Convert datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M')
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def calculate_basic_features(self, df):
        """Calculate basic price features"""
        # Price movements
        df['price_change'] = df['Close'].diff()
        df['price_change_pct'] = df['Close'].pct_change() * 100
        
        # High-Low relationships
        df['hl_range'] = df['High'] - df['Low']
        df['hl_range_pct'] = (df['hl_range'] / df['Close']) * 100
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['body_size_pct'] = (df['body_size'] / df['Close']) * 100
        
        # Candle type
        df['is_bullish'] = df['Close'] > df['Open']
        df['candle_strength'] = np.where(
            df['is_bullish'], 
            (df['Close'] - df['Open']) / df['hl_range'],
            (df['Open'] - df['Close']) / df['hl_range']
        )
        
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI multiple timeframes
        df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
        df['rsi_21'] = ta.momentum.rsi(df['Close'], window=21)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Support/Resistance levels (simplified)
        df['resistance_1'] = df['High'].rolling(50).max()
        df['support_1'] = df['Low'].rolling(50).min()
        
        # ATR for volatility
        df['atr_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        return df
    
    def calculate_market_structure(self, df):
        """Calculate market structure features"""
        # Higher highs/lows
        df['higher_high'] = df['High'] > df['High'].shift(1)
        df['higher_low'] = df['Low'] > df['Low'].shift(1)
        df['lower_high'] = df['High'] < df['High'].shift(1)
        df['lower_low'] = df['Low'] < df['Low'].shift(1)
        
        # Swing points
        df['is_swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
        df['is_swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
        
        # Trend identification
        df['trend_strength'] = df['Close'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() > 0 else 0
        )
        
        return df
    
    def create_trading_signals(self, df):
        """Generate trading signals"""
        # RSI signals
        df['rsi_oversold'] = df['rsi_14'] < 30
        df['rsi_overbought'] = df['rsi_14'] > 70
        
        # MACD signals
        df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Moving average crossovers
        df['ma_cross_bullish'] = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
        df['ma_cross_bearish'] = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
        
        # Bollinger Band signals
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.01
        df['bb_breakout_up'] = df['Close'] > df['bb_upper']
        df['bb_breakout_down'] = df['Close'] < df['bb_lower']
        
        return df
    
    def create_llm_friendly_chunks(self, df, lookback_periods=100):
        """Create LLM-friendly text chunks from the enriched data"""
        
        # Get the most recent data for analysis
        recent_data = df.tail(lookback_periods).copy()
        current = recent_data.iloc[-1]
        
        chunks = []
        
        # Chunk 1: Executive Summary
        executive_chunk = self._create_executive_chunk(current, recent_data)
        chunks.append(("executive_summary", executive_chunk))
        
        # Chunk 2: Technical Analysis
        technical_chunk = self._create_technical_chunk(current, recent_data)
        chunks.append(("technical_analysis", technical_chunk))
        
        # Chunk 3: Trading Signals
        signals_chunk = self._create_signals_chunk(current, recent_data)
        chunks.append(("trading_signals", signals_chunk))
        
        # Chunk 4: Market Context
        context_chunk = self._create_context_chunk(current, recent_data)
        chunks.append(("market_context", context_chunk))
        
        return chunks
    
    def _create_executive_chunk(self, current, recent_data):
        """Create executive summary chunk"""
        price_change_1h = ((current['Close'] - recent_data.iloc[-60]['Close']) / recent_data.iloc[-60]['Close']) * 100
        volatility = recent_data['hl_range_pct'].mean()
        
        chunk = f"""
XAUUSD TRADING EXECUTIVE SUMMARY - {current['DateTime'].strftime('%Y-%m-%d %H:%M')}

CURRENT PRICE ACTION:
- Price: {current['Close']:.2f}
- 1H Change: {price_change_1h:+.2f}%
- Volatility: {volatility:.2f}% avg range
- Volume: {current['Volume']:.0f} ({current['volume_ratio']:.1f}x avg)

KEY LEVELS (Recent):
- Resistance: {recent_data['High'].max():.2f}
- Support: {recent_data['Low'].min():.2f}
- Range: {recent_data['High'].max() - recent_data['Low'].min():.2f} points

MARKET CONDITION: {self._get_market_condition(current, recent_data)}
"""
        return chunk.strip()
    
    def _create_technical_chunk(self, current, recent_data):
        """Create technical analysis chunk"""
        chunk = f"""
TECHNICAL ANALYSIS:

TREND & MOMENTUM:
- Trend Strength: {current['trend_strength']:.2f} ({self._get_trend_strength(current['trend_strength'])})
- RSI 14: {current['rsi_14']:.1f} ({self._get_rsi_condition(current['rsi_14'])})
- RSI 7: {current['rsi_7']:.1f}, RSI 21: {current['rsi_21']:.1f}
- MACD: {current['macd']:.3f} (Signal: {current['macd_signal']:.3f})

MOVING AVERAGES:
- EMA 12: {current['ema_12']:.2f} | Position: {self._get_ma_position(current, 'ema_12')}
- EMA 26: {current['ema_26']:.2f} | Position: {self._get_ma_position(current, 'ema_26')}
- SMA 20: {current['sma_20']:.2f} | SMA 50: {current['sma_50']:.2f}

BOLLINGER BANDS:
- Position: {current['bb_position']:.1%} ({self._get_bb_position(current['bb_position'])})
- Band Width: {(current['bb_upper'] - current['bb_lower']):.2f}
- Squeeze: {'Yes' if current['bb_squeeze'] else 'No'}
"""
        return chunk.strip()
    
    def _create_signals_chunk(self, current, recent_data):
        """Create trading signals chunk"""
        signals = []
        
        if current['rsi_oversold']:
            signals.append("RSI OVERSOLD (<30)")
        if current['rsi_overbought']:
            signals.append("RSI OVERBOUGHT (>70)")
        if current['macd_bullish']:
            signals.append("MACD BULLISH CROSSOVER")
        if current['macd_bearish']:
            signals.append("MACD BEARISH CROSSOVER")
        if current['ma_cross_bullish']:
            signals.append("MA BULLISH CROSSOVER")
        if current['ma_cross_bearish']:
            signals.append("MA BEARISH CROSSOVER")
        if current['bb_breakout_up']:
            signals.append("BB UPPER BREAKOUT")
        if current['bb_breakout_down']:
            signals.append("BB LOWER BREAKOUT")
            
        signal_text = " | ".join(signals) if signals else "NO STRONG SIGNALS"
        
        chunk = f"""
TRADING SIGNALS:

ACTIVE SIGNALS: {signal_text}

CONFLUENCE FACTORS:
- Volume: {'Above average' if current['volume_ratio'] > 1.2 else 'Below average'}
- Trend: {'Aligning' if self._check_signal_alignment(current) else 'Mixed'}
- Volatility: {'High' if current['atr_14'] > recent_data['atr_14'].mean() else 'Normal'}

RISK METRICS:
- ATR (14): {current['atr_14']:.2f}
- Recent Range: {recent_data['hl_range'].mean():.2f}
- Stop Distance: {current['atr_14'] * 1.5:.2f} (1.5x ATR)
"""
        return chunk.strip()
    
    def _create_context_chunk(self, current, recent_data):
        """Create market context chunk"""
        # Calculate session information (simplified)
        current_hour = current['DateTime'].hour
        session = self._get_trading_session(current_hour)
        
        chunk = f"""
MARKET CONTEXT:

TRADING SESSION: {session}
TIME: {current['DateTime'].strftime('%H:%M')}

RECENT PRICE ACTION:
- Bullish Candles: {recent_data['is_bullish'].sum()} of {len(recent_data)}
- Avg Candle Size: {recent_data['body_size'].mean():.2f}
- Strongest Move: {recent_data['price_change_pct'].abs().max():.2f}%

VOLUME ANALYSIS:
- Current: {current['Volume']:.0f}
- Average: {recent_data['Volume'].mean():.0f}
- Trend: {'Increasing' if current['Volume'] > recent_data['Volume'].mean() else 'Decreasing'}

SWING POINTS (Last {len(recent_data)} periods):
- Swing Highs: {recent_data['is_swing_high'].sum()}
- Swing Lows: {recent_data['is_swing_low'].sum()}
"""
        return chunk.strip()
    
    def _get_market_condition(self, current, recent_data):
        """Determine overall market condition"""
        if current['trend_strength'] > 0.5:
            return "STRONG UPTREND"
        elif current['trend_strength'] < -0.5:
            return "STRONG DOWNTREND"
        elif abs(current['trend_strength']) < 0.2:
            return "RANGING/CONSOLIDATING"
        else:
            return "MILD TREND"
    
    def _get_trend_strength(self, strength):
        """Convert trend strength to text"""
        if abs(strength) > 1.0:
            return "Very Strong"
        elif abs(strength) > 0.5:
            return "Strong"
        elif abs(strength) > 0.2:
            return "Moderate"
        else:
            return "Weak"
    
    def _get_rsi_condition(self, rsi):
        """Convert RSI to condition text"""
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _get_ma_position(self, current, ma_column):
        """Get position relative to MA"""
        if current['Close'] > current[ma_column]:
            return "Above"
        else:
            return "Below"
    
    def _get_bb_position(self, bb_pos):
        """Get Bollinger Band position text"""
        if bb_pos > 0.8:
            return "Upper Band"
        elif bb_pos < 0.2:
            return "Lower Band"
        else:
            return "Middle Range"
    
    def _get_trading_session(self, hour):
        """Determine trading session"""
        if 0 <= hour < 5:
            return "ASIA"
        elif 5 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 21:
            return "NEW YORK"
        else:
            return "LATE NY/PACIFIC"
    
    def _check_signal_alignment(self, current):
        """Check if signals are aligned"""
        bullish_signals = sum([
            current['macd_bullish'],
            current['ma_cross_bullish'],
            current['rsi_oversold']
        ])
        
        bearish_signals = sum([
            current['macd_bearish'], 
            current['ma_cross_bearish'],
            current['rsi_overbought']
        ])
        
        return abs(bullish_signals - bearish_signals) >= 2
    
    def process_pipeline(self, input_file, output_dir="./enriched_data"):
        """Complete processing pipeline"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data
        print("Loading raw data...")
        df = self.load_raw_data(input_file)
        
        print("Calculating basic features...")
        df = self.calculate_basic_features(df)
        
        print("Calculating technical indicators...")
        df = self.calculate_technical_indicators(df)
        
        print("Analyzing market structure...")
        df = self.calculate_market_structure(df)
        
        print("Generating trading signals...")
        df = self.create_trading_signals(df)
        
        print("Creating LLM-friendly chunks...")
        chunks = self.create_llm_friendly_chunks(df)
        
        # Save results
        df.to_csv(f"{output_dir}/enriched_trading_data.csv", index=False)
        
        for chunk_name, chunk_content in chunks:
            with open(f"{output_dir}/{chunk_name}.txt", "w") as f:
                f.write(chunk_content)
        
        print(f"Processing complete! Files saved to {output_dir}")
        
        return df, chunks

# Usage example
if __name__ == "__main__":
    enricher = TradingDataEnricher()
    
    # Process your data
    df, chunks = enricher.process_pipeline("../data/XAUUSD_PERIOD_M1_0.csv")
    
    # Print sample chunks
    for chunk_name, chunk_content in chunks:
        print(f"\n{'='*50}")
        print(f"CHUNK: {chunk_name}")
        print(f"{'='*50}")
        print(chunk_content)
        print(f"{'='*50}")
