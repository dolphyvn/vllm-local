import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime, timedelta
import talib  # For technical indicators

class TradingDataProcessor:
    """
    Convert MT5/Sierra Chart CSV into vector database format
    """
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.chroma_client = chromadb.Client()
        
        # Create collections
        self.collections = {
            'patterns': self.chroma_client.create_collection("trading_patterns"),
            'levels': self.chroma_client.create_collection("support_resistance"),
            'setups': self.chroma_client.create_collection("indicator_setups"),
            'context': self.chroma_client.create_collection("market_context"),
        }
        
    def load_and_clean(self):
        """Load CSV and prepare data"""
        # Parse timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Calculate technical indicators
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=14)
        self.df['ema9'] = talib.EMA(self.df['close'], timeperiod=9)
        self.df['ema20'] = talib.EMA(self.df['close'], timeperiod=20)
        self.df['ema50'] = talib.EMA(self.df['close'], timeperiod=50)
        
        # Bollinger Bands
        self.df['bb_upper'], self.df['bb_middle'], self.df['bb_lower'] = \
            talib.BBANDS(self.df['close'], timeperiod=20)
        
        # ATR
        self.df['atr'] = talib.ATR(self.df['high'], self.df['low'], 
                                    self.df['close'], timeperiod=14)
        
        # Volume analysis
        self.df['vol_sma'] = self.df['tick_volume'].rolling(20).mean()
        self.df['vol_ratio'] = self.df['tick_volume'] / self.df['vol_sma']
        
        return self.df
    
    def detect_patterns(self):
        """Detect candlestick patterns and setups"""
        patterns = []
        
        for i in range(100, len(self.df)):  # Start after enough data
            row = self.df.iloc[i]
            prev_rows = self.df.iloc[i-5:i]
            
            # 1. Bullish Engulfing
            if self._is_bullish_engulfing(i):
                pattern = self._create_pattern_text(
                    row, prev_rows, 
                    pattern_name="Bullish Engulfing",
                    outcome=self._check_outcome(i, direction='long')
                )
                patterns.append(pattern)
            
            # 2. RSI Divergence
            if self._is_rsi_divergence(i):
                pattern = self._create_divergence_text(row, prev_rows)
                patterns.append(pattern)
            
            # 3. Support/Resistance Bounce
            if self._is_sr_bounce(i):
                pattern = self._create_bounce_text(row, prev_rows)
                patterns.append(pattern)
            
            # 4. Breakout
            if self._is_breakout(i):
                pattern = self._create_breakout_text(
                    row, prev_rows,
                    outcome=self._check_outcome(i, direction='long')
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_bullish_engulfing(self, idx):
        """Detect bullish engulfing pattern"""
        current = self.df.iloc[idx]
        prev = self.df.iloc[idx-1]
        
        # Bearish candle followed by bullish candle that engulfs it
        prev_bearish = prev['close'] < prev['open']
        curr_bullish = current['close'] > current['open']
        engulfing = (current['open'] < prev['close'] and 
                    current['close'] > prev['open'])
        
        return prev_bearish and curr_bullish and engulfing
    
    def _is_rsi_divergence(self, idx):
        """Detect RSI divergence"""
        if idx < 20:
            return False
        
        current = self.df.iloc[idx]
        lookback = self.df.iloc[idx-20:idx]
        
        # Bullish divergence: price lower low, RSI higher low
        price_ll = current['low'] < lookback['low'].min()
        rsi_hl = current['rsi'] > lookback['rsi'].min()
        
        return price_ll and rsi_hl
    
    def _is_sr_bounce(self, idx):
        """Detect support/resistance bounce"""
        current = self.df.iloc[idx]
        prev_candles = self.df.iloc[idx-100:idx]
        
        # Find nearby historical highs/lows
        resistance_levels = prev_candles.nlargest(5, 'high')['high'].values
        support_levels = prev_candles.nsmallest(5, 'low')['low'].values
        
        # Check if current price near any level
        tolerance = current['atr'] * 0.5
        
        for level in support_levels:
            if abs(current['close'] - level) < tolerance:
                # Bounce from support
                if current['close'] > current['open']:  # Bullish candle
                    return True
        
        return False
    
    def _is_breakout(self, idx):
        """Detect breakout pattern"""
        current = self.df.iloc[idx]
        prev_candles = self.df.iloc[idx-20:idx]
        
        # Breakout above recent high with volume
        recent_high = prev_candles['high'].max()
        breakout = current['close'] > recent_high
        volume_confirm = current['vol_ratio'] > 1.5
        
        return breakout and volume_confirm
    
    def _check_outcome(self, idx, direction='long', lookforward=20):
        """Check if setup was profitable"""
        entry = self.df.iloc[idx]['close']
        future = self.df.iloc[idx+1:idx+lookforward+1]
        
        if len(future) == 0:
            return None
        
        if direction == 'long':
            target = entry + (self.df.iloc[idx]['atr'] * 2)
            stop = entry - self.df.iloc[idx]['atr']
            
            # Check if target hit
            if future['high'].max() >= target:
                return {
                    'result': 'win',
                    'pnl': target - entry,
                    'bars': (future['high'] >= target).idxmax() - idx
                }
            # Check if stopped out
            elif future['low'].min() <= stop:
                return {
                    'result': 'loss',
                    'pnl': stop - entry,
                    'bars': (future['low'] <= stop).idxmax() - idx
                }
        
        return None
    
    def _create_pattern_text(self, row, prev_rows, pattern_name, outcome):
        """Convert pattern to narrative text for embedding"""
        
        # Calculate context
        trend = "bullish" if row['ema9'] > row['ema20'] else "bearish"
        rsi_status = "overbought" if row['rsi'] > 70 else \
                     "oversold" if row['rsi'] < 30 else "neutral"
        
        text = f"""
Symbol: XAUUSD
Date: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}
Pattern: {pattern_name}

Price Action:
- Entry price: {row['close']:.2f}
- Previous 5-bar range: {prev_rows['low'].min():.2f} - {prev_rows['high'].max():.2f}
- Current candle: Open {row['open']:.2f}, Close {row['close']:.2f}, 
  High {row['high']:.2f}, Low {row['low']:.2f}

Technical Context:
- Trend: {trend} (EMA9: {row['ema9']:.2f}, EMA20: {row['ema20']:.2f})
- RSI: {row['rsi']:.1f} ({rsi_status})
- Price vs BB: {self._bb_position(row)}
- Volume: {row['vol_ratio']:.2f}x average
- ATR: {row['atr']:.2f}

Setup:
- Entry: {row['close']:.2f}
- Stop: {row['close'] - row['atr']:.2f}
- Target: {row['close'] + (row['atr'] * 2):.2f}
- Risk/Reward: 1:2
"""
        
        # Add outcome if available
        if outcome:
            text += f"""
Outcome: {outcome['result'].upper()}
- P&L: {outcome['pnl']:.2f} points
- Bars to completion: {outcome['bars']}
- Setup quality: {"High" if outcome['result'] == 'win' else "Poor"}
"""
        
        return {
            'text': text,
            'metadata': {
                'pattern': pattern_name,
                'timestamp': row['timestamp'].isoformat(),
                'entry': float(row['close']),
                'rsi': float(row['rsi']),
                'trend': trend,
                'outcome': outcome['result'] if outcome else 'pending',
                'symbol': 'XAUUSD'
            }
        }
    
    def _bb_position(self, row):
        """Determine Bollinger Band position"""
        if row['close'] > row['bb_upper']:
            return "Above upper band (overbought)"
        elif row['close'] < row['bb_lower']:
            return "Below lower band (oversold)"
        else:
            pct = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if pct > 0.7:
                return "Upper third (bullish)"
            elif pct < 0.3:
                return "Lower third (bearish)"
            else:
                return "Middle band (neutral)"
    
    def find_support_resistance(self):
        """Identify key support/resistance levels"""
        levels = []
        
        # Use swing highs/lows
        self.df['swing_high'] = self.df['high'][(self.df['high'] > 
                                self.df['high'].shift(1)) & 
                                (self.df['high'] > 
                                self.df['high'].shift(-1))]
        
        self.df['swing_low'] = self.df['low'][(self.df['low'] < 
                               self.df['low'].shift(1)) & 
                               (self.df['low'] < 
                               self.df['low'].shift(-1))]
        
        # Cluster nearby levels
        swing_highs = self.df['swing_high'].dropna().values
        swing_lows = self.df['swing_low'].dropna().values
        
        # Find clustered resistance
        for level in self._cluster_levels(swing_highs):
            touches = np.sum(np.abs(swing_highs - level) < self.df['atr'].mean())
            
            text = f"""
XAUUSD Resistance Level: {level:.2f}
- Type: Swing High cluster
- Number of touches: {touches}
- Strength: {"Strong" if touches >= 3 else "Moderate"}
- Identified from: {len(swing_highs)} data points
- Typical reaction: Price rejected {touches}/{touches} times
- Trading strategy: Look for shorts near this level with confirmation
"""
            
            levels.append({
                'text': text,
                'metadata': {
                    'type': 'resistance',
                    'level': float(level),
                    'touches': int(touches),
                    'strength': 'strong' if touches >= 3 else 'moderate',
                    'symbol': 'XAUUSD'
                }
            })
        
        # Find clustered support
        for level in self._cluster_levels(swing_lows):
            touches = np.sum(np.abs(swing_lows - level) < self.df['atr'].mean())
            
            text = f"""
XAUUSD Support Level: {level:.2f}
- Type: Swing Low cluster
- Number of touches: {touches}
- Strength: {"Strong" if touches >= 3 else "Moderate"}
- Identified from: {len(swing_lows)} data points
- Typical reaction: Price bounced {touches}/{touches} times
- Trading strategy: Look for longs near this level with confirmation
"""
            
            levels.append({
                'text': text,
                'metadata': {
                    'type': 'support',
                    'level': float(level),
                    'touches': int(touches),
                    'strength': 'strong' if touches >= 3 else 'moderate',
                    'symbol': 'XAUUSD'
                }
            })
        
        return levels
    
    def _cluster_levels(self, prices, tolerance=None):
        """Cluster nearby price levels"""
        if tolerance is None:
            tolerance = self.df['atr'].mean()
        
        clusters = []
        sorted_prices = np.sort(prices)
        
        i = 0
        while i < len(sorted_prices):
            cluster = [sorted_prices[i]]
            j = i + 1
            
            while j < len(sorted_prices) and \
                  sorted_prices[j] - sorted_prices[i] < tolerance:
                cluster.append(sorted_prices[j])
                j += 1
            
            if len(cluster) >= 2:  # At least 2 touches
                clusters.append(np.mean(cluster))
            
            i = j
        
        return clusters
    
    def add_to_vector_db(self, documents, collection_name):
        """Add documents to vector database"""
        collection = self.collections[collection_name]
        
        for doc in documents:
            embedding = self.embed_model.encode(doc['text'])
            
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[doc['text']],
                metadatas=[doc['metadata']],
                ids=[f"{collection_name}_{datetime.now().timestamp()}_{np.random.randint(10000)}"]
            )
    
    def process_all(self):
        """Complete processing pipeline"""
        print("Loading and cleaning data...")
        self.load_and_clean()
        
        print("Detecting patterns...")
        patterns = self.detect_patterns()
        print(f"Found {len(patterns)} patterns")
        
        print("Finding support/resistance...")
        levels = self.find_support_resistance()
        print(f"Found {len(levels)} key levels")
        
        print("Adding to vector database...")
        self.add_to_vector_db(patterns, 'patterns')
        self.add_to_vector_db(levels, 'levels')
        
        print("Processing complete!")
        return {
            'patterns': len(patterns),
            'levels': len(levels),
            'total_bars': len(self.df)
        }

# Usage
if __name__ == "__main__":
    # Process your MT5/Sierra Chart export
    processor = TradingDataProcessor('XAUUSD_M15.csv')
    results = processor.process_all()
    
    print(f"\n=== Processing Complete ===")
    print(f"Patterns stored: {results['patterns']}")
    print(f"Levels stored: {results['levels']}")
    print(f"Total bars processed: {results['total_bars']}")
