#!/usr/bin/env python3
"""
Live Trading Data Analyzer for Real-time Trade Recommendations
Integrates with existing RAG system to provide live trade advice
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import ta

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from knowledge_feeder import KnowledgeEntry
    from feed_to_rag_direct import RAGKnowledgeBase
    KNOWLEDGE_FEEDER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_FEEDER_AVAILABLE = False
    print("Warning: knowledge_feeder modules not available")

@dataclass
class MarketConditions:
    """Current market conditions for analysis"""
    timestamp: datetime
    price: float
    volume: int
    rsi: float
    vwap: float
    session: str
    trend: str
    volatility: float
    correlation_score: float = 0.0

@dataclass
class TradeRecommendation:
    """Live trade recommendation"""
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-100
    reasoning: str
    similar_patterns: List[Dict]
    risk_reward_ratio: float
    market_context: Dict

class LiveTradingAnalyzer:
    """Real-time trading analysis system"""

    def __init__(self, rag_base_url="http://localhost:8080", mt5_data_path=None):
        self.rag_base_url = rag_base_url
        self.mt5_data_path = mt5_data_path or "./data/live"
        self.session = requests.Session()

        # Initialize technical indicators calculator
        self.indicators_calculator = TechnicalIndicators()

        # Session definitions (UTC)
        self.sessions = {
            'late_asia': (0, 2),
            'asia': (2, 6),
            'asia_london_overlap': (6, 8),
            'london': (8, 12),
            'london_ny_overlap': (12, 13),
            'new_york': (13, 17),
            'late_ny': (17, 20),
            'quiet_hours': (20, 24)
        }

        print("🔴 Live Trading Analyzer initialized")
        print(f"📊 RAG URL: {rag_base_url}")
        print(f"📈 Data path: {self.mt5_data_path}")

    def get_current_session(self, timestamp: datetime) -> str:
        """Determine current trading session"""
        hour = timestamp.hour

        for session_name, (start, end) in self.sessions.items():
            if start <= hour < end:
                return session_name

        return 'quiet_hours'

    def calculate_live_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate real-time technical indicators"""
        # Enhanced indicators matching our enhanced processor
        df = self.indicators_calculator.calculate_basic_indicators(df)
        df = self.indicators_calculator.calculate_advanced_indicators(df)
        df = self.indicators_calculator.calculate_session_vwap(df)
        df = self.indicators_calculator.calculate_market_profile(df)

        return df

    def detect_live_patterns(self, df: pd.DataFrame, last_candles: int = 5) -> List[Dict]:
        """Detect forming patterns in live data"""
        patterns = []
        current_candles = df.tail(last_candles)

        if len(current_candles) < 5:
            return patterns

        latest_candle = current_candles.iloc[-1]
        prev_candle = current_candles.iloc[-2]

        # Get current market conditions
        current_session = self.get_current_session(latest_candle.name)
        rsi = latest_candle.get('rsi', 50)
        vwap = latest_candle.get('vwap', latest_candle['close'])
        volume_ratio = latest_candle.get('volume_ratio', 1.0)

        # Detect Bull Flag formation
        if self._detect_bull_flag_formation(current_candles):
            patterns.append({
                'pattern': 'Bull Flag Forming',
                'direction': 'BUY',
                'confidence': self._calculate_pattern_confidence('bull_flag', current_candles),
                'entry_trigger': latest_candle['close'] + (latest_candle['high'] - latest_candle['low']) * 0.1,
                'stop_loss': latest_candle['low'] - (latest_candle['high'] - latest_candle['low']) * 0.05,
                'session': current_session,
                'rsi': rsi,
                'vwap_deviation': (latest_candle['close'] - vwap) / vwap * 100,
                'volume_ratio': volume_ratio
            })

        # Detect Bear Flag formation
        if self._detect_bear_flag_formation(current_candles):
            patterns.append({
                'pattern': 'Bear Flag Forming',
                'direction': 'SELL',
                'confidence': self._calculate_pattern_confidence('bear_flag', current_candles),
                'entry_trigger': latest_candle['close'] - (latest_candle['high'] - latest_candle['low']) * 0.1,
                'stop_loss': latest_candle['high'] + (latest_candle['high'] - latest_candle['low']) * 0.05,
                'session': current_session,
                'rsi': rsi,
                'vwap_deviation': (latest_candle['close'] - vwap) / vwap * 100,
                'volume_ratio': volume_ratio
            })

        # Detect VWAP bounce/rejection
        if self._detect_vwap_reaction(latest_candle, vwap):
            direction = 'BUY' if latest_candle['close'] > vwap else 'SELL'
            patterns.append({
                'pattern': 'VWAP Reaction',
                'direction': direction,
                'confidence': 75,
                'entry_trigger': latest_candle['close'],
                'stop_loss': vwap if direction == 'SELL' else latest_candle['low'],
                'session': current_session,
                'rsi': rsi,
                'vwap_deviation': (latest_candle['close'] - vwap) / vwap * 100,
                'volume_ratio': volume_ratio
            })

        return patterns

    def _detect_bull_flag_formation(self, candles: pd.DataFrame) -> bool:
        """Detect bull flag pattern formation"""
        if len(candles) < 5:
            return False

        # Strong move up followed by consolidation
        first_candle = candles.iloc[0]
        last_candle = candles.iloc[-1]

        # Check for strong initial move
        price_change = (last_candle['close'] - first_candle['open']) / first_candle['open']

        # Consolidation pattern (small range candles)
        consolidation = True
        for i in range(1, len(candles)):
            candle_range = (candles.iloc[i]['high'] - candles.iloc[i]['low']) / candles.iloc[i]['close']
            if candle_range > 0.01:  # More than 1% range
                consolidation = False

        return price_change > 0.005 and consolidation  # 0.5% move up

    def _detect_bear_flag_formation(self, candles: pd.DataFrame) -> bool:
        """Detect bear flag pattern formation"""
        if len(candles) < 5:
            return False

        first_candle = candles.iloc[0]
        last_candle = candles.iloc[-1]

        price_change = (last_candle['close'] - first_candle['open']) / first_candle['open']

        consolidation = True
        for i in range(1, len(candles)):
            candle_range = (candles.iloc[i]['high'] - candles.iloc[i]['low']) / candles.iloc[i]['close']
            if candle_range > 0.01:
                consolidation = False

        return price_change < -0.005 and consolidation  # 0.5% move down

    def _detect_vwap_reaction(self, candle: pd.Series, vwap: float) -> bool:
        """Detect price reaction at VWAP"""
        distance_from_vwap = abs(candle['close'] - vwap) / vwap
        return distance_from_vwap < 0.002  # Within 0.2% of VWAP

    def _calculate_pattern_confidence(self, pattern_type: str, candles: pd.DataFrame) -> float:
        """Calculate confidence score for pattern"""
        base_confidence = 70

        # Adjust based on volume
        latest_volume = candles.iloc[-1]['tick_volume']
        avg_volume = candles['tick_volume'].mean()
        volume_confidence = min(20, (latest_volume / avg_volume - 1) * 100)

        # Adjust based on RSI
        rsi = candles.iloc[-1].get('rsi', 50)
        if 30 <= rsi <= 70:
            rsi_confidence = 10
        else:
            rsi_confidence = -10

        return min(95, base_confidence + volume_confidence + rsi_confidence)

    def query_similar_patterns(self, current_pattern: Dict, limit: int = 10) -> List[Dict]:
        """Query RAG for similar historical patterns"""
        try:
            # Construct search query
            query_text = f"""
            Pattern: {current_pattern['pattern']}
            Direction: {current_pattern['direction']}
            Session: {current_pattern['session']}
            RSI: {current_pattern['rsi']:.1f}
            VWAP Deviation: {current_pattern['vwap_deviation']:.2f}%
            Volume Ratio: {current_pattern['volume_ratio']:.2f}

            Show me similar patterns with actual outcomes, including:
            - Entry price
            - Stop loss level
            - Take profit level
            - Win rate
            - Average risk/reward ratio
            """

            # Enhanced query with LLM integration
            query_payload = {
                "query": query_text,
                "limit": limit,
                "max_context": 5,  # Request enhanced context for LLM analysis
                "filters": {
                    "pattern": current_pattern['pattern'],
                    "direction": current_pattern['direction'],
                    "session": current_pattern['session']
                }
            }

            response = self.session.post(
                f"{self.rag_base_url}/query",
                json=query_payload
            )

            if response.status_code == 200:
                result = response.json()

                # Check if we have enhanced LLM analysis
                if 'enhanced_recommendation' in result:
                    print("🧠 Using LLM-enhanced analysis")
                    return self._process_enhanced_response(result, current_pattern)
                else:
                    # Fallback to regular RAG results
                    return result.get('results', [])
            else:
                print(f"❌ RAG query failed: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Error querying RAG: {e}")
            return []

    def _process_enhanced_response(self, result: Dict, current_pattern: Dict) -> List[Dict]:
        """Process enhanced LLM+RAG response and return compatible format"""

        enhanced_recommendation = result.get('enhanced_recommendation', {})
        llm_insights = enhanced_recommendation.get('llm_insights', {})

        # Create enhanced result entry that's compatible with existing processing
        enhanced_entry = {
            'pattern': current_pattern['pattern'],
            'direction': current_pattern['direction'],
            'session': current_pattern['session'],
            'outcome': {
                'result': enhanced_recommendation.get('strategy', 'HOLD'),
                'confidence': enhanced_recommendation.get('enhanced_confidence', 50),
                'entry_price': enhanced_recommendation.get('entry_price'),
                'stop_loss': enhanced_recommendation.get('stop_loss'),
                'take_profit': enhanced_recommendation.get('take_profit'),
                'risk_reward_ratio': enhanced_recommendation.get('risk_reward_ratio', 0),
                'win_rate': enhanced_recommendation.get('confidence', 50) / 100,  # Convert to decimal
                'future_candles': []  # No future data in live analysis
            },
            'llm_analysis': {
                'market_analysis': enhanced_recommendation.get('market_analysis'),
                'risk_assessment': enhanced_recommendation.get('risk_assessment'),
                'confidence_level': llm_insights.get('confidence_level', 5),
                'key_levels': llm_insights.get('key_levels', []),
                'summary': llm_insights.get('summary', '')
            },
            'enhanced_signals': enhanced_recommendation.get('enhanced_signals', []),
            'analysis_sources': enhanced_recommendation.get('analysis_sources', ['RAG Database']),
            'reasoning': enhanced_recommendation.get('reasoning', '')
        }

        # Return as single-item list to maintain compatibility
        return [enhanced_entry]

    def generate_trade_recommendation(self, live_pattern: Dict, similar_patterns: List[Dict]) -> TradeRecommendation:
        """Generate comprehensive trade recommendation with LLM enhancement"""

        if not similar_patterns:
            # Fallback recommendation based on pattern
            return TradeRecommendation(
                direction=live_pattern['direction'],
                entry_price=live_pattern['entry_trigger'],
                stop_loss=live_pattern['stop_loss'],
                take_profit=self._calculate_default_take_profit(live_pattern),
                confidence=live_pattern['confidence'],
                reasoning=f"Based on {live_pattern['pattern']} pattern during {live_pattern['session']} session",
                similar_patterns=[],
                risk_reward_ratio=2.0,
                market_context=live_pattern
            )

        # Check if we have LLM-enhanced analysis
        llm_analysis = None
        enhanced_recommendation = None

        if similar_patterns and 'llm_analysis' in similar_patterns[0]:
            llm_analysis = similar_patterns[0]['llm_analysis']
            # Extract enhanced recommendation data if available
            if 'outcome' in similar_patterns[0]:
                outcome = similar_patterns[0]['outcome']
                enhanced_recommendation = {
                    'strategy': outcome.get('result', 'HOLD'),
                    'confidence': outcome.get('confidence', 50),
                    'entry_price': outcome.get('entry_price'),
                    'stop_loss': outcome.get('stop_loss'),
                    'take_profit': outcome.get('take_profit'),
                    'risk_reward_ratio': outcome.get('risk_reward_ratio', 2.0)
                }

        # Analyze similar patterns
        successful_trades = [p for p in similar_patterns if p.get('outcome', {}).get('result') == 'WIN']
        win_rate = len(successful_trades) / len(similar_patterns) if similar_patterns else 0

        # Calculate average levels from successful trades
        if successful_trades:
            avg_risk_reward = np.mean([p.get('outcome', {}).get('risk_reward_ratio', 2.0) for p in successful_trades])
            avg_win_rate = np.mean([p.get('outcome', {}).get('win_rate', 0.6) for p in successful_trades])
        else:
            avg_risk_reward = 2.0
            avg_win_rate = 0.5

        # Use enhanced recommendation if available, otherwise calculate from pattern
        if enhanced_recommendation:
            entry_price = enhanced_recommendation.get('entry_price', live_pattern['entry_trigger'])
            stop_loss = enhanced_recommendation.get('stop_loss', live_pattern['stop_loss'])
            take_profit = enhanced_recommendation.get('take_profit', self._calculate_default_take_profit(live_pattern))
            confidence = enhanced_recommendation.get('confidence', live_pattern['confidence'])
            risk_reward_ratio = enhanced_recommendation.get('risk_reward_ratio', avg_risk_reward)
        else:
            # Adjust confidence based on historical performance
            confidence = min(95, live_pattern['confidence'] * (0.5 + win_rate))
            entry_price = live_pattern['entry_trigger']
            stop_loss = live_pattern['stop_loss']
            take_profit = self._calculate_optimal_take_profit(entry_price, stop_loss, avg_risk_reward)
            risk_reward_ratio = avg_risk_reward

        # Build enhanced reasoning
        reasoning = f"""
{live_pattern['pattern']} detected during {live_pattern['session']} session

Historical Analysis:
- {len(similar_patterns)} similar patterns found
- Win rate: {win_rate*100:.1f}%
- Average risk/reward: {avg_risk_reward:.1f}:1

Current Conditions:
- RSI: {live_pattern['rsi']:.1f}
- VWAP Deviation: {live_pattern['vwap_deviation']:.2f}%
- Volume Ratio: {live_pattern['volume_ratio']:.2f}
"""

        # Add LLM insights if available
        if llm_analysis:
            reasoning += f"""
LLM Analysis (Confidence: {llm_analysis.get('confidence_level', 5)}/10):
{llm_analysis.get('summary', '')}

Key Levels: {', '.join([f'${level:.2f}' for level in llm_analysis.get('key_levels', [])[:3]])}
Analysis Sources: {', '.join(similar_patterns[0].get('analysis_sources', ['RAG Database']))}
"""

        return TradeRecommendation(
            direction=enhanced_recommendation.get('strategy', live_pattern['direction']) if enhanced_recommendation else live_pattern['direction'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning.strip(),
            similar_patterns=similar_patterns[:5],  # Top 5 similar patterns
            risk_reward_ratio=risk_reward_ratio,
            market_context={
                **live_pattern,
                'llm_analysis': llm_analysis,
                'enhanced_signals': similar_patterns[0].get('enhanced_signals', []) if llm_analysis else []
            }
        )

    def _calculate_default_take_profit(self, pattern: Dict) -> float:
        """Calculate default take profit based on pattern"""
        entry = pattern['entry_trigger']
        stop = pattern['stop_loss']
        risk = abs(entry - stop)

        if pattern['direction'] == 'BUY':
            return entry + risk * 2.0  # 2:1 risk/reward
        else:
            return entry - risk * 2.0

    def _calculate_optimal_take_profit(self, entry: float, stop: float, avg_risk_reward: float) -> float:
        """Calculate optimal take profit based on historical averages"""
        risk = abs(entry - stop)

        if entry > stop:  # Long position
            return entry + risk * avg_risk_reward
        else:  # Short position
            return entry - risk * avg_risk_reward

    def analyze_live_market(self, live_data_path: str) -> List[TradeRecommendation]:
        """Analyze live market data and generate trade recommendations"""
        print(f"\n🔍 Analyzing live market data: {live_data_path}")

        # Load live data
        try:
            df = pd.read_csv(live_data_path)

            # Handle both RAG format (DateTime) and MT5 format (timestamp)
            if 'DateTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['DateTime'])
                # Map RAG format columns to expected lowercase names
                column_mapping = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'tick_volume'
                }
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                print(f"✅ Loaded RAG format data: {len(df)} candles")
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Loaded MT5 format data: {len(df)} candles")
            else:
                print(f"❌ No valid timestamp column found. Columns: {list(df.columns)}")
                return []

            df.set_index('timestamp', inplace=True)

            if len(df) < 50:  # Need at least 50 candles for analysis
                print(f"❌ Insufficient data: {len(df)} candles (need 50+)")
                return []

        except Exception as e:
            print(f"❌ Error loading live data: {e}")
            return []

        # Calculate indicators
        df = self.calculate_live_indicators(df)

        # Detect patterns
        live_patterns = self.detect_live_patterns(df)

        if not live_patterns:
            print("📊 No patterns detected in current market")
            return []

        recommendations = []
        print(f"🎯 Detected {len(live_patterns)} potential patterns")

        for pattern in live_patterns:
            print(f"\n📈 Analyzing {pattern['pattern']} - {pattern['direction']}")

            # Query similar patterns from RAG
            similar_patterns = self.query_similar_patterns(pattern)

            # Generate recommendation
            recommendation = self.generate_trade_recommendation(pattern, similar_patterns)
            recommendations.append(recommendation)

            # Display recommendation
            self._display_recommendation(recommendation)

        return recommendations

    def _display_recommendation(self, rec: TradeRecommendation):
        """Display trade recommendation in console"""
        print(f"\n{'='*60}")
        print(f"🎯 TRADE RECOMMENDATION - {rec.direction}")
        print(f"{'='*60}")

        # Safely format values that might be None
        if rec.entry_price:
            print(f"📍 Entry Price: ${rec.entry_price:.5f}")
        else:
            print("📍 Entry Price: N/A")

        if rec.stop_loss:
            print(f"🛑 Stop Loss:   ${rec.stop_loss:.5f}")
        else:
            print("🛑 Stop Loss:   N/A")

        if rec.take_profit:
            print(f"🎯 Take Profit: ${rec.take_profit:.5f}")
        else:
            print("🎯 Take Profit: N/A")

        if rec.risk_reward_ratio:
            print(f"📊 Risk/Reward:  {rec.risk_reward_ratio:.1f}:1")
        else:
            print("📊 Risk/Reward:  N/A")

        if rec.confidence:
            print(f"💪 Confidence:   {rec.confidence:.0f}%")
        else:
            print("💪 Confidence:   N/A")

        print(f"\n📋 Reasoning:")
        print(rec.reasoning)

        if rec.similar_patterns:
            print(f"\n📈 Similar Historical Patterns: {len(rec.similar_patterns)}")
            for i, pattern in enumerate(rec.similar_patterns[:3], 1):
                outcome = pattern.get('outcome', {}).get('result', 'UNKNOWN')
                pattern_date = pattern.get('date', 'N/A')
                print(f"  {i}. {pattern_date} - {outcome}")

                # Show LLM enhancement if available
                if pattern.get('llm_analysis'):
                    llm_conf = pattern['llm_analysis'].get('confidence_level', 'N/A')
                    print(f"      🧠 LLM Confidence: {llm_conf}/10")

                if pattern.get('enhanced_signals'):
                    signals = pattern['enhanced_signals']
                    print(f"      🚨 Enhanced Signals: {', '.join(signals[:2])}")  # Show first 2

        # Show enhanced market context if available
        if hasattr(rec, 'market_context') and rec.market_context:
            context = rec.market_context
            if context.get('llm_analysis'):
                print(f"\n🧠 LLM Analysis Summary:")
                llm_summary = context['llm_analysis'].get('summary', 'No summary available')
                print(f"   {llm_summary[:200]}...")

            if context.get('enhanced_signals'):
                print(f"\n🚨 Enhanced Signals:")
                for signal in context['enhanced_signals']:
                    print(f"   • {signal}")

class TechnicalIndicators:
    """Technical indicators calculator matching enhanced processor"""

    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

        # EMAs
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Volume indicators
        df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']

        return df

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced indicators"""
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

        # CCI
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        return df

    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-specific VWAP"""
        df = df.copy()
        df['date'] = df.index.date
        df['hour'] = df.index.hour

        # Daily VWAP - simplified approach to avoid pandas issues
        df['daily_vwap'] = df['close'].expanding().mean()  # Use expanding mean as simple alternative

        # Simple VWAP for live analysis
        df['vwap'] = (df['close'] * df['tick_volume']).rolling(window=50).sum() / df['tick_volume'].rolling(window=50).sum()

        return df

    def calculate_market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market profile metrics"""
        df = df.copy()

        # Simple POC calculation (highest volume price in recent window)
        # Since rolling apply works on individual columns, we'll use a different approach
        df['poc_price'] = np.nan

        for i in range(49, len(df)):
            window = df.iloc[i-49:i+1]  # 50-bar window
            if not window.empty and not window['tick_volume'].isna().all():
                max_vol_idx = window['tick_volume'].idxmax()
                if not pd.isna(max_vol_idx) and max_vol_idx in window.index:
                    df.iloc[i, df.columns.get_loc('poc_price')] = window.loc[max_vol_idx, 'close']

        return df

def main():
    """Main function for live trading analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Live Trading Analysis')
    parser.add_argument('--data', required=True, help='Path to live CSV data file')
    parser.add_argument('--rag-url', default='http://localhost:8000', help='RAG system URL')
    parser.add_argument('--output', help='Save recommendations to file')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = LiveTradingAnalyzer(rag_base_url=args.rag_url)

    # Analyze live market
    recommendations = analyzer.analyze_live_market(args.data)

    if recommendations:
        print(f"\n✅ Generated {len(recommendations)} trade recommendations")

        # Save to file if requested
        if args.output:
            output_data = []
            for rec in recommendations:
                output_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'direction': rec.direction,
                    'entry_price': rec.entry_price,
                    'stop_loss': rec.stop_loss,
                    'take_profit': rec.take_profit,
                    'confidence': rec.confidence,
                    'risk_reward_ratio': rec.risk_reward_ratio,
                    'reasoning': rec.reasoning,
                    'similar_patterns_count': len(rec.similar_patterns)
                })

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Recommendations saved to: {args.output}")
    else:
        print("📊 No trade recommendations generated")

if __name__ == "__main__":
    main()