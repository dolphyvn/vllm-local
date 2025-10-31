#!/usr/bin/env python3
"""
Live Data Feeder for Real-time Market Analysis
Simulates and manages live market data updates
"""

import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import random
import threading
from pathlib import Path

class LiveDataFeeder:
    """Manages live market data updates and simulation"""

    def __init__(self, source_file: str, output_dir: str = "./data/live", update_interval: int = 60):
        self.source_file = source_file
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval  # seconds
        self.is_running = False
        self.current_data = None
        self.update_thread = None

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        print("📡 Live Data Feeder initialized")
        print(f"📂 Source: {source_file}")
        print(f"💾 Output: {output_dir}")
        print(f"⏰ Update interval: {update_interval}s")

    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data for simulation"""
        try:
            df = pd.read_csv(self.source_file)

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
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                print(f"❌ No valid timestamp column found. Columns: {list(df.columns)}")
                return pd.DataFrame()

            df.set_index('timestamp', inplace=True)
            return df.sort_index()
        except Exception as e:
            print(f"❌ Error loading source data: {e}")
            return pd.DataFrame()

    def generate_live_candle(self, last_candle: pd.Series, market_noise: float = 0.001) -> Dict:
        """Generate a realistic new candle based on last candle"""

        # Base price movement with market noise
        price_change = np.random.normal(0, market_noise)

        # Add session-specific volatility
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 16:  # High volatility hours
            volatility_multiplier = 1.5
        else:  # Low volatility hours
            volatility_multiplier = 0.8

        price_change *= volatility_multiplier

        # Calculate OHLC
        last_close = last_candle['close']
        new_open = last_close + np.random.normal(0, market_noise * 0.5)

        # High and Low based on price movement
        if price_change > 0:
            new_high = new_open + abs(price_change) * random.uniform(1.2, 2.0)
            new_low = new_open + abs(price_change) * random.uniform(-0.3, 0.2)
        else:
            new_high = new_open + abs(price_change) * random.uniform(-0.2, 0.3)
            new_low = new_open - abs(price_change) * random.uniform(1.2, 2.0)

        new_close = new_open + price_change

        # Generate realistic volume
        base_volume = last_candle['tick_volume']
        volume_change = random.uniform(0.5, 2.0)
        new_volume = int(base_volume * volume_change)

        return {
            'timestamp': datetime.now(timezone.utc),
            'open': round(new_open, 2),
            'high': round(new_high, 2),
            'low': round(new_low, 2),
            'close': round(new_close, 2),
            'tick_volume': new_volume,
            'spread': random.uniform(0.5, 2.0),
            'real_volume': int(new_volume * random.uniform(10, 50))
        }

    def create_live_dataset(self, num_candles: int = 200) -> pd.DataFrame:
        """Create initial live dataset from historical data"""
        print(f"📊 Creating live dataset with {num_candles} recent candles")

        # Load historical data
        hist_data = self.load_historical_data()
        if hist_data.empty:
            return pd.DataFrame()

        # Take last N candles
        live_data = hist_data.tail(num_candles).copy()

        # Update timestamps to recent times
        now = datetime.now(timezone.utc)
        time_delta = timedelta(minutes=15)  # M15 timeframe

        for i in range(len(live_data)):
            live_data.index.values[i] = now - (len(live_data) - i) * time_delta

        return live_data.reset_index().rename(columns={'index': 'timestamp'})

    def update_live_data(self):
        """Update live data with new candle"""
        if self.current_data is None or len(self.current_data) == 0:
            print("❌ No current data to update")
            return

        # Generate new candle
        last_candle = self.current_data.iloc[-1]
        new_candle = self.generate_live_candle(last_candle)

        # Append new candle
        new_row = pd.DataFrame([new_candle])
        new_row.set_index('timestamp', inplace=True)

        # Keep only last 200 candles
        self.current_data = pd.concat([self.current_data.iloc[-199:], new_row])

        # Save updated data
        self.save_live_data()

        print(f"📈 Updated live data: {new_candle['close']} @ {new_candle['timestamp'].strftime('%H:%M:%S')}")

    def save_live_data(self):
        """Save current live data to file"""
        if self.current_data is not None:
            output_file = self.output_dir / "XAUUSD_M15_LIVE.csv"
            self.current_data.to_csv(output_file)

            # Also save latest snapshot
            snapshot_file = self.output_dir / f"XAUUSD_M15_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.current_data.to_csv(snapshot_file)

    def start_live_feed(self):
        """Start the live data feed simulation"""
        if self.is_running:
            print("⚠️ Live feed is already running")
            return

        # Initialize with historical data
        self.current_data = self.create_live_dataset()
        if self.current_data.empty:
            print("❌ Failed to initialize live data")
            return

        self.is_running = True
        print("🚀 Starting live data feed...")

        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        """Main update loop for live data"""
        while self.is_running:
            try:
                self.update_live_data()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"❌ Error in live update: {e}")
                time.sleep(5)  # Wait 5 seconds before retrying

    def stop_live_feed(self):
        """Stop the live data feed"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("⏹️ Live data feed stopped")

    def get_current_price(self) -> Optional[float]:
        """Get current price from live data"""
        if self.current_data is not None and len(self.current_data) > 0:
            return self.current_data.iloc[-1]['close']
        return None

    def get_market_status(self) -> Dict:
        """Get current market status"""
        if self.current_data is None or len(self.current_data) == 0:
            return {"status": "No data"}

        last_candle = self.current_data.iloc[-1]
        prev_candle = self.current_data.iloc[-2] if len(self.current_data) > 1 else last_candle

        price_change = last_candle['close'] - prev_candle['close']
        price_change_pct = (price_change / prev_candle['close']) * 100

        return {
            "status": "Live",
            "current_price": last_candle['close'],
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "volume": last_candle['tick_volume'],
            "timestamp": last_candle.name,
            "session": self._get_current_session()
        }

    def _get_current_session(self) -> str:
        """Determine current trading session"""
        hour = datetime.now().hour

        if 0 <= hour < 2:
            return "Late Asia"
        elif 2 <= hour < 6:
            return "Asia"
        elif 6 <= hour < 8:
            return "Asia/London Overlap"
        elif 8 <= hour < 12:
            return "London"
        elif 12 <= hour < 13:
            return "London/NY Overlap"
        elif 13 <= hour < 17:
            return "New York"
        elif 17 <= hour < 20:
            return "Late New York"
        else:
            return "Quiet Hours"

def main():
    """Main function for live data feeder"""
    import argparse

    parser = argparse.ArgumentParser(description='Live Data Feeder')
    parser.add_argument('--source', required=True, help='Source CSV file with historical data')
    parser.add_argument('--output-dir', default='./data/live', help='Output directory for live data')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--duration', type=int, help='Run for specified duration in minutes')
    parser.add_argument('--init-candles', type=int, default=200, help='Number of candles to initialize with')

    args = parser.parse_args()

    # Initialize feeder
    feeder = LiveDataFeeder(
        source_file=args.source,
        output_dir=args.output_dir,
        update_interval=args.interval
    )

    try:
        # Start live feed
        feeder.start_live_feed()

        # Run for specified duration or indefinitely
        if args.duration:
            print(f"⏰ Running for {args.duration} minutes...")
            time.sleep(args.duration * 60)
            feeder.stop_live_feed()
        else:
            print("🔄 Running indefinitely. Press Ctrl+C to stop...")
            while True:
                time.sleep(10)
                status = feeder.get_market_status()
                print(f"📊 {status['session']} - ${status['current_price']:.2f} ({status['price_change_pct']:+.2f}%)")

    except KeyboardInterrupt:
        print("\n⏹️ Stopping live feed...")
        feeder.stop_live_feed()

    print("✅ Live data feeder stopped")

if __name__ == "__main__":
    main()