#!/usr/bin/env python3
"""
MT5 Live Data Exporter
Automates exporting live data from MT5 with correct formatting
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 library not installed. Install with: pip install MetaTrader5")

class MT5DataExporter:
    """Automated MT5 data exporter for live trading system"""

    def __init__(self, output_dir: str = "./data/live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Live file path (always overwritten)
        self.live_file = self.output_dir / "XAUUSD_M15_LIVE.csv"

        # Snapshot file path (dated)
        today = datetime.now().strftime("%Y-%m-%d")
        self.snapshot_file = self.output_dir / f"XAUUSD_M15_{today}.csv"

    def export_live_data(self, days_back: int = 7, symbol: str = "XAUUSD") -> bool:
        """Export live data from MT5 with correct formatting"""

        if not MT5_AVAILABLE:
            print("❌ MT5 library not available")
            return False

        print(f"🔗 Connecting to MT5...")
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return False

        try:
            print(f"📊 Exporting {symbol} M15 data for last {days_back} days...")

            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Request M15 data
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_time, end_time)

            if rates is None or len(rates) == 0:
                print(f"❌ No data received for {symbol}")
                return False

            print(f"✅ Received {len(rates)} candles from MT5")

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to readable format
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')

            # Select and rename columns to match required format
            df_formatted = pd.DataFrame({
                'timestamp': df['timestamp'],
                'open': df['open'].round(2),
                'high': df['high'].round(2),
                'low': df['low'].round(2),
                'close': df['close'].round(2),
                'tick_volume': df['tick_volume'].astype(int),
                'spread': df['spread'].round(1),
                'real_volume': df['real_volume'].fillna(0).astype(int)
            })

            # Sort by timestamp (newest first)
            df_formatted = df_formatted.sort_values('timestamp', ascending=False)

            # Keep only last 500 candles (most recent)
            df_formatted = df_formatted.head(500)

            # Save live file (always overwrite)
            df_formatted.to_csv(self.live_file, index=False)
            print(f"💾 Live data saved: {self.live_file}")
            print(f"📅 Date range: {df_formatted['timestamp'].min()} to {df_formatted['timestamp'].max()}")
            print(f"📊 Candles: {len(df_formatted)}")

            # Save snapshot (preserve daily record)
            if not self.snapshot_file.exists():
                df_formatted.to_csv(self.snapshot_file, index=False)
                print(f"📸 Snapshot saved: {self.snapshot_file}")

            # Display sample
            print("\n📋 Sample data:")
            print(df_formatted.head(3).to_string(index=False))

            return True

        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False

        finally:
            mt5.shutdown()

    def validate_exported_data(self) -> bool:
        """Validate the exported data meets requirements"""
        if not self.live_file.exists():
            print(f"❌ Live file not found: {self.live_file}")
            return False

        try:
            df = pd.read_csv(self.live_file)

            # Check requirements
            issues = []

            # Check columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            if not all(col in df.columns for col in required_columns):
                issues.append(f"Missing columns. Need: {required_columns}")

            # Check row count
            if len(df) < 100:
                issues.append(f"Too few candles: {len(df)} (minimum 100)")
            elif len(df) > 1000:
                issues.append(f"Too many candles: {len(df)} (maximum 1000)")

            # Check timestamp format
            try:
                pd.to_datetime(df['timestamp'])
            except:
                issues.append("Invalid timestamp format")

            # Check data freshness
            latest_time = pd.to_datetime(df['timestamp']).max()
            now = datetime.now()
            if (now - latest_time).total_seconds() > 3600:  # More than 1 hour old
                issues.append(f"Data too old: {latest_time} (latest should be within 1 hour)")

            if issues:
                print("❌ Validation issues:")
                for issue in issues:
                    print(f"  • {issue}")
                return False

            print("✅ Data validation passed")
            return True

        except Exception as e:
            print(f"❌ Error validating data: {e}")
            return False

    def generate_manual_instructions(self) -> str:
        """Generate instructions for manual MT5 export"""

        instructions = f"""
📋 Manual MT5 Export Instructions
================================

If automated export fails, follow these steps:

1️⃣ Open MT5 Terminal
2️⃣ Press F2 or go to Tools → History Center
3️⃣ Navigate to XAUUSD → 15 Minutes
4️⃣ Set date range: Last 7 days
5️⃣ Click "Export" or "Save as CSV"
6️⃣ Save file to: {self.live_file}
7️⃣ Verify columns are in this order:
   timestamp,open,high,low,close,tick_volume,spread,real_volume

8️⃣ Run validation:
   python scripts/mt5_data_exporter.py --validate

Current target file: {self.live_file}
"""
        return instructions

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='MT5 Live Data Exporter')
    parser.add_argument('--days', type=int, default=7, help='Days of data to export (default: 7)')
    parser.add_argument('--symbol', default='XAUUSD', help='Symbol to export (default: XAUUSD)')
    parser.add_argument('--output-dir', default='./data/live', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Validate existing data only')
    parser.add_argument('--manual', action='store_true', help='Show manual export instructions')

    args = parser.parse_args()

    exporter = MT5DataExporter(args.output_dir)

    if args.manual:
        print(exporter.generate_manual_instructions())
        return

    if args.validate:
        print(f"🔍 Validating existing data: {exporter.live_file}")
        success = exporter.validate_exported_data()
        sys.exit(0 if success else 1)

    print("🚀 MT5 Live Data Exporter")
    print("=" * 40)

    # Export data
    success = exporter.export_live_data(days_back=args.days, symbol=args.symbol)

    if success:
        # Validate exported data
        print(f"\n🔍 Validating exported data...")
        validation_success = exporter.validate_exported_data()

        if validation_success:
            print(f"\n✅ Success! Live data ready for trading system")
            print(f"📁 File location: {exporter.live_file}")
            print(f"\n🎯 Next steps:")
            print(f"1. Start live trading: python scripts/setup_live_trading.py start")
            print(f"2. Or test analysis: python scripts/live_trading_analyzer.py --data {exporter.live_file}")
        else:
            print(f"\n❌ Data exported but validation failed")
            print(f"🔧 Check file format and try again")
    else:
        print(f"\n❌ Automated export failed")
        print(f"📋 Try manual export:")
        print(exporter.generate_manual_instructions())

if __name__ == "__main__":
    main()