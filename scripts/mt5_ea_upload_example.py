#!/usr/bin/env python3
"""
MT5 EA Upload Example
Demonstrates how to upload CSV files from MT5 EA to the FastAPI /upload endpoint
"""

import requests
import json
from pathlib import Path

class MT5EAUploader:
    """Example MT5 EA uploader for CSV data"""

    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url.rstrip('/')
        self.upload_url = f"{self.api_base_url}/upload"
        self.status_url = f"{self.api_base_url}/upload/status"

    def upload_csv_file(self, csv_file_path: str, symbol: str, timeframe: str, candles: int) -> dict:
        """
        Upload a CSV file to the FastAPI endpoint

        Args:
            csv_file_path: Path to the CSV file
            symbol: Trading symbol (e.g., 'XAUUSD', 'BTCUSD')
            timeframe: Timeframe (e.g., 'M1', 'M5', 'M15', 'H1')
            candles: Number of candles in the file

        Returns:
            Dictionary with upload result
        """
        try:
            # Validate file exists
            file_path = Path(csv_file_path)
            if not file_path.exists():
                return {
                    "success": False,
                    "message": f"File not found: {csv_file_path}"
                }

            # Prepare the upload data
            files = {
                'file': (file_path.name, open(file_path, 'rb'), 'text/csv')
            }

            data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': str(candles)
            }

            print(f"📤 Uploading {symbol} {timeframe} {candles} candles...")
            print(f"📁 File: {csv_file_path}")

            # Send POST request to /upload endpoint
            response = requests.post(
                self.upload_url,
                files=files,
                data=data,
                timeout=30
            )

            # Close the file
            files['file'][1].close()

            # Parse response
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Upload successful!")
                    print(f"📄 Saved as: {result.get('filename')}")
                    print(f"📊 Actual rows: {result.get('actual_rows')}")
                    print(f"💾 File size: {result.get('file_size')} bytes")
                    if result.get('live_updated'):
                        print(f"🔄 Live data updated for trading system")
                else:
                    print(f"❌ Upload failed: {result.get('message')}")
                return result
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                return {
                    "success": False,
                    "message": f"HTTP Error {response.status_code}: {response.text}"
                }

        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg
            }

    def get_upload_status(self) -> dict:
        """
        Get the status of all uploaded files

        Returns:
            Dictionary with upload status
        """
        try:
            print("📊 Getting upload status...")
            response = requests.get(self.status_url, timeout=10)

            if response.status_code == 200:
                status = response.json()

                print(f"📁 Total MT5 files: {status.get('total_mt5_files', 0)}")
                print(f"🔄 Total live files: {status.get('total_live_files', 0)}")

                # Show MT5 files
                mt5_files = status.get('mt5_files', [])
                if mt5_files:
                    print("\n📈 MT5 Data Files:")
                    for file in mt5_files:
                        print(f"  • {file['symbol']} {file['timeframe']} ({file['candles']} candles)")
                        print(f"    File: {file['filename']} ({file['size']} bytes)")
                        print(f"    Modified: {file['modified']}")

                # Show live files
                live_files = status.get('live_files', [])
                if live_files:
                    print("\n🔄 Live Data Files:")
                    for file in live_files:
                        print(f"  • {file['symbol']} M15 LIVE")
                        print(f"    File: {file['filename']} ({file['size']} bytes)")
                        print(f"    Modified: {file['modified']}")

                return status
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                return {"success": False, "message": f"HTTP Error {response.status_code}"}

        except Exception as e:
            error_msg = f"Failed to get status: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "message": error_msg}

def create_sample_csv():
    """Create a sample CSV file for testing"""
    import pandas as pd
    from datetime import datetime, timedelta

    print("📝 Creating sample CSV file for testing...")

    # Generate sample data
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15 * 200)  # 200 M15 candles

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    spreads = []
    real_volumes = []

    base_price = 1840.0

    for i in range(200):
        current_time = start_time + timedelta(minutes=15 * i)
        timestamps.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))

        # Generate realistic OHLC data
        price_change = (hash(str(i)) % 100 - 50) / 1000  # Random price change
        open_price = base_price + price_change
        high_price = open_price + abs((hash(str(i * 2)) % 50)) / 100
        low_price = open_price - abs((hash(str(i * 3)) % 50)) / 100
        close_price = open_price + (hash(str(i * 4)) % 20 - 10) / 100

        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))
        volumes.append(100 + (hash(str(i * 5)) % 200))
        spreads.append(1.0 + (hash(str(i * 6)) % 20) / 10)
        real_volumes.append(volumes[-1] * 50)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'tick_volume': volumes,
        'spread': spreads,
        'real_volume': real_volumes
    })

    # Save to file
    filename = "data/sample_XAUUSD_M15_200.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Sample CSV created: {filename}")
    print(f"📊 Contains {len(df)} rows of XAUUSD M15 data")

    return filename

def main():
    """Main function demonstrating MT5 EA upload"""
    import argparse

    parser = argparse.ArgumentParser(description='MT5 EA CSV Upload Example')
    parser.add_argument('--file', help='CSV file to upload')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol')
    parser.add_argument('--timeframe', default='M15', help='Timeframe')
    parser.add_argument('--candles', type=int, help='Number of candles')
    parser.add_argument('--url', default='http://localhost:8080', help='API base URL')
    parser.add_argument('--create-sample', action='store_true', help='Create sample CSV for testing')
    parser.add_argument('--status', action='store_true', help='Show upload status only')

    args = parser.parse_args()

    # Initialize uploader
    uploader = MT5EAUploader(args.url)

    # Show status if requested
    if args.status:
        uploader.get_upload_status()
        return

    # Create sample file if requested
    if args.create_sample:
        args.file = create_sample_csv()
        args.symbol = 'XAUUSD'
        args.timeframe = 'M15'
        args.candles = 200

    # Validate required arguments
    if not args.file:
        print("❌ Please provide a CSV file with --file or use --create-sample")
        return

    if not args.candles:
        print("❌ Please provide candle count with --candles")
        return

    # Upload the file
    result = uploader.upload_csv_file(
        csv_file_path=args.file,
        symbol=args.symbol,
        timeframe=args.timeframe,
        candles=args.candles
    )

    # Show status after upload
    if result.get('success'):
        print("\n" + "="*50)
        uploader.get_upload_status()

if __name__ == "__main__":
    main()