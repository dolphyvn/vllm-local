#!/usr/bin/env python3
"""
Complete Live Trading Setup Script
Configures and starts the entire live trading analysis pipeline
"""

import os
import sys
import time
import json
import requests
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class LiveTradingSetup:
    """Complete setup and management of live trading system"""

    def __init__(self, config_file: str = "./config/live_trading.json"):
        self.config_file = Path(config_file)
        self.processes = []
        self.config = self.load_config()

        # Ensure directories exist
        Path("./data/live").mkdir(exist_ok=True)
        Path("./config").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "data_source": "./data/XAUUSD_PERIOD_M15_0.csv",
            "rag_base_url": "http://localhost:8080",
            "live_data_dir": "./data/live",
            "update_interval": 60,
            "alert_check_interval": 60,
            "alert_config": {
                "min_confidence": 70,
                "min_risk_reward": 1.5,
                "enabled_sessions": ["London", "New York", "Asia/London Overlap", "London/NY Overlap"],
                "max_alerts_per_hour": 10,
                "cooldown_minutes": 30
            },
            "webhook_url": None,
            "timezone": "UTC"
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
            except Exception as e:
                print(f"⚠️ Error loading config, using defaults: {e}")

        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        print("🔍 Checking dependencies...")

        required_packages = [
            'pandas', 'numpy', 'ta', 'requests', 'chromadb'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - NOT INSTALLED")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install"
                ] + missing_packages, check=True)
                print("✅ Packages installed successfully")
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages. Please install manually:")
                print(f"pip install {' '.join(missing_packages)}")
                return False

        # Check if RAG system is running
        try:
            response = requests.get(f"{self.config['rag_base_url']}/health", timeout=5)
            if response.status_code == 200:
                print("✅ RAG system is running")
            else:
                print(f"⚠️ RAG system responded with status {response.status_code}")
        except:
            print(f"❌ RAG system not responding at {self.config['rag_base_url']}")
            print("Please ensure your RAG system is running before starting live trading")

        return True

    def verify_rag_data(self) -> bool:
        """Verify RAG system has trading data"""
        try:
            response = requests.get(f"{self.config['rag_base_url']}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                total_docs = stats.get('total_documents', 0)
                if total_docs > 0:
                    print(f"✅ RAG system has {total_docs} documents")
                    return True
                else:
                    print("❌ RAG system has no data")
                    print("Please feed processed trading data to RAG first:")
                    print("python scripts/feed_to_rag_direct.py --file data/XAUUSD_PERIOD_M15_0_processed.json")
                    return False
            else:
                print(f"❌ RAG stats endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error checking RAG data: {e}")
            return False

    def start_live_feeder(self) -> subprocess.Popen:
        """Start the live data feeder"""
        print("🚀 Starting live data feeder...")

        cmd = [
            sys.executable, "scripts/live_data_feeder.py",
            "--source", self.config['data_source'],
            "--output-dir", self.config['live_data_dir'],
            "--interval", str(self.config['update_interval'])
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        return process

    def start_alert_system(self) -> subprocess.Popen:
        """Start the alert system"""
        print("🚀 Starting alert system...")

        live_data_file = Path(self.config['live_data_dir']) / "XAUUSD_M15_LIVE.csv"

        cmd = [
            sys.executable, "scripts/live_alert_system.py",
            "--live-data", str(live_data_file),
            "--rag-url", self.config['rag_base_url'],
            "--interval", str(self.config['alert_check_interval']),
            "--min-confidence", str(self.config['alert_config']['min_confidence']),
            "--min-risk-reward", str(self.config['alert_config']['min_risk_reward'])
        ]

        if self.config['alert_config']['enabled_sessions']:
            cmd.extend(["--sessions"] + self.config['alert_config']['enabled_sessions'])

        if self.config['webhook_url']:
            cmd.extend(["--webhook", self.config['webhook_url']])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        return process

    def start_all(self):
        """Start all live trading components"""
        print("🎯 Starting complete live trading system...")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed. Exiting.")
            return

        # Verify RAG data
        if not self.verify_rag_data():
            print("❌ RAG data verification failed. Exiting.")
            return

        # Check data source exists
        if not Path(self.config['data_source']).exists():
            print(f"❌ Data source not found: {self.config['data_source']}")
            print("Please ensure you have processed historical data first")
            return

        try:
            # Start live data feeder
            feeder_process = self.start_live_feeder()
            self.processes.append(("Live Data Feeder", feeder_process))

            # Wait a moment for feeder to initialize
            time.sleep(3)

            # Start alert system
            alert_process = self.start_alert_system()
            self.processes.append(("Alert System", alert_process))

            print("="*60)
            print("✅ Live trading system started successfully!")
            print(f"📊 Live data directory: {self.config['live_data_dir']}")
            print(f"🔔 RAG URL: {self.config['rag_base_url']}")
            print(f"⚙️ Config file: {self.config_file}")
            print("\n📝 System logs are being displayed below.")
            print("🛑 Press Ctrl+C to stop all components")
            print("="*60)

            # Monitor processes
            self.monitor_processes()

        except KeyboardInterrupt:
            print("\n⏹️ Received interrupt signal...")
        except Exception as e:
            print(f"❌ Error starting system: {e}")
        finally:
            self.stop_all()

    def monitor_processes(self):
        """Monitor running processes and display their output"""
        def signal_handler(signum, frame):
            print("\n⏹️ Received signal to stop...")
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while True:
                # Check if any process has stopped
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"❌ {name} has stopped unexpectedly (exit code: {process.poll()})")
                        # Read any remaining output
                        output, _ = process.communicate()
                        if output:
                            print(f"📝 {name} output:\n{output}")

                time.sleep(5)

        except KeyboardInterrupt:
            pass

    def stop_all(self):
        """Stop all running processes"""
        print("\n🛑 Stopping all components...")

        for name, process in self.processes:
            try:
                print(f"⏹️ Stopping {name}...")
                process.terminate()
                process.wait(timeout=10)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {name}...")
                process.kill()
                process.wait()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")

        self.processes.clear()
        print("✅ All components stopped")

    def test_system(self):
        """Test the live trading system"""
        print("🧪 Testing live trading system...")

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Verify RAG data
        if not self.verify_rag_data():
            return False

        # Test live analyzer
        try:
            from live_trading_analyzer import LiveTradingAnalyzer

            # Create test live data file
            test_file = Path(self.config['live_data_dir']) / "XAUUSD_M15_TEST.csv"

            # Generate test data if it doesn't exist
            if not test_file.exists():
                print("📊 Generating test data...")
                from live_data_feeder import LiveDataFeeder
                feeder = LiveDataFeeder(self.config['data_source'], str(test_file.parent))
                test_data = feeder.create_live_dataset(100)
                test_data.to_csv(test_file)
                print(f"✅ Test data generated: {test_file}")

            # Test analyzer
            analyzer = LiveTradingAnalyzer(self.config['rag_base_url'])
            recommendations = analyzer.analyze_live_market(str(test_file))

            print(f"✅ Test completed. Generated {len(recommendations)} recommendations")
            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Live Trading Setup')
    parser.add_argument('action', choices=['start', 'stop', 'test', 'config'],
                       help='Action to perform')
    parser.add_argument('--config', default='./config/live_trading.json',
                       help='Configuration file path')

    args = parser.parse_args()

    setup = LiveTradingSetup(args.config)

    if args.action == 'start':
        setup.start_all()
    elif args.action == 'stop':
        setup.stop_all()
    elif args.action == 'test':
        success = setup.test_system()
        sys.exit(0 if success else 1)
    elif args.action == 'config':
        print(f"📋 Configuration file: {args.config}")
        print(json.dumps(setup.config, indent=2))

if __name__ == "__main__":
    main()