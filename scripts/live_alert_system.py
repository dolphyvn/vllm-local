#!/usr/bin/env python3
"""
Live Alert System for Real-time Trade Notifications
Monitors live data and sends alerts when patterns are detected
"""

import os
import time
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
from pathlib import Path

from live_trading_analyzer import LiveTradingAnalyzer, TradeRecommendation

@dataclass
class AlertConfig:
    """Configuration for alerts"""
    min_confidence: float = 70
    min_risk_reward: float = 1.5
    enabled_sessions: List[str] = None
    max_alerts_per_hour: int = 10
    cooldown_minutes: int = 30

class LiveAlertSystem:
    """Real-time alert system for trade opportunities"""

    def __init__(self, rag_base_url: str = "http://localhost:8000",
                 alert_config: AlertConfig = None,
                 webhook_url: str = None):
        self.rag_base_url = rag_base_url
        self.config = alert_config or AlertConfig()
        self.webhook_url = webhook_url
        self.analyzer = LiveTradingAnalyzer(rag_base_url)

        # Alert tracking
        self.recent_alerts = []
        self.last_alert_time = None
        self.alerts_sent_today = []
        self.is_running = False
        self.alert_thread = None

        print("🚨 Live Alert System initialized")
        print(f"📊 RAG URL: {rag_base_url}")
        print(f"⚙️ Min Confidence: {self.config.min_confidence}%")
        print(f"💰 Min Risk/Reward: {self.config.min_risk_reward}:1")
        print(f"🔔 Webhook: {webhook_url or 'None'}")

    def check_alert_conditions(self, recommendation: TradeRecommendation) -> bool:
        """Check if recommendation meets alert criteria"""

        # Check confidence threshold
        if recommendation.confidence < self.config.min_confidence:
            return False

        # Check risk/reward ratio
        if recommendation.risk_reward_ratio < self.config.min_risk_reward:
            return False

        # Check session filter
        current_session = recommendation.market_context.get('session', '')
        if self.config.enabled_sessions and current_session not in self.config.enabled_sessions:
            return False

        # Check cooldown period
        if self.last_alert_time:
            time_since_last = (datetime.now(timezone.utc) - self.last_alert_time).total_seconds() / 60
            if time_since_last < self.config.cooldown_minutes:
                return False

        # Check hourly limit
        current_hour = datetime.now().hour
        alerts_this_hour = len([a for a in self.recent_alerts
                              if a['timestamp'].hour == current_hour])
        if alerts_this_hour >= self.config.max_alerts_per_hour:
            return False

        return True

    def create_alert_message(self, recommendation: TradeRecommendation) -> Dict:
        """Create formatted alert message"""

        emoji = "🟢" if recommendation.direction == "BUY" else "🔴"

        message = f"""
{emoji} **{recommendation.direction} SIGNAL DETECTED**

📍 **Entry:** ${recommendation.entry_price:.2f}
🛑 **Stop Loss:** ${recommendation.stop_loss:.2f}
🎯 **Take Profit:** ${recommendation.take_profit:.2f}
📊 **Risk/Reward:** {recommendation.risk_reward_ratio:.1f}:1
💪 **Confidence:** {recommendation.confidence:.0f}%

📋 **Details:**
• Pattern: {recommendation.market_context.get('pattern', 'N/A')}
• Session: {recommendation.market_context.get('session', 'N/A')}
• RSI: {recommendation.market_context.get('rsi', 0):.1f}
• VWAP Deviation: {recommendation.market_context.get('vwap_deviation', 0):.2f}%

📈 **Similar Patterns:** {len(recommendation.similar_patterns)} found
🕐 **Time:** {datetime.now().strftime('%H:%M:%S UTC')}
        """

        return {
            "type": "trade_alert",
            "direction": recommendation.direction,
            "entry_price": recommendation.entry_price,
            "stop_loss": recommendation.stop_loss,
            "take_profit": recommendation.take_profit,
            "confidence": recommendation.confidence,
            "risk_reward_ratio": recommendation.risk_reward_ratio,
            "pattern": recommendation.market_context.get('pattern', 'N/A'),
            "session": recommendation.market_context.get('session', 'N/A'),
            "message": message.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def send_alert(self, alert_data: Dict):
        """Send alert via webhook or console"""

        # Log alert
        self.recent_alerts.append({
            'timestamp': datetime.now(timezone.utc),
            'direction': alert_data['direction'],
            'confidence': alert_data['confidence'],
            'pattern': alert_data['pattern']
        })
        self.last_alert_time = datetime.now(timezone.utc)

        # Send webhook if configured
        if self.webhook_url:
            try:
                response = requests.post(self.webhook_url, json=alert_data, timeout=10)
                if response.status_code == 200:
                    print(f"📤 Alert sent via webhook: {alert_data['direction']} @ ${alert_data['entry_price']:.2f}")
                else:
                    print(f"❌ Webhook failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Error sending webhook: {e}")

        # Always display in console
        print("\n" + "="*60)
        print("🚨 TRADE ALERT")
        print("="*60)
        print(alert_data['message'])
        print("="*60)

        # Save alert to file
        self.save_alert(alert_data)

    def save_alert(self, alert_data: Dict):
        """Save alert to log file"""
        alert_log_path = Path("./data/live/alerts.json")
        alert_log_path.parent.mkdir(exist_ok=True)

        try:
            # Load existing alerts
            if alert_log_path.exists():
                with open(alert_log_path, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []

            # Add new alert
            alerts.append(alert_data)

            # Keep only last 100 alerts
            alerts = alerts[-100:]

            # Save
            with open(alert_log_path, 'w') as f:
                json.dump(alerts, f, indent=2)

        except Exception as e:
            print(f"❌ Error saving alert: {e}")

    def monitor_live_data(self, live_data_path: str, check_interval: int = 60):
        """Monitor live data and send alerts"""
        print(f"🔍 Starting live monitoring: {live_data_path}")
        print(f"⏰ Check interval: {check_interval}s")

        while self.is_running:
            try:
                # Check if live data file exists and is recent
                if Path(live_data_path).exists():
                    file_time = Path(live_data_path).stat().st_mtime
                    current_time = time.time()

                    # Only analyze if file was updated in last 5 minutes
                    if (current_time - file_time) < 300:
                        # Analyze live market
                        recommendations = self.analyzer.analyze_live_market(live_data_path)

                        # Check each recommendation
                        for rec in recommendations:
                            if self.check_alert_conditions(rec):
                                alert_data = self.create_alert_message(rec)
                                self.send_alert(alert_data)
                            else:
                                print(f"📊 Pattern detected but below alert threshold: {rec.direction} @ {rec.entry_price:.2f}")
                    else:
                        print(f"⏸️ Live data not recent ({(current_time - file_time)/60:.1f} min old)")
                else:
                    print(f"❌ Live data file not found: {live_data_path}")

                # Wait before next check
                time.sleep(check_interval)

            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying

    def start_monitoring(self, live_data_path: str, check_interval: int = 60):
        """Start the monitoring system"""
        if self.is_running:
            print("⚠️ Alert system is already running")
            return

        self.is_running = True
        print("🚀 Starting live alert system...")

        # Start monitoring thread
        self.alert_thread = threading.Thread(
            target=self.monitor_live_data,
            args=(live_data_path, check_interval),
            daemon=True
        )
        self.alert_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        print("⏹️ Live alert system stopped")

    def get_alert_statistics(self) -> Dict:
        """Get statistics about recent alerts"""
        current_hour = datetime.now().hour
        alerts_today = [a for a in self.recent_alerts if a['timestamp'].date() == datetime.now().date()]
        alerts_this_hour = [a for a in alerts_today if a['timestamp'].hour == current_hour]

        buy_alerts = len([a for a in alerts_today if a['direction'] == 'BUY'])
        sell_alerts = len([a for a in alerts_today if a['direction'] == 'SELL'])

        return {
            "alerts_today": len(alerts_today),
            "alerts_this_hour": len(alerts_this_hour),
            "buy_signals_today": buy_alerts,
            "sell_signals_today": sell_alerts,
            "last_alert": self.last_alert_time.isoformat() if self.last_alert_time else None,
            "is_running": self.is_running
        }

def main():
    """Main function for live alert system"""
    import argparse

    parser = argparse.ArgumentParser(description='Live Alert System')
    parser.add_argument('--live-data', required=True, help='Path to live CSV data file')
    parser.add_argument('--rag-url', default='http://localhost:8000', help='RAG system URL')
    parser.add_argument('--webhook', help='Webhook URL for alerts')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--min-confidence', type=float, default=70, help='Minimum confidence for alerts')
    parser.add_argument('--min-risk-reward', type=float, default=1.5, help='Minimum risk/reward ratio')
    parser.add_argument('--sessions', nargs='+', default=['London', 'New York', 'Asia/London Overlap', 'London/NY Overlap'],
                       help='Enabled trading sessions')
    parser.add_argument('--duration', type=int, help='Run for specified duration in minutes')

    args = parser.parse_args()

    # Create alert configuration
    config = AlertConfig(
        min_confidence=args.min_confidence,
        min_risk_reward=args.min_risk_reward,
        enabled_sessions=args.sessions
    )

    # Initialize alert system
    alert_system = LiveAlertSystem(
        rag_base_url=args.rag_url,
        alert_config=config,
        webhook_url=args.webhook
    )

    try:
        # Start monitoring
        alert_system.start_monitoring(args.live_data, args.interval)

        print(f"\n🔔 Alert system is running...")
        print(f"📊 Monitoring: {args.live_data}")
        print(f"⏱️ Check interval: {args.interval}s")
        print(f"🎯 Min confidence: {args.min_confidence}%")
        print(f"💰 Min risk/reward: {args.min_risk_reward}:1")
        print(f"🌍 Enabled sessions: {', '.join(args.sessions)}")

        # Display statistics every 5 minutes
        while alert_system.is_running:
            time.sleep(300)  # 5 minutes
            stats = alert_system.get_alert_statistics()
            print(f"\n📊 Alert Stats - Today: {stats['alerts_today']} (Buy: {stats['buy_signals_today']}, Sell: {stats['sell_signals_today']}), This Hour: {stats['alerts_this_hour']}")

            if args.duration and time.time() - start_time > args.duration * 60:
                break

    except KeyboardInterrupt:
        print("\n⏹️ Stopping alert system...")
        alert_system.stop_monitoring()

    print("✅ Live alert system stopped")

if __name__ == "__main__":
    start_time = time.time()
    main()