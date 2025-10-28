#!/usr/bin/env python3
"""
Scheduler for RAG MT5 Knowledge Feeding
Automated scheduling system for continuous learning
"""

import schedule
import time
import logging
from datetime import datetime
from integrate_rag_mt5_data import RAGMT5Integrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGFeedingScheduler:
    def __init__(self, base_url: str = "http://localhost:8080", password: str = "admin123",
                 export_path: str = "./data", log_file: str = "./processed_rag_files.log"):
        self.base_url = base_url
        self.password = password
        self.export_path = export_path
        self.log_file = log_file
        self.integrator = None

    def initialize(self):
        """Initialize the integrator"""
        try:
            self.integrator = RAGMT5Integrator(
                base_url=self.base_url,
                password=self.password,
                export_path=self.export_path,
                log_file=self.log_file
            )
            logger.info("✅ RAG Feeding Scheduler initialized")
            logger.info(f"   Export path: {self.export_path}")
            logger.info(f"   API Server: {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False

    def process_rag_data(self):
        """Process RAG data - scheduled job"""
        logger.info(f"🕐 {datetime.now()}: Starting scheduled RAG data processing...")

        try:
            results = self.integrator.process_all_files()

            if results["processed"] > 0:
                logger.info(f"✅ Scheduled processing successful: {results}")
            else:
                logger.info("ℹ️ No new RAG data to process in scheduled run")

        except Exception as e:
            logger.error(f"❌ Scheduled processing failed: {e}")

    def setup_schedule(self):
        """Set up the processing schedule"""
        # Process every 30 minutes during trading hours
        schedule.every(30).minutes.do(self.process_rag_data)

        # Process at specific times during major trading sessions
        schedule.every().day.at("02:30").do(self.process_rag_data)  # Asia open
        schedule.every().day.at("08:00").do(self.process_rag_data)  # Europe open
        schedule.every().day.at("13:30").do(self.process_rag_data)  # US open
        schedule.every().day.at("20:00").do(self.process_rag_data)  # Asia close

        # Weekly cleanup and analysis
        schedule.every().sunday.at("22:00").do(self.weekly_analysis)

        logger.info("📅 Schedule configured:")
        logger.info("  - Every 30 minutes: Process new RAG data")
        logger.info("  - 02:30, 08:00, 13:30, 20:00: Major session processing")
        logger.info("  - Sunday 22:00: Weekly analysis")

    def weekly_analysis(self):
        """Perform weekly analysis of knowledge base"""
        logger.info("📊 Starting weekly analysis...")

        try:
            # Get knowledge statistics
            response = self.integrator.session.get(
                f"{self.base_url}/api/knowledge/stats"
            )

            if response.status_code == 200:
                stats = response.json()
                logger.info(f"📈 Knowledge base stats: {stats}")

                # Log summary
                total_entries = stats.get('total_entries', 0)
                trading_entries = stats.get('by_category', {}).get('trading_analysis', 0)
                lessons = stats.get('lessons_count', 0)

                logger.info(f"📊 Weekly Summary:")
                logger.info(f"  - Total knowledge entries: {total_entries}")
                logger.info(f"  - Trading analysis entries: {trading_entries}")
                logger.info(f"  - Lessons created: {lessons}")
                logger.info(f"  - Knowledge base growth: {total_entries - stats.get('last_week_entries', 0)} new entries")

            else:
                logger.error(f"❌ Failed to get stats: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Weekly analysis failed: {e}")

    def run(self):
        """Run the scheduler"""
        logger.info("🚀 RAG Feeding Scheduler starting...")

        if not self.initialize():
            return

        self.setup_schedule()

        # Process immediately on start
        logger.info("⚡ Processing any pending RAG data on startup...")
        self.process_rag_data()

        # Run scheduled jobs
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("⏹️ Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def validate_directory(path: str) -> str:
    """Validate and normalize directory path"""
    if not os.path.exists(path):
        logger.warning(f"Directory does not exist: {path}")
        logger.info(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)

    if not os.path.isdir(path):
        logger.error(f"❌ Path is not a directory: {path}")
        sys.exit(1)

    return os.path.abspath(path)

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG MT5 Feeding Scheduler - Automated knowledge feeding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python3 schedule_rag_feeding.py

  # Monitor custom directory
  python3 schedule_rag_feeding.py --export-path /path/to/trading/exports

  # Use remote server with custom directory
  python3 schedule_rag_feeding.py --base-url http://ai.vn.aliases.me --export-path ./trading_data
        """
    )

    parser.add_argument("--base-url", default="http://localhost:8080",
                       help="Base URL for Financial Assistant API (default: http://localhost:8080)")
    parser.add_argument("--export-path", default="./data",
                       help="Path to MT5 RAG export files (default: ./data)")
    parser.add_argument("--log-file", default="./processed_rag_files.log",
                       help="Log file for tracking processed files (default: ./processed_rag_files.log)")
    parser.add_argument("--password", default="admin123",
                       help="API password (default: admin123)")

    args = parser.parse_args()

    # Validate export directory
    export_path = validate_directory(args.export_path)

    # Create scheduler
    scheduler = RAGFeedingScheduler(
        base_url=args.base_url,
        password=args.password,
        export_path=export_path,
        log_file=args.log_file
    )

    logger.info(f"🚀 Starting RAG Feeding Scheduler")
    logger.info(f"   Export path: {export_path}")
    logger.info(f"   API Server: {args.base_url}")
    logger.info(f"   Log file: {args.log_file}")

    scheduler.run()

if __name__ == "__main__":
    main()