#!/usr/bin/env python3
"""
install_financial_embeddings.py - Installation Script for Financial Embedding Models
Installs and configures financial-specific embedding models for trading analysis
"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package_name}: {e}")
        return False

def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        logger.info(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        logger.info(f"📦 {package_name} not found, will install")
        return False

def download_financial_models():
    """Download or setup financial embedding models"""
    try:
        from sentence_transformers import SentenceTransformer
        from financial_embedding_config import FinancialEmbeddingConfig

        logger.info("🔄 Testing financial embedding models...")

        # Test recommended model
        recommended_config = FinancialEmbeddingConfig.get_recommended_model()
        model_name = recommended_config["model_name"]

        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Dimensions: {recommended_config['dimensions']}")
        logger.info(f"   Speed: {recommended_config['speed']}")
        logger.info(f"   Use case: {recommended_config['use_case']}")

        # Test embedding functionality
        test_text = "AAPL stock shows bullish momentum with RSI at 65"
        embedding = model.encode(test_text)
        logger.info(f"✅ Embedding test successful: {len(embedding)} dimensions")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to setup financial models: {e}")
        return False

def test_financial_memory():
    """Test the financial memory manager"""
    try:
        from financial_memory_manager import FinancialMemoryManager

        logger.info("🧪 Testing financial memory manager...")
        fm_manager = FinancialMemoryManager()

        # Test adding a trading memory
        fm_manager.add_trading_memory(
            user_query="What's the market outlook for tech stocks?",
            ai_response="Technical indicators suggest bullish momentum for major tech stocks.",
            market_context={"symbols": ["AAPL", "GOOGL"], "indicators": ["RSI", "MACD"]},
            trading_signals=["bullish"],
            confidence_score=0.8
        )

        # Test searching
        results = fm_manager.search_trading_memories("tech stocks")
        logger.info(f"✅ Memory test successful: Found {len(results)} memories")

        return True

    except Exception as e:
        logger.error(f"❌ Financial memory test failed: {e}")
        return False

def create_requirements_file():
    """Create a requirements file for financial embeddings"""
    requirements = """
# Financial Embedding Requirements
sentence-transformers>=2.2.2
torch>=1.9.0
numpy>=1.21.0
chromadb>=0.4.0
"""

    try:
        with open("requirements_financial.txt", "w") as f:
            f.write(requirements.strip())
        logger.info("✅ Created requirements_financial.txt")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create requirements file: {e}")
        return False

def main():
    """Main installation process"""
    logger.info("🚀 Financial Embedding Installation for Trading Analysis")
    logger.info("=" * 60)

    # Check system requirements
    logger.info("📋 Checking system requirements...")

    required_packages = [
        "torch",
        "sentence-transformers",
        "numpy",
        "chromadb"
    ]

    failed_packages = []

    for package in required_packages:
        if not check_package_installed(package):
            if not install_package(package):
                failed_packages.append(package)

    if failed_packages:
        logger.error(f"❌ Installation failed for packages: {failed_packages}")
        logger.error("Please install them manually and run the script again")
        return False

    # Create requirements file
    create_requirements_file()

    # Test financial models
    if not download_financial_models():
        logger.error("❌ Financial model setup failed")
        return False

    # Test financial memory
    if not test_financial_memory():
        logger.error("❌ Financial memory test failed")
        return False

    logger.info("🎉 Installation completed successfully!")
    logger.info("=" * 60)
    logger.info("📋 Next Steps:")
    logger.info("1. Update your config.json to enable financial embeddings")
    logger.info("2. Replace MemoryManager with FinancialMemoryManager in main.py")
    logger.info("3. Restart your application")
    logger.info("4. Test with trading queries to see enhanced context retrieval")

    logger.info("\n💡 Example configuration:")
    logger.info('   "financial_embedding": {"enabled": true, "model_type": "finance-news-v1"}')

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)