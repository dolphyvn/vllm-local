"""
financial_embedding_config.py - Financial Embedding Model Configuration
Optimized embedding models for trading analysis and financial applications
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class FinancialEmbeddingConfig:
    """
    Configuration for financial-specific embedding models
    """

    FINANCIAL_EMBEDDING_MODELS = {
        # Best for general trading analysis
        "finance-news-v1": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2-finance",
            "dimensions": 384,
            "max_sequence_length": 512,
            "speed": "fast",
            "accuracy": "high",
            "use_case": "General trading analysis, news sentiment, market reports",
            "memory_usage": "low"
        },

        # High-performance for detailed analysis
        "finance-roberta-large": {
            "model_name": "sentence-transformers/all-roberta-large-v1-finance",
            "dimensions": 1024,
            "max_sequence_length": 512,
            "speed": "medium",
            "accuracy": "very_high",
            "use_case": "Detailed financial research, complex analysis",
            "memory_usage": "high"
        },

        # Fast for real-time trading
        "finance-e5-small": {
            "model_name": "intfloat/e5-small-v2-finance",
            "dimensions": 336,
            "max_sequence_length": 512,
            "speed": "very_fast",
            "accuracy": "medium",
            "use_case": "Real-time market monitoring, high-frequency queries",
            "memory_usage": "very_low"
        },

        # Specialized for technical analysis
        "finance-technical-v1": {
            "model_name": "sentence-transformers/technical-analysis-v1",
            "dimensions": 768,
            "max_sequence_length": 512,
            "speed": "fast",
            "accuracy": "high",
            "use_case": "Technical indicators, chart patterns, trading strategies",
            "memory_usage": "medium"
        }
    }

    @staticmethod
    def get_recommended_model() -> dict:
        """
        Get the recommended embedding model for trading analysis

        Returns:
            Recommended model configuration
        """
        return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS["finance-news-v1"]

    @staticmethod
    def get_model_by_use_case(use_case: str) -> dict:
        """
        Get model recommendation based on specific use case

        Args:
            use_case: Use case description

        Returns:
            Best model for the use case
        """
        use_case_lower = use_case.lower()

        if any(keyword in use_case_lower for keyword in ["real-time", "fast", "monitoring", "high-frequency"]):
            return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS["finance-e5-small"]
        elif any(keyword in use_case_lower for keyword in ["research", "detailed", "complex", "analysis"]):
            return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS["finance-roberta-large"]
        elif any(keyword in use_case_lower for keyword in ["technical", "indicators", "patterns", "strategies"]):
            return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS["finance-technical-v1"]
        else:
            return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS["finance-news-v1"]

    @staticmethod
    def validate_model_config(model_config: dict) -> bool:
        """
        Validate if a model configuration is suitable for financial analysis

        Args:
            model_config: Model configuration dictionary

        Returns:
            True if suitable for financial analysis
        """
        required_keys = ["model_name", "dimensions", "max_sequence_length"]
        return all(key in model_config for key in required_keys)

    @staticmethod
    def get_config_json(model_key: str = "finance-news-v1") -> dict:
        """
        Get configuration in JSON format for embedding functions

        Args:
            model_key: Key for the model configuration

        Returns:
            Configuration for embedding function
        """
        model = FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS.get(model_key)
        if not model:
            model = FinancialEmbeddingConfig.get_recommended_model()

        return {
            "model_name": model["model_name"],
            "normalize_embeddings": True,
            "batch_size": 32,
            "device": "cpu",  # Can be changed to "cuda" for GPU acceleration
            "trust_remote_code": True
        }

# Example usage for different trading scenarios
TRADING_SCENARIOS = {
    "market_sentiment": "finance-news-v1",
    "technical_analysis": "finance-technical-v1",
    "real_time_monitoring": "finance-e5-small",
    "research_reports": "finance-roberta-large",
    "portfolio_analysis": "finance-news-v1",
    "risk_assessment": "finance-roberta-large"
}

def get_embedding_model_for_scenario(scenario: str) -> dict:
    """
    Get the best embedding model for a specific trading scenario

    Args:
        scenario: Trading scenario (e.g., "market_sentiment", "technical_analysis")

    Returns:
        Model configuration for the scenario
    """
    model_key = TRADING_SCENARIOS.get(scenario.lower(), "finance-news-v1")
    return FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS[model_key]

# Financial text preprocessing functions
def preprocess_financial_text(text: str) -> str:
    """
    Preprocess financial text for better embedding results

    Args:
        text: Raw financial text

    Returns:
        Preprocessed text
    """
    # Normalize common financial symbols and formats
    text = text.replace("$", "USD ")
    text = text.replace("%", " percent")
    text = text.replace("M", " million")
    text = text.replace("B", " billion")

    # Expand common abbreviations
    abbreviations = {
        "P/E": "price-to-earnings ratio",
        "EPS": "earnings per share",
        "ROI": "return on investment",
        "ROE": "return on equity",
        "P/E ratio": "price-to-earnings ratio"
    }

    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    return text.strip()

if __name__ == "__main__":
    # Demonstrate model recommendations
    print("🎯 Financial Embedding Model Recommendations")
    print("=" * 50)

    for scenario, model_key in TRADING_SCENARIOS.items():
        model = FinancialEmbeddingConfig.FINANCIAL_EMBEDDING_MODELS[model_key]
        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  Model: {model['model_name']}")
        print(f"  Dimensions: {model['dimensions']}")
        print(f"  Speed: {model['speed']}")
        print(f"  Use case: {model['use_case']}")

    print(f"\n📊 Recommended model for general trading:")
    recommended = FinancialEmbeddingConfig.get_recommended_model()
    print(f"  Model: {recommended['model_name']}")
    print(f"  Dimensions: {recommended['dimensions']}")
    print(f"  Speed: {recommended['speed']}")