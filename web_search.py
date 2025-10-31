"""
web_search.py - Web search and content fetching capabilities for the trading system
Provides internet access to LLM for real-time information
"""

import logging
import requests
from typing import List, Dict, Optional
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool using DuckDuckGo (free, no API key required)
    Provides internet access capabilities to the local LLM
    """

    def __init__(self):
        self.ddg = DDGS()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search_web(self, query: str, max_results: int = 5) -> str:
        """
        Search the web and return formatted results

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            Formatted string with search results
        """
        try:
            logger.info(f"Web search: {query}")
            results = self.ddg.text(query, max_results=max_results)

            if not results:
                return "No results found."

            formatted = f"🔍 Web Search Results for: '{query}'\n"
            formatted += f"Found {len(results)} results\n\n"

            for i, result in enumerate(results, 1):
                formatted += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                formatted += f"Result {i}:\n"
                formatted += f"📌 Title: {result.get('title', 'N/A')}\n"
                formatted += f"📝 Summary: {result.get('body', 'N/A')}\n"
                formatted += f"🔗 Source: {result.get('href', 'N/A')}\n\n"

            return formatted

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Search error: {str(e)}"

    def search_news(self, query: str, max_results: int = 5) -> str:
        """
        Search for recent news articles

        Args:
            query: Search query string
            max_results: Maximum number of news articles

        Returns:
            Formatted string with news results
        """
        try:
            logger.info(f"News search: {query}")
            results = self.ddg.news(query, max_results=max_results)

            if not results:
                return "No news found."

            formatted = f"📰 Latest News for: '{query}'\n"
            formatted += f"Found {len(results)} articles\n\n"

            for i, article in enumerate(results, 1):
                formatted += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                formatted += f"Article {i}:\n"
                formatted += f"📌 {article.get('title', 'N/A')}\n"
                formatted += f"📅 Published: {article.get('date', 'N/A')}\n"
                formatted += f"📰 Source: {article.get('source', 'N/A')}\n"
                formatted += f"📝 {article.get('body', 'N/A')}\n"
                formatted += f"🔗 {article.get('url', 'N/A')}\n\n"

            return formatted

        except Exception as e:
            logger.error(f"News search error: {e}")
            return f"News search error: {str(e)}"

    def get_url_content(self, url: str, max_length: int = 5000) -> str:
        """
        Fetch and extract text content from a URL

        Args:
            url: URL to fetch
            max_length: Maximum content length to return

        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + f"\n\n[Content truncated, {len(text)} total characters]"

            return f"📄 Content from {url}:\n\n{text}"

        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return f"Error parsing content: {str(e)}"


class TradingNewsAPI:
    """
    Specialized trading news fetcher for market-specific information
    """

    def __init__(self, web_search: WebSearchTool):
        self.web_search = web_search

    def get_symbol_news(self, symbol: str = "XAUUSD", max_results: int = 5) -> str:
        """
        Get latest news for a specific trading symbol

        Args:
            symbol: Trading symbol (e.g., XAUUSD, EURUSD, BTCUSD)
            max_results: Number of news articles to fetch

        Returns:
            Formatted news summary
        """
        try:
            # Map common symbols to search-friendly names
            symbol_map = {
                "XAUUSD": "gold XAU/USD",
                "XAGUSD": "silver XAG/USD",
                "EURUSD": "EUR/USD euro dollar",
                "GBPUSD": "GBP/USD pound dollar",
                "USDJPY": "USD/JPY dollar yen",
                "BTCUSD": "Bitcoin BTC/USD",
                "ETHUSD": "Ethereum ETH/USD"
            }

            search_term = symbol_map.get(symbol.upper(), symbol)
            today = datetime.now().strftime('%Y-%m-%d')

            query = f"{search_term} trading news forecast {today}"
            return self.web_search.search_news(query, max_results)

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return f"Error fetching news: {str(e)}"

    def get_market_sentiment(self, symbol: str = "XAUUSD") -> str:
        """
        Get current market sentiment for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Market sentiment summary
        """
        query = f"{symbol} market sentiment analysis today"
        return self.web_search.search_web(query, max_results=3)

    def get_economic_calendar(self) -> str:
        """
        Get today's important economic events

        Returns:
            Economic calendar information
        """
        today = datetime.now().strftime('%Y-%m-%d')
        query = f"forex economic calendar events {today} high impact"
        return self.web_search.search_web(query, max_results=5)

    def get_technical_analysis(self, symbol: str = "XAUUSD") -> str:
        """
        Get latest technical analysis for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Technical analysis summary
        """
        today = datetime.now().strftime('%Y-%m-%d')
        query = f"{symbol} technical analysis forecast {today}"
        return self.web_search.search_news(query, max_results=3)

    def get_market_overview(self) -> str:
        """
        Get general market overview and major events

        Returns:
            Market overview summary
        """
        today = datetime.now().strftime('%Y-%m-%d')
        query = f"forex market overview trading {today}"
        return self.web_search.search_news(query, max_results=5)


class TradingTools:
    """
    Complete toolset for LLM with web access and trading-specific functions
    """

    def __init__(self, web_search: WebSearchTool, news_api: TradingNewsAPI):
        self.web_search = web_search
        self.news_api = news_api

    def get_tools_description(self) -> str:
        """
        Provide tool descriptions to the LLM

        Returns:
            Formatted tool descriptions
        """
        return """
🛠️ Available Internet Tools:

1. search_web(query)
   - Search the internet for any information
   - Example: search_web("latest gold price forecast")

2. search_news(query)
   - Search for recent news articles
   - Example: search_news("XAUUSD trading news")

3. get_symbol_news(symbol)
   - Get latest news for a specific trading symbol
   - Example: get_symbol_news("XAUUSD")

4. get_market_sentiment(symbol)
   - Get current market sentiment analysis
   - Example: get_market_sentiment("EURUSD")

5. get_economic_calendar()
   - Get today's important economic events
   - Example: get_economic_calendar()

6. get_technical_analysis(symbol)
   - Get latest technical analysis and forecasts
   - Example: get_technical_analysis("XAUUSD")

7. get_market_overview()
   - Get general market overview
   - Example: get_market_overview()

8. get_url_content(url)
   - Fetch and read content from a specific URL
   - Example: get_url_content("https://example.com/article")

To use a tool, respond with:
TOOL: tool_name
ARGS: {"arg1": "value1"}
"""

    def execute_tool(self, tool_name: str, args: Dict) -> str:
        """
        Execute a tool and return results

        Args:
            tool_name: Name of the tool to execute
            args: Dictionary of arguments

        Returns:
            Tool execution results
        """
        try:
            logger.info(f"Executing tool: {tool_name} with args: {args}")

            if tool_name == "search_web":
                query = args.get("query", "")
                max_results = args.get("max_results", 5)
                return self.web_search.search_web(query, max_results)

            elif tool_name == "search_news":
                query = args.get("query", "")
                max_results = args.get("max_results", 5)
                return self.web_search.search_news(query, max_results)

            elif tool_name == "get_symbol_news":
                symbol = args.get("symbol", "XAUUSD")
                return self.news_api.get_symbol_news(symbol)

            elif tool_name == "get_market_sentiment":
                symbol = args.get("symbol", "XAUUSD")
                return self.news_api.get_market_sentiment(symbol)

            elif tool_name == "get_economic_calendar":
                return self.news_api.get_economic_calendar()

            elif tool_name == "get_technical_analysis":
                symbol = args.get("symbol", "XAUUSD")
                return self.news_api.get_technical_analysis(symbol)

            elif tool_name == "get_market_overview":
                return self.news_api.get_market_overview()

            elif tool_name == "get_url_content":
                url = args.get("url", "")
                return self.web_search.get_url_content(url)

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Tool execution error: {str(e)}"

    def should_use_web_search(self, query: str) -> bool:
        """
        Determine if a query should trigger web search

        Args:
            query: User query

        Returns:
            True if web search should be used
        """
        web_keywords = [
            "news", "today", "current", "latest", "now", "recent",
            "what's happening", "update", "forecast", "sentiment",
            "calendar", "event", "announcement", "market overview"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_keywords)
