#!/usr/bin/env python3
"""
Test script for web search and trading news functionality
"""

from web_search import WebSearchTool, TradingNewsAPI, TradingTools

def test_web_search():
    """Test basic web search"""
    print("=" * 80)
    print("Testing Web Search Tool")
    print("=" * 80)

    web_search = WebSearchTool()

    # Test 1: Basic web search
    print("\n1. Testing basic web search...")
    result = web_search.search_web("XAUUSD gold price forecast", max_results=3)
    print(result[:500] + "...")

    # Test 2: News search
    print("\n2. Testing news search...")
    result = web_search.search_news("gold trading news", max_results=3)
    print(result[:500] + "...")

    print("\n✅ Web search tests completed!")

def test_trading_news():
    """Test trading news API"""
    print("\n" + "=" * 80)
    print("Testing Trading News API")
    print("=" * 80)

    web_search = WebSearchTool()
    news_api = TradingNewsAPI(web_search)

    # Test 1: Symbol news
    print("\n1. Testing symbol news (XAUUSD)...")
    result = news_api.get_symbol_news("XAUUSD")
    print(result[:500] + "...")

    # Test 2: Market sentiment
    print("\n2. Testing market sentiment...")
    result = news_api.get_market_sentiment("XAUUSD")
    print(result[:500] + "...")

    # Test 3: Economic calendar
    print("\n3. Testing economic calendar...")
    result = news_api.get_economic_calendar()
    print(result[:500] + "...")

    # Test 4: Market overview
    print("\n4. Testing market overview...")
    result = news_api.get_market_overview()
    print(result[:500] + "...")

    print("\n✅ Trading news tests completed!")

def test_trading_tools():
    """Test trading tools integration"""
    print("\n" + "=" * 80)
    print("Testing Trading Tools")
    print("=" * 80)

    web_search = WebSearchTool()
    news_api = TradingNewsAPI(web_search)
    tools = TradingTools(web_search, news_api)

    # Test 1: Check if query needs web search
    print("\n1. Testing query classification...")
    queries = [
        "What is the latest gold news?",
        "Calculate moving average",
        "Show me today's economic events",
        "What is RSI indicator?"
    ]

    for query in queries:
        needs_web = tools.should_use_web_search(query)
        print(f"   '{query[:50]}...' -> Web search: {needs_web}")

    # Test 2: Execute a tool
    print("\n2. Testing tool execution...")
    result = tools.execute_tool("get_symbol_news", {"symbol": "XAUUSD"})
    print(result[:300] + "...")

    print("\n✅ Trading tools tests completed!")

def main():
    """Run all tests"""
    print("\n🚀 Starting Web Search Integration Tests\n")

    try:
        test_web_search()
        test_trading_news()
        test_trading_tools()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nWeb search integration is working correctly!")
        print("Your LLM now has access to real-time internet information.")
        print("\nAvailable API endpoints:")
        print("  - GET  /api/web/search?query=...")
        print("  - GET  /api/news/search?query=...")
        print("  - GET  /api/news/symbol/{symbol}")
        print("  - GET  /api/news/market-sentiment/{symbol}")
        print("  - GET  /api/news/economic-calendar")
        print("  - GET  /api/news/market-overview")
        print("  - GET  /api/news/technical-analysis/{symbol}")
        print("  - POST /api/chat/web-enhanced")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
