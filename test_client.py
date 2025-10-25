"""
test_client.py - Test client for the local financial assistant API
Demonstrates OpenAI-style API calls and tests all endpoints
"""

import requests
import json
import time
from typing import Dict, Any

# API configuration
BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/v1"  # OpenAI-compatible endpoint style

class FinancialAssistantClient:
    """Client for testing the financial assistant API"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def check_models(self):
        """Check available models"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Models check failed: {e}")
            return {"error": str(e)}

    def chat(self, message: str, model: str = "llama3.2") -> Dict[str, Any]:
        """
        Send a chat message to the assistant

        Args:
            message: User message
            model: Model name to use

        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/chat"

        payload = {
            "message": message,
            "model": model,
            "memory_context": 3
        }

        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Chat request failed: {e}")
            return {"error": str(e)}

    def memorize(self, key: str, value: str, category: str = "general") -> Dict[str, Any]:
        """
        Store information in memory

        Args:
            key: Memory key
            value: Memory value
            category: Memory category

        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/memorize"

        payload = {
            "key": key,
            "value": value,
            "category": category
        }

        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Memorize request failed: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check system health

        Returns:
            Health status dictionary
        """
        url = f"{self.base_url}/health"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return {"error": str(e)}

    def openai_style_chat(self, messages: list, model: str = "llama3.2") -> Dict[str, Any]:
        """
        OpenAI-style chat completion call (for compatibility testing)

        Args:
            messages: List of message dicts
            model: Model name

        Returns:
            Response in OpenAI format
        """
        # Convert to our chat format
        if messages:
            user_message = messages[-1].get("content", "")
        else:
            user_message = "Hello"

        response = self.chat(user_message, model)

        # Convert to OpenAI-style response
        if "error" not in response:
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("response", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # vLLM doesn't provide token count by default
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        else:
            return {"error": response["error"]}

    def chat_stream(self, message: str, model: str = "llama3.2"):
        """
        Streaming chat completion that yields tokens as they arrive

        Args:
            message: User message
            model: Model name

        Yields:
            Token chunks as they arrive
        """
        url = f"{self.base_url}/chat/stream"
        payload = {
            "message": message,
            "model": model,
            "memory_context": 3
        }

        try:
            response = self.session.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            print(f"Streaming request failed: {e}")
            yield f"ERROR: Streaming failed - {str(e)}"

def run_tests():
    """Run comprehensive tests of the financial assistant API"""
    client = FinancialAssistantClient()

    print("🧪 Testing Local Financial Assistant API")
    print("=" * 50)

    # Test 1: Health Check
    print("\n1️⃣ Health Check Test")
    print("-" * 30)
    health = client.health_check()
    if "error" not in health:
        print(f"✅ Health check passed: {health}")
    else:
        print(f"❌ Health check failed: {health}")
        return

    # Test 1.5: Models Check
    print("\n1️⃣5️⃣ Models Check Test")
    print("-" * 30)
    models = client.check_models()
    if "error" not in models:
        print(f"✅ Models check passed:")
        print(f"   Current model: {models.get('current_model', 'Unknown')}")
        print(f"   Model available: {models.get('model_available', 'Unknown')}")
        print(f"   Available models: {models.get('available_models', [])}")
        print(f"   Total models: {models.get('total_models', 0)}")
    else:
        print(f"❌ Models check failed: {models}")

    # Test 2: Memory Storage
    print("\n2️⃣ Memory Storage Test")
    print("-" * 30)

    # Store some trading rules
    memory_tests = [
        ("risk_management", "Never risk more than 2% per trade", "trading"),
        ("preferred_pairs", "Focus on XAUUSD and EURUSD during London session", "trading"),
        ("strategy", "Wait for confirmation from multiple timeframes before entry", "strategy")
    ]

    for key, value, category in memory_tests:
        result = client.memorize(key, value, category)
        if "error" not in result:
            print(f"✅ Stored memory: {key}")
        else:
            print(f"❌ Failed to store memory: {result}")

    time.sleep(1)  # Brief pause to ensure storage

    # Test 3: Basic Chat
    print("\n3️⃣ Basic Chat Test")
    print("-" * 30)

    chat_tests = [
        "What's the current market sentiment for gold?",
        "Analyze XAUUSD order flow for potential entry points",
        "What are the key risk management principles?",
        "Explain the impact of Fed interest rate decisions on forex markets",
        "What technical indicators work best for gold trading?"
    ]

    for i, message in enumerate(chat_tests, 1):
        print(f"\nTest {i}: {message}")
        response = client.chat(message)

        if "error" not in response:
            print(f"✅ Response received:")
            print(f"   Model: {response.get('model', 'Unknown')}")
            print(f"   Memory used: {response.get('memory_used', False)}")
            print(f"   Response: {response.get('response', 'No response')[:200]}...")
        else:
            print(f"❌ Chat failed: {response}")

        time.sleep(2)  # Pause between requests

    # Test 4: Context Memory Test
    print("\n4️⃣ Context Memory Test")
    print("-" * 30)

    # Follow-up question to test memory retrieval
    follow_up = "Based on our previous discussion, what risk management rules should I follow?"
    response = client.chat(follow_up)

    if "error" not in response:
        print(f"✅ Follow-up response received:")
        print(f"   Memory used: {response.get('memory_used', False)}")
        print(f"   Response: {response.get('response', 'No response')[:200]}...")
    else:
        print(f"❌ Follow-up failed: {response}")

    # Test 5: OpenAI Compatibility Test
    print("\n5️⃣ OpenAI Compatibility Test")
    print("-" * 30)

    openai_messages = [
        {"role": "user", "content": "Provide a quick technical analysis of EUR/USD"}
    ]

    openai_response = client.openai_style_chat(openai_messages)

    if "error" not in openai_response:
        print("✅ OpenAI-style response received:")
        print(f"   ID: {openai_response.get('id')}")
        print(f"   Model: {openai_response.get('model')}")
        print(f"   Content: {openai_response['choices'][0]['message']['content'][:200]}...")
    else:
        print(f"❌ OpenAI compatibility test failed: {openai_response}")

    # Test 6: Streaming Chat Test
    print("\n6️⃣ Streaming Chat Test")
    print("-" * 30)

    streaming_message = "Explain the concept of technical analysis in trading"
    print(f"Testing streaming with: {streaming_message}")
    print("Response: ", end="", flush=True)

    try:
        response_parts = []
        for token in client.chat_stream(streaming_message):
            print(token, end="", flush=True)
            response_parts.append(token)
            time.sleep(0.01)  # Small delay to visualize streaming

        print("\n✅ Streaming test completed successfully!")
        print(f"   Total response length: {len(''.join(response_parts))} characters")

    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")

    print("\n" + "=" * 50)
    print("🎉 All tests completed!")

def interactive_mode():
    """Interactive mode for manual testing"""
    client = FinancialAssistantClient()

    print("\n🤖 Interactive Financial Assistant")
    print("Type 'quit' to exit, 'memorize' to store memory, 'health' for health check, 'stream' to toggle streaming, 'models' to list models")
    print("-" * 60)

    streaming_mode = False

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break

            if user_input.lower() == 'health':
                health = client.health_check()
                print(f"Health: {json.dumps(health, indent=2)}")
                continue

            if user_input.lower() == 'memorize':
                key = input("Key: ").strip()
                value = input("Value: ").strip()
                category = input("Category (default: general): ").strip() or "general"

                result = client.memorize(key, value, category)
                print(f"Memory: {json.dumps(result, indent=2)}")
                continue

            if user_input.lower() == 'stream':
                streaming_mode = not streaming_mode
                print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue

            if user_input.lower() == 'models':
                models_info = client.check_models()
                if "error" not in models_info:
                    print(f"Available Models: {models_info}")
                else:
                    print(f"Models check failed: {models_info}")
                continue

            if user_input:
                print("Assistant: ", end="", flush=True)

                if streaming_mode:
                    # Use streaming
                    try:
                        for token in client.chat_stream(user_input):
                            print(token, end="", flush=True)
                        print()  # New line after completion
                    except Exception as e:
                        print(f"\nError: {e}")
                else:
                    # Use regular chat
                    response = client.chat(user_input)

                    if "error" not in response:
                        print(response.get("response", "No response"))
                        if response.get("memory_used"):
                            print("📝 (Used memory context)")
                    else:
                        print(f"Error: {response['error']}")

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        run_tests()