#!/usr/bin/env python3
"""
test_self_improving.py - Comprehensive test suite for self-improving lesson system
Tests lesson storage, retrieval, feedback, and integration with chat
"""

import requests
import json
import time
import uuid
from datetime import datetime

class SelfImprovingTester:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.test_results = []

    def log_test(self, test_name, success, message=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        print(f"{status} {test_name}: {message}")

    def test_health_check(self):
        """Test system health including lessons"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get("services", {})
                success = (
                    services.get("ollama") == "ok" and
                    services.get("chroma") == "ok" and
                    services.get("sqlite") == "ok"
                )
                self.log_test(
                    "Health Check",
                    success,
                    f"Services: Ollama={services.get('ollama')}, Chroma={services.get('chroma')}, SQLite={services.get('sqlite')}"
                )
                return success
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {e}")
            return False

    def test_add_lesson(self):
        """Test adding a lesson"""
        try:
            lesson_data = {
                "title": "Test Trading Strategy",
                "content": "Always use stop-loss orders to limit downside risk. Set stop-loss at 2% below entry price for gold trading.",
                "category": "risk_management",
                "confidence": 0.9,
                "tags": ["risk", "gold", "stop-loss"]
            }

            response = requests.post(f"{self.base_url}/lessons", json=lesson_data, timeout=10)
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                lesson_id = data.get("lesson_id")
                self.log_test(
                    "Add Lesson",
                    success,
                    f"Lesson ID: {lesson_id}"
                )
                return lesson_id if success else None
            else:
                self.log_test("Add Lesson", False, f"HTTP {response.status_code}")
                return None
        except Exception as e:
            self.log_test("Add Lesson", False, f"Exception: {e}")
            return None

    def test_get_lessons(self):
        """Test retrieving lessons"""
        try:
            response = requests.get(f"{self.base_url}/lessons?query=risk&limit=5", timeout=10)
            if response.status_code == 200:
                data = response.json()
                lessons = data.get("lessons", [])
                success = len(lessons) > 0
                self.log_test(
                    "Get Lessons",
                    success,
                    f"Found {len(lessons)} lessons"
                )
                return lessons
            else:
                self.log_test("Get Lessons", False, f"HTTP {response.status_code}")
                return []
        except Exception as e:
            self.log_test("Get Lessons", False, f"Exception: {e}")
            return []

    def test_lesson_feedback(self, lesson_id):
        """Test adding lesson feedback"""
        if not lesson_id:
            self.log_test("Lesson Feedback", False, "No lesson ID available")
            return False

        try:
            feedback_data = {
                "rating": 5,
                "feedback_text": "Very helpful advice!",
                "helpful": True,
                "user_context": {"query_type": "risk_management"}
            }

            response = requests.post(
                f"{self.base_url}/lessons/{lesson_id}/feedback",
                json=feedback_data,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                self.log_test(
                    "Lesson Feedback",
                    success,
                    f"Feedback ID: {data.get('feedback_id')}"
                )
                return success
            else:
                self.log_test("Lesson Feedback", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Lesson Feedback", False, f"Exception: {e}")
            return False

    def test_correction(self):
        """Test adding a correction"""
        try:
            correction_data = {
                "original_response": "Buy gold now without any risk management.",
                "corrected_response": "Consider buying gold but set a stop-loss at 2% below entry to manage risk.",
                "correction_reason": "Risk management is crucial in trading",
                "conversation_id": str(uuid.uuid4())
            }

            response = requests.post(f"{self.base_url}/corrections", json=correction_data, timeout=10)
            if response.status_code == 200:
                data = response.json()
                success = data.get("success", False)
                self.log_test(
                    "Add Correction",
                    success,
                    f"Correction ID: {data.get('correction_id')}, Lesson ID: {data.get('lesson_id')}"
                )
                return success
            else:
                self.log_test("Add Correction", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Add Correction", False, f"Exception: {e}")
            return False

    def test_lesson_stats(self):
        """Test lesson statistics"""
        try:
            response = requests.get(f"{self.base_url}/lessons/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                success = data.get("total_lessons", 0) >= 0
                self.log_test(
                    "Lesson Statistics",
                    success,
                    f"Total lessons: {data.get('total_lessons')}, Categories: {list(data.get('lessons_by_category', {}).keys())}"
                )
                return success
            else:
                self.log_test("Lesson Statistics", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Lesson Statistics", False, f"Exception: {e}")
            return False

    def test_chat_with_lessons(self):
        """Test chat functionality with lesson integration"""
        try:
            chat_data = {
                "message": "What should I consider when trading gold?",
                "model": "phi3:latest",
                "memory_context": 2,
                "stream": False
            }

            response = requests.post(f"{self.base_url}/chat", json=chat_data, timeout=30)
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                success = len(response_text) > 10
                self.log_test(
                    "Chat with Lessons",
                    success,
                    f"Response length: {len(response_text)} characters"
                )
                return success
            else:
                self.log_test("Chat with Lessons", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Chat with Lessons", False, f"Exception: {e}")
            return False

    def test_streaming_chat(self):
        """Test streaming chat with lesson integration"""
        try:
            chat_data = {
                "message": "Explain position sizing in forex trading",
                "model": "phi3:latest",
                "memory_context": 2
            }

            response = requests.post(f"{self.base_url}/chat/stream", json=chat_data, stream=True, timeout=30)
            if response.status_code == 200:
                content = ""
                for line in response.iter_lines():
                    if line:
                        content += line.decode('utf-8')
                success = len(content) > 50
                self.log_test(
                    "Streaming Chat with Lessons",
                    success,
                    f"Streamed response length: {len(content)} characters"
                )
                return success
            else:
                self.log_test("Streaming Chat with Lessons", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Streaming Chat with Lessons", False, f"Exception: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and return summary"""
        print("🧠 Testing Self-Improving Lesson System")
        print("=" * 50)

        # Run tests in logical order
        lesson_id = None

        self.test_health_check()
        time.sleep(1)

        lesson_id = self.test_add_lesson()
        time.sleep(1)

        lessons = self.test_get_lessons()
        time.sleep(1)

        if lesson_id:
            self.test_lesson_feedback(lesson_id)
            time.sleep(1)

        self.test_correction()
        time.sleep(1)

        self.test_lesson_stats()
        time.sleep(1)

        self.test_chat_with_lessons()
        time.sleep(1)

        self.test_streaming_chat()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)

        passed = sum(1 for result in self.test_results if "PASS" in result["status"])
        total = len(self.test_results)

        for result in self.test_results:
            print(f"{result['status']} {result['test']}: {result['message']}")

        print(f"\nResults: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 ALL TESTS PASSED! Self-improving system is working correctly.")
        else:
            print(f"⚠️  {total - passed} tests failed. Check the logs above.")

        return passed == total

if __name__ == "__main__":
    # Run tests
    tester = SelfImprovingTester()
    tester.run_all_tests()