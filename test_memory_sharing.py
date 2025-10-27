#!/usr/bin/env python3
"""
Test script to verify memory sharing across different sessions
"""

import requests
import json
import time

def test_memory_sharing():
    """Test if memory is shared across different sessions"""

    base_url = "http://localhost:8080"  # Change to your server URL

    # Session 1: Browser A
    print("🧪 Session 1: Browser A - Setting user name")
    session1 = requests.Session()

    # Login
    login_response = session1.post(f"{base_url}/auth/login", json={"password": "admin123"})
    if login_response.status_code != 200:
        print("❌ Session 1 login failed")
        return False

    print("✅ Session 1 authenticated")

    # Set user name
    chat_response = session1.post(f"{base_url}/chat", json={
        "message": "My name is Dolphy. Please remember this for future conversations.",
        "memory_context": 3
    })

    if chat_response.status_code != 200:
        print("❌ Session 1 chat failed")
        return False

    print("✅ Session 1: Set user name")

    # Test within same session
    test_response = session1.post(f"{base_url}/chat", json={
        "message": "What is my name?",
        "memory_context": 3
    })

    if test_response.status_code == 200:
        result = test_response.json()
        print(f"✅ Session 1 response: {result['response'][:100]}...")
        if "dolphy" in result['response'].lower():
            print("✅ Session 1: Memory works within same session")
        else:
            print("⚠️  Session 1: Memory might not be working properly")

    time.sleep(2)  # Small delay

    # Session 2: Browser B (different session)
    print("\n🧪 Session 2: Browser B - Testing memory sharing")
    session2 = requests.Session()

    # Login
    login_response = session2.post(f"{base_url}/auth/login", json={"password": "admin123"})
    if login_response.status_code != 200:
        print("❌ Session 2 login failed")
        return False

    print("✅ Session 2 authenticated")

    # Test if memory is shared from Session 1
    test_response = session2.post(f"{base_url}/chat", json={
        "message": "What is my name?",
        "memory_context": 3
    })

    if test_response.status_code == 200:
        result = test_response.json()
        print(f"✅ Session 2 response: {result['response'][:100]}...")
        if "dolphy" in result['response'].lower():
            print("🎉 SUCCESS: Memory sharing works across sessions!")
            return True
        else:
            print("❌ FAILED: Memory is not shared across sessions")
            print(f"   Expected: Response should mention 'Dolphy'")
            print(f"   Got: {result['response']}")
            return False
    else:
        print("❌ Session 2 chat failed")
        return False

def test_memory_statistics():
    """Test memory statistics"""
    base_url = "http://localhost:8080"

    session = requests.Session()
    login_response = session.post(f"{base_url}/auth/login", json={"password": "admin123"})

    if login_response.status_code == 200:
        token = login_response.json().get("session_token")

        stats_response = session.get(f"{base_url}/api/knowledge/stats",
                                  headers={"Authorization": f"Bearer {token}"})

        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"\n📊 Memory Statistics:")
            print(f"   Total entries: {stats.get('total_entries', 0)}")
            print(f"   Entries by category: {stats.get('entries_by_category', {})}")
            return True

    return False

if __name__ == "__main__":
    print("🧪 Testing Memory Sharing Across Sessions")
    print("="*50)

    success = test_memory_sharing()

    if success:
        print("\n" + "="*50)
        print("🎉 Memory sharing test PASSED!")
        print("The AI can now remember user information across different browser sessions.")
    else:
        print("\n" + "="*50)
        print("❌ Memory sharing test FAILED!")
        print("Memory is still isolated between browser sessions.")

    print("\n📊 Checking memory statistics...")
    test_memory_statistics()