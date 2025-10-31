#!/usr/bin/env python3
"""
Test both upload endpoints to diagnose multipart parsing issues
"""

import requests
import json
import tempfile
import os
from pathlib import Path

def test_main_endpoint():
    """Test the main multipart upload endpoint"""
    print("🧪 Testing main /upload endpoint...")

    # Use the sample CSV we created
    csv_file = "data/sample_XAUUSD_M15_200.csv"

    if not os.path.exists(csv_file):
        print(f"❌ Sample CSV not found: {csv_file}")
        return False

    with open(csv_file, 'rb') as f:
        files = {'file': (csv_file, f, 'text/csv')}
        data = {
            'symbol': 'XAUUSD',
            'timeframe': 'M15',
            'candles': '200'
        }

        try:
            response = requests.post(
                "http://localhost:8080/upload",
                files=files,
                data=data,
                timeout=30
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Main endpoint success!")
                print(f"Filename: {result.get('filename')}")
                return True
            else:
                print(f"❌ Main endpoint failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Main endpoint error: {e}")
            return False

def test_simple_endpoint():
    """Test the simple raw upload endpoint"""
    print("\n🧪 Testing simple /upload/simple endpoint...")

    # Use the sample CSV we created
    csv_file = "data/sample_XAUUSD_M15_200.csv"

    if not os.path.exists(csv_file):
        print(f"❌ Sample CSV not found: {csv_file}")
        return False

    with open(csv_file, 'rb') as f:
        # Create a multipart boundary manually to test format
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

        # Build multipart body manually
        body = []
        body.append(f"--{boundary}\r\n")
        body.append('Content-Disposition: form-data; name="file"; filename="XAUUSD_PERIOD_M15_200.csv"\r\n')
        body.append('Content-Type: text/csv\r\n\r\n')

        # Add file content
        content = f.read()
        body.append(content.decode('utf-8'))
        body.append(f"\r\n--{boundary}--\r\n")

        multipart_body = '\n'.join(body)

        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(multipart_body.encode('utf-8')))
        }

        try:
            response = requests.post(
                "http://localhost:8080/upload/simple",
                data=multipart_body.encode('utf-8'),
                headers=headers,
                timeout=30
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Simple endpoint success!")
                print(f"Filename: {result.get('filename')}")
                return True
            else:
                print(f"❌ Simple endpoint failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Simple endpoint error: {e}")
            return False

def test_curl_command():
    """Generate curl command for manual testing"""
    print("\n📋 Curl command for manual testing:")

    csv_file = os.path.abspath("data/sample_XAUUSD_M15_200.csv")

    print(f"""
# Test main endpoint
curl -X POST "http://localhost:8080/upload" \\
  -F "file=@{csv_file}" \\
  -F "symbol=XAUUSD" \\
  -F "timeframe=M15" \\
  -F "candles=200"

# Test simple endpoint
curl -X POST "http://localhost:8080/upload/simple" \\
  -H "Content-Type: text/csv" \\
  --data-binary @{csv_file}
""")

def main():
    """Main test function"""
    print("🚀 MT5 Upload Endpoint Test Suite")
    print("="*50)

    print("This test will help diagnose the multipart parsing issues.")
    print("Please ensure your FastAPI server is running on http://localhost:8080")
    print()

    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned {response.status_code}")
            return
    except:
        print("❌ Server is not running or not accessible")
        print("Please start your FastAPI server first: python main.py")
        return

    # Test both endpoints
    main_success = test_main_endpoint()
    simple_success = test_simple_endpoint()

    # Generate curl commands
    test_curl_command()

    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"Main endpoint (/upload): {'✅ SUCCESS' if main_success else '❌ FAILED'}")
    print(f"Simple endpoint (/upload/simple): {'✅ SUCCESS' if simple_success else '❌ FAILED'}")

    if main_success or simple_success:
        print("\n✅ At least one endpoint works! Use the working one in your MT5 EA.")
    else:
        print("\n❌ Both endpoints failed. Check server logs for more details.")

if __name__ == "__main__":
    main()