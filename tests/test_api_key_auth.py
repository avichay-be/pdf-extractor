"""
Test Bearer token authentication.

This script tests:
1. Request without Bearer token (should fail with 401)
2. Request with invalid Bearer token (should fail with 403)
3. Request with valid Bearer token (should succeed)
"""
import requests
import sys
from pathlib import Path

# API configuration
API_URL = "http://localhost:8000"
TEST_API_KEY = "test_key_12345"  # Set this in your .env file as API_KEY


def test_no_bearer_token():
    """Test request without Bearer token."""
    print("=" * 80)
    print("TEST 1: Request without Bearer token")
    print("=" * 80)

    response = requests.get(f"{API_URL}/")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 401:
        print("✅ PASS: Got expected 401 Unauthorized")
        # Verify Bearer token is mentioned in error message
        detail = response.json().get("detail", "")
        if "Bearer" in detail:
            print("✅ PASS: Error message mentions Bearer token")
        else:
            print("⚠️  WARNING: Error message doesn't mention Bearer token")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")

    print()


def test_invalid_bearer_token():
    """Test request with invalid Bearer token."""
    print("=" * 80)
    print("TEST 2: Request with invalid Bearer token")
    print("=" * 80)

    headers = {"Authorization": "Bearer wrong_key"}
    response = requests.get(f"{API_URL}/", headers=headers)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 403:
        print("✅ PASS: Got expected 403 Forbidden")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")

    print()


def test_valid_bearer_token():
    """Test request with valid Bearer token."""
    print("=" * 80)
    print("TEST 3: Request with valid Bearer token")
    print("=" * 80)

    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    response = requests.get(f"{API_URL}/", headers=headers)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 200:
        print("✅ PASS: Got expected 200 OK")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")

    print()


def test_health_endpoint():
    """Test health endpoint with valid Bearer token."""
    print("=" * 80)
    print("TEST 4: Health endpoint with valid Bearer token")
    print("=" * 80)

    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    response = requests.get(f"{API_URL}/health", headers=headers)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 200:
        print("✅ PASS: Got expected 200 OK")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BEARER TOKEN AUTHENTICATION TESTS")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"Test Bearer Token: {TEST_API_KEY}")
    print()
    print("IMPORTANT: Make sure you have:")
    print("  1. Set API_KEY=test_key_12345 in your .env file")
    print("  2. Set REQUIRE_API_KEY=true in your .env file")
    print("  3. Started the server with: python run.py")
    print()
    print("NOTE: Authentication now uses Bearer token format:")
    print("  Authorization: Bearer <token>")
    print()
    input("Press Enter to continue...")
    print()

    try:
        # Test 1: No Bearer token
        test_no_bearer_token()

        # Test 2: Invalid Bearer token
        test_invalid_bearer_token()

        # Test 3: Valid Bearer token
        test_valid_bearer_token()

        # Test 4: Health endpoint
        test_health_endpoint()

        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
