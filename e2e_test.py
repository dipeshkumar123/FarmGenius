import requests
import sys

BASE_URL = "https://farmgenius-monorepo.vercel.app/api"

def run_tests():
    print(f"Testing FarmGenius E2E at {BASE_URL}...")
    
    # 1. Test Auth OTP verification (Minting Custom JWT)
    print("\n--- Testing Auth (/auth/verify-otp) ---")
    auth_payload = {"phone": "1234567890", "otp": "123456"}
    headers_base = {"Bypass-Tunnel-Reminder": "true", "Content-Type": "application/json"}
    try:
        response = requests.post(f"{BASE_URL}/auth/verify-otp", json=auth_payload, headers=headers_base)
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            print("[FAIL] No token returned.")
            sys.exit(1)
        print("[SUCCESS] Auth successful. Token received.")
    except Exception as e:
        print(f"[FAIL] Auth failed: {e}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}", "Bypass-Tunnel-Reminder": "true"}

    # 2. Test Weather API
    print("\n--- Testing Weather (/weather) ---")
    try:
        weather_res = requests.get(f"{BASE_URL}/weather?district=dharwad&state=karnataka", headers=headers)
        weather_res.raise_for_status()
        print("[SUCCESS] Weather fetched successfully:")
        print(weather_res.json()[:1])
    except Exception as e:
        print(f"[FAIL] Weather fetch failed: {e}")
        
    # 3. Test Chat API
    print("\n--- Testing Chat (/chat) ---")
    chat_payload = {"query": "How to grow wheat?", "language": "en", "farmer_id": "1234567890"}
    try:
        chat_res = requests.post(f"{BASE_URL}/chat/", json=chat_payload, headers=headers)
        chat_res.raise_for_status()
        print("[SUCCESS] Chat response received:")
        print(chat_res.json())
    except Exception as e:
        print(f"[FAIL] Chat fetch failed: {e}")

    print("\n[COMPLETE] All End-to-End API tests completed.")

if __name__ == "__main__":
    run_tests()
