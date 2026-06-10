"""End-to-end authentication test."""
import requests
import json

base = "http://127.0.0.1:8000/api"

# 1. Register
print("=== REGISTER ===")
r = requests.post(f"{base}/auth/register", json={
    "email": "test@farm.com",
    "password": "password123",
    "display_name": "Test Farmer"
})
print(f"Status: {r.status_code}")
data = r.json()
assert r.status_code == 200, f"Register failed: {data}"
token = data["token"]
user = data["user"]
print(f"User: {user['user_id']} ({user['email']})")
print(f"Name: {user['display_name']}")
print(f"Token: {token[:30]}...")

# 2. Login
print("\n=== LOGIN ===")
r = requests.post(f"{base}/auth/login", json={
    "email": "test@farm.com",
    "password": "password123"
})
assert r.status_code == 200, f"Login failed: {r.json()}"
data = r.json()
token = data["token"]
print(f"Status: {r.status_code}")
print(f"Token: {token[:30]}...")

# 3. Get profile
print("\n=== GET ME ===")
headers = {"Authorization": f"Bearer {token}"}
r = requests.get(f"{base}/auth/me", headers=headers)
assert r.status_code == 200, f"Get me failed: {r.json()}"
me = r.json()
print(f"Status: {r.status_code}")
print(f"Display name: {me['display_name']}")
print(f"Email: {me['email']}")

# 4. Send a chat with auth
print("\n=== CHAT (AUTHENTICATED) ===")
r = requests.post(f"{base}/query", json={"query": "What is crop rotation?"}, headers=headers)
assert r.status_code == 200, f"Chat failed: {r.json()}"
chat = r.json()
print(f"Status: {r.status_code}")
print(f"Intent: {chat['intent']}")
print(f"Response: {chat['response_text'][:80]}...")

# 5. Get my history
print("\n=== MY HISTORY ===")
r = requests.get(f"{base}/my/history", headers=headers)
assert r.status_code == 200
hist = r.json()
print(f"Status: {r.status_code}")
print(f"Entries: {len(hist['history'])}")
assert len(hist["history"]) >= 1, "History should have at least 1 entry"
print(f"Last query: {hist['history'][0]['query']}")

# 6. Register another user and verify data isolation
print("\n=== REGISTER USER 2 ===")
r2 = requests.post(f"{base}/auth/register", json={
    "email": "farmer2@test.com",
    "password": "abc123",
    "display_name": "Farmer Two"
})
assert r2.status_code == 200
data2 = r2.json()
token2 = data2["token"]
print(f"User2: {data2['user']['display_name']}")

# User 2 should have EMPTY history (data isolation)
r2h = requests.get(f"{base}/my/history", headers={"Authorization": f"Bearer {token2}"})
hist2 = r2h.json()
print(f"User2 history: {len(hist2['history'])} entries (should be 0)")
assert len(hist2["history"]) == 0, "User 2 should have NO history (data isolation!)"

# 7. Duplicate email test
print("\n=== DUPLICATE EMAIL ===")
r3 = requests.post(f"{base}/auth/register", json={"email": "test@farm.com", "password": "xxx"})
print(f"Status: {r3.status_code} (should be 409)")
assert r3.status_code == 409, f"Should reject duplicate: {r3.json()}"
print(f"Detail: {r3.json()['detail']}")

# 8. Wrong password
print("\n=== WRONG PASSWORD ===")
r4 = requests.post(f"{base}/auth/login", json={"email": "test@farm.com", "password": "wrongpassword"})
print(f"Status: {r4.status_code} (should be 401)")
assert r4.status_code == 401

# 9. Update profile
print("\n=== UPDATE PROFILE ===")
r5 = requests.put(f"{base}/auth/profile", json={
    "display_name": "Test Farmer Pro",
    "language": "hi",
    "preferences": {"crops": ["wheat", "rice"], "region": "Punjab"}
}, headers=headers)
assert r5.status_code == 200
updated = r5.json()
print(f"Name: {updated['display_name']}")
print(f"Language: {updated['language']}")
print(f"Preferences: {updated['preferences']}")

# 10. Clear history
print("\n=== CLEAR HISTORY ===")
r6 = requests.delete(f"{base}/my/history", headers=headers)
assert r6.status_code == 200
print(f"Cleared: {r6.json()['cleared']} entries")

# Verify empty
r7 = requests.get(f"{base}/my/history", headers=headers)
assert len(r7.json()["history"]) == 0, "History should be empty after clear"

# 11. Unauthenticated access to /my/history → 401
print("\n=== UNAUTH ACCESS ===")
r8 = requests.get(f"{base}/my/history")
print(f"Status: {r8.status_code} (should be 401)")
assert r8.status_code == 401

print("\n✅ ALL 11 AUTH TESTS PASSED ✅")
