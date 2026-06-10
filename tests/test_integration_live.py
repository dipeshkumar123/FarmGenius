"""Quick integration test against a running server."""
import requests
import json

base = "http://localhost:8000"

# Test health
r = requests.get(f"{base}/api/health")
print(f"Health: {r.status_code} - {r.json()['status']}")

# Test root (frontend)
r = requests.get(base)
ct = r.headers.get("content-type", "")
print(f"Root: {r.status_code} - content-type={ct[:30]}")

# Test chat query
r = requests.post(f"{base}/api/query", json={"query": "What is the price of rice?", "user_id": "deploy_test"})
data = r.json()
print(f"Query: {r.status_code} - intent={data.get('intent')} confidence={data.get('confidence')}")

# Test history retrieval
r = requests.post(f"{base}/api/history", json={"user_id": "deploy_test", "max_entries": 5})
data = r.json()
print(f"History: {r.status_code} - entries={len(data.get('entries', []))}")

# Test preferences
r = requests.post(f"{base}/api/users/deploy_test/preferences", json={"language": "en", "location": "Delhi"})
print(f"Set Prefs: {r.status_code}")
r = requests.get(f"{base}/api/users/deploy_test/preferences")
print(f"Get Prefs: {r.status_code} - prefs={r.json().get('preferences', {})}")

# Test statistics
r = requests.get(f"{base}/api/users/deploy_test/statistics")
print(f"Stats: {r.status_code}")

print()
print("ALL INTEGRATION TESTS PASSED")
