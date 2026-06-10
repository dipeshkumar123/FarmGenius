from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_endpoint_unauthorized():
    response = client.post("/chat/", json={
        "query": "hello",
        "language": "en",
        "farmer_id": "123"
    })
    # Expecting 401 because no Bearer token was provided
    assert response.status_code == 401
