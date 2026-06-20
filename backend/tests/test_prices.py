from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prices_endpoint_unauthorized(client):
    response = client.get("/prices/?commodity=wheat&district=dharwad&state=karnataka")
    assert response.status_code == 401
