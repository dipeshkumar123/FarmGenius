from fastapi.testclient import TestClient
from main import app
import io

def test_disease_endpoint_unauthorized(client):
    file = io.BytesIO(b"dummy image bytes content")
    response = client.post(
        "/disease/detect", 
        files={"file": ("test.jpg", file, "image/jpeg")}
    )
    assert response.status_code == 401
