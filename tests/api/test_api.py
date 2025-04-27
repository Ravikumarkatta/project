import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_single_prediction():
    test_data = {
        "text": "For God so loved the world",
        "context": "John 3:16"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "prediction" in data
    assert "confidence" in data
    assert "theological_score" in data

def test_batch_prediction():
    test_data = {
        "texts": ["In the beginning", "The Lord is my shepherd"],
        "contexts": ["Genesis 1:1", "Psalm 23:1"]
    }
    response = client.post("/predict_batch", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2

def test_rate_limiting():
    for _ in range(100):
        response = client.get("/health")
        assert response.status_code == 200
    
    response = client.get("/health")
    assert response.status_code == 429
