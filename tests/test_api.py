import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_get_verse():
    response = client.get("/api/v1/verse/Genesis/1/1")
    assert response.status_code == 200
    data = response.json()
    assert data["book"] == "Genesis"
    assert data["chapter"] == 1
    assert data["verse"] == 1
    assert "text" in data


def test_get_nonexistent_verse():
    response = client.get("/api/v1/verse/NonexistentBook/1/1")
    assert response.status_code == 404


def test_get_chapter():
    response = client.get("/api/v1/chapter/Genesis/1")
    assert response.status_code == 200
    data = response.json()
    assert data["book"] == "Genesis"
    assert data["chapter"] == 1
    assert "verses" in data
    assert isinstance(data["verses"], dict)


def test_search_bible():
    # Test with valid query
    response = client.get("/api/v1/search?q=beginning")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "count" in data
    assert "query" in data
    assert data["query"] == "beginning"

    # Test with empty query
    response = client.get("/api/v1/search")
    assert response.status_code == 422

    # Test with book filter
    response = client.get("/api/v1/search?q=beginning&book=Genesis")
    assert response.status_code == 200
    data = response.json()
    assert all(result["book"] == "Genesis" for result in data["results"])


def test_list_books():
    response = client.get("/api/v1/books")
    assert response.status_code == 200
    data = response.json()
    assert "books" in data
    assert "count" in data
    assert isinstance(data["books"], list)
    assert data["count"] > 0


def test_get_cross_references():
    response = client.get("/api/v1/cross-references/Genesis/1/1")
    assert response.status_code == 200
    data = response.json()
    assert "source_verse" in data
    assert "cross_references" in data
    assert "relationship_type" in data
    assert isinstance(data["cross_references"], list)
    assert len(data["cross_references"]) <= 5  # Verify limit is enforced


def test_get_verse_context():
    response = client.get("/api/v1/context/Genesis/1/1")
    assert response.status_code == 200
    data = response.json()
    assert "verse" in data
    assert "historical_context" in data
    assert "literary_context" in data
    assert "theological_context" in data
    assert "chapter_context" in data
    assert isinstance(data["chapter_context"], str)


def test_nonexistent_verse_context():
    response = client.get("/api/v1/context/NonexistentBook/1/1")
    assert response.status_code == 404
