import pytest
from app import app

@pytest.fixture
def client():
    """Create a test client for the app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test the home endpoint returns the correct message."""
    response = client.get('/')
    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data['message'] == "Welcome to Bible AI backend service."

def test_list_bibles_no_downloads(client, monkeypatch, tmp_path):
    """Test listing bibles when no downloads exist."""
    import os
    # Mock the DOWNLOAD_DIR to a temporary directory
    monkeypatch.setattr('app.DOWNLOAD_DIR', str(tmp_path))
    
    response = client.get('/bibles')
    assert response.status_code == 404
    assert 'error' in response.get_json()

def test_list_bibles_with_downloads(client, monkeypatch, tmp_path):
    """Test listing bibles when downloads exist."""
    import os
    # Mock the DOWNLOAD_DIR to a temporary directory
    monkeypatch.setattr('app.DOWNLOAD_DIR', str(tmp_path))
    
    # Create some fake Bible files
    (tmp_path / "KJV.txt").write_text("King James Version content")
    (tmp_path / "NIV.txt").write_text("New International Version content")
    
    response = client.get('/bibles')
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'bibles' in json_data
    assert set(json_data['bibles']) == {'KJV', 'NIV'}
