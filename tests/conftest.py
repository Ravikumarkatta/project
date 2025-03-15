# tests/conftest.py
import pytest
import os
import sys

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_bible_data():
    """Fixture that provides some sample Bible data for tests"""
    return {
        "verses": {
            "John 3:16": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
            "Genesis 1:1": "In the beginning God created the heavens and the earth.",
            "Psalm 23:1": "The LORD is my shepherd, I lack nothing."
        }
    }