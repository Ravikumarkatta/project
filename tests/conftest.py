# tests/conftest.py
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_bible_data():
    """Provide sample Bible data for testing."""
    return {
        "Genesis": {
            "1": {
                "1": "In the beginning God created the heaven and the earth.",
                "2": "And the earth was without form, and void.",
            }
        },
        "John": {
            "3": {
                "16": "For God so loved the world, that he gave his only begotten Son."
            }
        },
    }


@pytest.fixture
def theological_validator():
    """Provide theological validator instance for testing."""
    from src.theology.validator import TheologicalValidator

    return TheologicalValidator()
