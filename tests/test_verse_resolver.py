import pytest
from src.serve.verse_resolver import VerseResolver, VerseReference
from pathlib import Path
import json

@pytest.fixture
def sample_bible_data():
    return {
        "Genesis": {
            "1": {
                "1": "In the beginning God created the heaven and the earth.",
                "2": "And the earth was without form, and void."
            },
            "2": {
                "1": "Thus the heavens and the earth were finished.",
                "2": "And on the seventh day God ended his work."
            }
        },
        "John": {
            "3": {
                "16": "For God so loved the world, that he gave his only begotten Son.",
                "17": "For God sent not his Son into the world to condemn the world."
            }
        }
    }

@pytest.fixture
def bible_data_file(tmp_path, sample_bible_data):
    data_file = tmp_path / "test_bible.json"
    with open(data_file, 'w') as f:
        json.dump(sample_bible_data, f)
    return str(data_file)

@pytest.fixture
def verse_resolver(bible_data_file):
    return VerseResolver(bible_data_file)

def test_parse_reference(verse_resolver):
    # Test standard format
    ref = verse_resolver.parse_reference("Genesis 1:1")
    assert ref.book == "Genesis"
    assert ref.chapter == 1
    assert ref.verse == 1
    assert ref.end_verse is None
    
    # Test verse range
    ref = verse_resolver.parse_reference("John 3:16-17")
    assert ref.book == "John"
    assert ref.chapter == 3
    assert ref.verse == 16
    assert ref.end_verse == 17
    
    # Test abbreviated book name
    ref = verse_resolver.parse_reference("Gen 1:2")
    assert ref.book == "Genesis"
    assert ref.chapter == 1
    assert ref.verse == 2

def test_validate_reference(verse_resolver):
    # Test valid references
    assert verse_resolver.validate_reference(VerseReference("Genesis", 1, 1))
    assert verse_resolver.validate_reference(VerseReference("John", 3, 16))
    
    # Test invalid references
    assert not verse_resolver.validate_reference(VerseReference("Genesis", 100, 1))  # Invalid chapter
    assert not verse_resolver.validate_reference(VerseReference("Genesis", 1, 100))  # Invalid verse
    assert not verse_resolver.validate_reference(VerseReference("InvalidBook", 1, 1))  # Invalid book

def test_resolve_references(verse_resolver):
    text = "Look at Genesis 1:1-2 and John 3:16"
    refs = verse_resolver.resolve_references(text)
    
    assert len(refs) == 2
    assert refs[0]["reference"] == "Genesis 1:1-2"
    assert refs[1]["reference"] == "John 3:16"

def test_get_verse_text(verse_resolver):
    # Test single verse
    text = verse_resolver.get_verse_text("Genesis 1:1")
    assert text == "In the beginning God created the heaven and the earth."
    
    # Test verse range
    text = verse_resolver.get_verse_text("John 3:16-17")
    expected = "For God so loved the world, that he gave his only begotten Son. For God sent not his Son into the world to condemn the world."
    assert text == expected
    
    # Test invalid reference
    assert verse_resolver.get_verse_text("InvalidBook 1:1") is None