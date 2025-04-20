import pytest

from src.bible_manager.verse_reference import VerseReference, VerseReferenceDetector


@pytest.fixture
def detector():
    return VerseReferenceDetector()


def test_single_verse_detection(detector):
    text = "Let's look at John 3:16 for this study."
    refs = detector.detect_references(text)
    assert len(refs) == 1
    assert refs[0].book == "John"
    assert refs[0].chapter == 3
    assert refs[0].verse == 16


def test_multiple_verse_detection(detector):
    text = "Read Genesis 1:1 and John 3:16"
    refs = detector.detect_references(text)
    assert len(refs) == 2
    assert refs[0].book == "Genesis"
    assert refs[1].book == "John"


def test_verse_range_detection(detector):
    text = "Study Matthew 5:3-12 for the Beatitudes"
    refs = detector.detect_references(text)
    assert len(refs) == 1
    assert refs[0].book == "Matthew"
    assert refs[0].chapter == 5
    assert refs[0].verse == 3
    assert refs[0].end_verse == 12


def test_cross_chapter_range(detector):
    text = "Read Psalm 1:1-2:3"
    refs = detector.detect_references(text)
    assert len(refs) == 1
    assert refs[0].book == "Psalms"
    assert refs[0].chapter == 1
    assert refs[0].verse == 1
    assert refs[0].end_chapter == 2
    assert refs[0].end_verse == 3


def test_whole_chapter_reference(detector):
    text = "Read Psalm 23"
    refs = detector.detect_references(text)
    assert len(refs) == 1
    assert refs[0].book == "Psalms"
    assert refs[0].chapter == 23
    assert refs[0].verse is None


def test_abbreviations(detector):
    text = "Compare Gen 1:1, Jn 3:16, and Ps 23"
    refs = detector.detect_references(text)
    assert len(refs) == 3
    assert refs[0].book == "Genesis"
    assert refs[1].book == "John"
    assert refs[2].book == "Psalms"


def test_normalize_reference(detector):
    ref = detector.normalize_reference("John 3:16")
    assert ref is not None
    assert ref.book == "John"
    assert ref.chapter == 3
    assert ref.verse == 16


def test_format_reference(detector):
    # Test single verse
    ref = VerseReference(book="John", chapter=3, verse=16)
    assert detector.format_reference(ref) == "John 3:16"

    # Test verse range
    ref = VerseReference(book="Matthew", chapter=5, verse=3, end_verse=12)
    assert detector.format_reference(ref) == "Matthew 5:3-12"

    # Test chapter range
    ref = VerseReference(book="Psalms", chapter=1, verse=1, end_chapter=2, end_verse=3)
    assert detector.format_reference(ref) == "Psalms 1:1-2:3"

    # Test whole chapter
    ref = VerseReference(book="Psalms", chapter=23)
    assert detector.format_reference(ref) == "Psalms 23"


def test_invalid_references(detector):
    # Should return None for invalid references
    assert detector.normalize_reference("Invalid 1:1") is None
    assert detector.normalize_reference("Genesis 0:0") is None

    # Should return empty list for text with no valid references
    assert len(detector.detect_references("No references here")) == 0
