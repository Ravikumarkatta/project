from standalone_verse_detector import StandaloneVerseDetector, VerseReference


def test_detector():
    detector = StandaloneVerseDetector()

    # Test same chapter range
    text1 = "Study Matthew 5:3-12 for the Beatitudes"
    refs1 = detector.detect_references(text1)
    print(f"Matthew 5:3-12 -> {len(refs1)} refs: {refs1}")
    assert len(refs1) == 1

    # Test cross-chapter range
    text2 = "Read Psalm 1:1-2:3"
    refs2 = detector.detect_references(text2)
    print(f"Psalm 1:1-2:3 -> {len(refs2)} refs: {refs2}")
    assert len(refs2) == 1

    # Test multiple references
    text3 = "Look at Genesis 1:1-2 and John 3:16"
    refs3 = detector.detect_references(text3)
    print(f"Genesis 1:1-2 and John 3:16 -> {len(refs3)} refs: {refs3}")
    assert len(refs3) == 2

    print("All tests passed!")


if __name__ == "__main__":
    test_detector()
