# src/utils/verse_utils.py

# 1.1: Add type hints for function signature
from typing import Optional

# Corrected function definition with type hints
# Addresses: src/utils/verse_utils.py:4: error: Function is missing a type annotation [no-untyped-def]
def is_valid_verse_reference(reference: str) -> bool:
    """
    Validate if a given string is a valid Bible verse reference.

    Args:
        reference (str): The Bible verse reference to validate

    Returns:
        bool: True if the reference is valid, False otherwise
    """
    # This is a simplified implementation
    # In a real-world scenario, you'd want to check against actual Bible data

    # Basic validation rules:
    # 1. Must contain a book name and chapter number
    # 2. Book must be a valid Bible book
    # 3. Chapter must be within valid range for that book
    # 4. If verse is specified, it must be within valid range for that chapter

    valid_books = {
        "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
        "Joshua": 24, "Judges": 21, "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24,
        "1 Kings": 22, "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36, "Ezra": 10,
        "Nehemiah": 13, "Esther": 10, "Job": 42, "Psalm": 150, "Psalms": 150,
        "Proverbs": 31, "Ecclesiastes": 12, "Song of Solomon": 8, "Isaiah": 66,
        "Jeremiah": 52, "Lamentations": 5, "Ezekiel": 48, "Daniel": 12, "Hosea": 14,
        "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4, "Micah": 7, "Nahum": 3,
        "Habakkuk": 3, "Zephaniah": 3, "Haggai": 2, "Zechariah": 14, "Malachi": 4,
        "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21, "Acts": 28, "Romans": 16,
        "1 Corinthians": 16, "2 Corinthians": 13, "Galatians": 6, "Ephesians": 6,
        "Philippians": 4, "Colossians": 4, "1 Thessalonians": 5, "2 Thessalonians": 3,
        "1 Timothy": 6, "2 Timothy": 4, "Titus": 3, "Philemon": 1, "Hebrews": 13,
        "James": 5, "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1, "3 John": 1,
        "Jude": 1, "Revelation": 22,
    }

    # Parse the reference
    parts = reference.split()

    # Handle books with spaces in their names
    book_name: str = ""
    chapter_verse_str: str = "" # Use a distinct name for the string part

    for i, part in enumerate(parts):
        # Check if the part looks like a chapter/verse indicator
        if ":" in part or "-" in part or (part.isdigit() and i > 0):
            chapter_verse_str = part
            # The book name is everything before this part
            book_name = " ".join(parts[:i])
            break
        # else: continue building book_name implicitly handled by loop

    # If we didn't find a chapter/verse indicator, the last part might be a chapter
    # or the whole thing might be just a book name
    if not chapter_verse_str:
        if len(parts) > 1 and parts[-1].isdigit():
            # Assume last part is chapter, rest is book
            chapter_verse_str = parts[-1]
            book_name = " ".join(parts[:-1])
        elif len(parts) >= 1:
             # Assume the whole reference is just the book name
             book_name = " ".join(parts)
             # chapter_verse_str remains empty

    # If we still don't have a book name, return False
    if not book_name:
        return False

    # Check if the book exists
    if book_name not in valid_books:
        # Try normalizing common aliases (like Psalm -> Psalms)
        normalized_book = "Psalms" if book_name == "Psalm" else book_name
        if normalized_book not in valid_books:
            return False
        book_name = normalized_book # Use the normalized name going forward

    # If no chapter/verse specified, this is a book reference (valid)
    if not chapter_verse_str:
        return True

    # 1.2 & 1.3: Use distinct variables for int versions and perform checks on ints
    # These declarations fix the core issue behind most errors.
    chapter_num: Optional[int] = None
    verse_num: Optional[int] = None
    start_verse_num: Optional[int] = None
    end_verse_num: Optional[int] = None

    # Parse chapter and verse
    if ":" in chapter_verse_str:
        # Chapter and verse specified (e.g., "3:16")
        try:
            chapter_str, verse_part_str = chapter_verse_str.split(":", 1)
            # Assign result of int() to the integer variable chapter_num
            # Addresses: src/utils/verse_utils.py:133: error: Incompatible types in assignment...
            chapter_num = int(chapter_str)

            # Handle verse ranges (e.g., "3:16-18")
            if "-" in verse_part_str:
                start_verse_str, end_verse_str = verse_part_str.split("-", 1)
                # Assign result of int() to the integer variables
                # Addresses: src/utils/verse_utils.py:138: error: Incompatible types in assignment...
                start_verse_num = int(start_verse_str)
                # Addresses: src/utils/verse_utils.py:139: error: Incompatible types in assignment...
                end_verse_num = int(end_verse_str)

                # Check using integer variables (start_verse_num, end_verse_num)
                # Addresses: src/utils/verse_utils.py:141: error: Unsupported operand types...
                if start_verse_num <= 0 or end_verse_num <= 0 or start_verse_num > end_verse_num:
                    return False
            else:
                # Single verse
                # Assign result of int() to the integer variable verse_num
                # Addresses: src/utils/verse_utils.py:145: error: Incompatible types in assignment...
                verse_num = int(verse_part_str)
                # Check using integer variable (verse_num)
                # Addresses: src/utils/verse_utils.py:146: error: Unsupported operand types...
                if verse_num <= 0:
                    return False
        except ValueError:
            # Handle cases like "John 3:abc" or "John 3:16-abc"
            return False
    elif chapter_verse_str.isdigit(): # type: ignore # Keep ignore for now, logic seems okay
        # Only chapter specified (e.g., "3")
        try:
            # Assign result of int() to the integer variable chapter_num
            # Addresses: src/utils/verse_utils.py:153: error: Incompatible types in assignment...
            chapter_num = int(chapter_verse_str)
        except ValueError:
            # Should not happen if isdigit() is true, but good practice
            return False
    else:
         # Invalid format if it's not a number and doesn't contain ':'
         return False


    # Check if chapter is valid for this book (only if chapter_num was parsed)
    if chapter_num is not None:
        max_chapters = valid_books[book_name]
        # Check using integer variable (chapter_num)
        # Addresses: src/utils/verse_utils.py:159: error: Unsupported operand types... (both errors)
        if chapter_num <= 0 or chapter_num > max_chapters:
            return False

    # If we've made it here, the reference structure appears valid
    # Note: This doesn't validate verse numbers against chapter lengths, only basic structure.
    return True
