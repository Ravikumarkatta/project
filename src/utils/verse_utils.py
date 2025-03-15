# src/utils/verse_utils.py

def is_valid_verse_reference(reference):
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
        "1 Kings": 22, "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36,
        "Ezra": 10, "Nehemiah": 13, "Esther": 10, "Job": 42, "Psalm": 150, "Psalms": 150,
        "Proverbs": 31, "Ecclesiastes": 12, "Song of Solomon": 8, "Isaiah": 66,
        "Jeremiah": 52, "Lamentations": 5, "Ezekiel": 48, "Daniel": 12, "Hosea": 14,
        "Joel": 3, "Amos": 9, "Obadiah": 1, "Jonah": 4, "Micah": 7, "Nahum": 3,
        "Habakkuk": 3, "Zephaniah": 3, "Haggai": 2, "Zechariah": 14, "Malachi": 4,
        "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21, "Acts": 28, "Romans": 16,
        "1 Corinthians": 16, "2 Corinthians": 13, "Galatians": 6, "Ephesians": 6,
        "Philippians": 4, "Colossians": 4, "1 Thessalonians": 5, "2 Thessalonians": 3,
        "1 Timothy": 6, "2 Timothy": 4, "Titus": 3, "Philemon": 1, "Hebrews": 13,
        "James": 5, "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1, "3 John": 1,
        "Jude": 1, "Revelation": 22
    }
    
    # Parse the reference
    parts = reference.split()
    
    # Handle books with spaces in their names
    book_name = ""
    chapter_verse = ""
    
    for i, part in enumerate(parts):
        if ":" in part or "-" in part or (part.isdigit() and i > 0):
            chapter_verse = part
            break
        else:
            if book_name:
                book_name += " " + part
            else:
                book_name = part
    
    # If we didn't find a chapter/verse, the last part might be a chapter
    if not chapter_verse and len(parts) > 1:
        if parts[-1].isdigit():
            chapter_verse = parts[-1]
            book_name = " ".join(parts[:-1])
    
    # If we still don't have a book name, return False
    if not book_name:
        return False
    
    # Check if the book exists
    if book_name not in valid_books:
        return False
    
    # If no chapter/verse specified, this is a book reference (valid)
    if not chapter_verse:
        return True
    
    # Parse chapter and verse
    if ":" in chapter_verse:
        # Chapter and verse specified (e.g., "3:16")
        try:
            chapter, verse = chapter_verse.split(":")
            chapter = int(chapter)
            
            # Handle verse ranges (e.g., "3:16-18")
            if "-" in verse:
                start_verse, end_verse = verse.split("-")
                start_verse = int(start_verse)
                end_verse = int(end_verse)
                # Simplified: assume verses are valid if they're positive numbers
                if start_verse <= 0 or end_verse <= 0 or start_verse > end_verse:
                    return False
            else:
                # Single verse
                verse = int(verse)
                if verse <= 0:
                    return False
        except ValueError:
            return False
    else:
        # Only chapter specified (e.g., "3")
        try:
            chapter = int(chapter_verse)
        except ValueError:
            return False
    
    # Check if chapter is valid for this book
    max_chapters = valid_books[book_name]
    if chapter <= 0 or chapter > max_chapters:
        return False
    
    # If we've made it here, the reference appears valid
    return True