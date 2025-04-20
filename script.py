import json

# List of the 66 books in the Protestant Bible
canonical_books = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
    "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Jude", "Revelation"
]

# Expected chapter counts per book
expected_chapter_counts = {
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numbers": 36, "Deuteronomy": 34,
    "Joshua": 24, "Judges": 21, "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24,
    "1 Kings": 22, "2 Kings": 25, "1 Chronicles": 29, "2 Chronicles": 36, "Ezra": 10,
    "Nehemiah": 13, "Esther": 10, "Job": 42, "Psalms": 150, "Proverbs": 31,
    "Ecclesiastes": 12, "Song of Solomon": 8, "Isaiah": 66, "Jeremiah": 52, "Lamentations": 5,
    "Ezekiel": 48, "Daniel": 12, "Hosea": 14, "Joel": 3, "Amos": 9,
    "Obadiah": 1, "Jonah": 4, "Micah": 7, "Nahum": 3, "Habakkuk": 3,
    "Zephaniah": 3, "Haggai": 2, "Zechariah": 14, "Malachi": 4,
    "Matthew": 28, "Mark": 16, "Luke": 24, "John": 21, "Acts": 28,
    "Romans": 16, "1 Corinthians": 16, "2 Corinthians": 13, "Galatians": 6, "Ephesians": 6,
    "Philippians": 4, "Colossians": 4, "1 Thessalonians": 5, "2 Thessalonians": 3, "1 Timothy": 6,
    "2 Timothy": 4, "Titus": 3, "Philemon": 1, "Hebrews": 13, "James": 5,
    "1 Peter": 5, "2 Peter": 3, "1 John": 5, "2 John": 1, "3 John": 1,
    "Jude": 1, "Revelation": 22
}

def load_json(filename):
    """Loads a JSON file and returns the parsed data."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Loaded data from '{filename}' (first 100 chars): {str(data)[:100]}...")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file '{filename}': {e}")
        return None

def convert_flat_to_structured(flat_data):
    """Converts flat book-chapter-verse structure to the expected 'books' format."""
    books = []
    for book_name, chapters in flat_data.items():
        book = {
            "name": book_name,
            "code": book_name[:3].upper(),
            "chapters": []
        }
        for chapter_num, verses in chapters.items():
            chapter = {
                "chapter": chapter_num,
                "verses": [{"verse": v_num, "text": v_text} for v_num, v_text in verses.items()]
            }
            book["chapters"].append(chapter)
        books.append(book)
    return {"books": books}

def validate_books(bible_data):
    """Checks for missing books."""
    books_in_file = {book["name"] for book in bible_data["books"]}
    missing_books = [book for book in canonical_books if book not in books_in_file]
    if missing_books:
        print("Missing books:", missing_books)
    else:
        print("All 66 books are present.")
    return missing_books

def count_chapters_and_verses(book_data):
    """Counts total chapters and verses in a book."""
    chapter_count = len(book_data.get("chapters", []))
    verse_count = sum(len(chap.get("verses", [])) for chap in book_data.get("chapters", []))
    return chapter_count, verse_count

def validate_chapter_counts(bible_data):
    """Validates chapter counts against expected values."""
    books = bible_data.get("books", [])
    if not books:
        print("No books found to validate chapter counts.")
        return []
    discrepancies = []
    for book in books:
        book_name = book["name"]
        actual_chapters = len(book.get("chapters", []))
        expected_chapters = expected_chapter_counts.get(book_name, 0)
        if actual_chapters != expected_chapters:
            discrepancies.append(
                f"{book_name}: Expected {expected_chapters} chapters, found {actual_chapters}"
            )
    if discrepancies:
        print("Chapter count discrepancies found:")
        for discrepancy in discrepancies:
            print(f"  - {discrepancy}")
    else:
        print("All chapter counts match expected values.")
    return discrepancies

def build_structured_json(bible_data):
    """Creates a structured JSON file with enhanced metadata."""
    structured_bible = {
        "books": [],
        "metadata": {
            "translation": "kjv",
            "name": "King James Version",
            "format": "json_structured",
            "source": "processed_file",
            "book_count": 0,
            "chapter_count": 0,
            "verse_count": 0
        }
    }
    total_chapters, total_verses = 0, 0

    # Convert flat data if necessary
    if "books" not in bible_data:
        bible_data = convert_flat_to_structured(bible_data)

    for book_name in canonical_books:
        book_data = next((b for b in bible_data["books"] if b["name"] == book_name), None)
        if book_data:
            chapter_count, verse_count = count_chapters_and_verses(book_data)
            total_chapters += chapter_count
            total_verses += verse_count
            structured_bible["books"].append({
                "name": book_data["name"],
                "code": book_data["code"],
                "chapters": book_data["chapters"]
            })
        else:
            print(f"Warning: '{book_name}' not found in input data, adding placeholder.")
            structured_bible["books"].append({
                "name": book_name,
                "code": book_name[:3].upper(),
                "chapters": []
            })

    structured_bible["metadata"]["book_count"] = len(structured_bible["books"])
    structured_bible["metadata"]["chapter_count"] = total_chapters
    structured_bible["metadata"]["verse_count"] = total_verses
    return structured_bible

def save_json(data, filename):
    """Saves a JSON file with proper formatting and error handling."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"New structured JSON file '{filename}' has been created.")
    except IOError as e:
        print(f"Error writing JSON file '{filename}': {e}")

if __name__ == "__main__":
    input_filename = 'kjv_processed.json'
    output_filename = 'kjv_structured_complete.json'
    
    bible_data = load_json(input_filename)
    if bible_data is not None:
        missing_books = validate_books(convert_flat_to_structured(bible_data))
        validate_chapter_counts(convert_flat_to_structured(bible_data))
        structured_bible = build_structured_json(bible_data)
        save_json(structured_bible, output_filename)