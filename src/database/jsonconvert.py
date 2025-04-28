#!/usr/bin/env python3
"""
Bible TXT to SQL Converter

This script reads the Bible text data from file and converts it into a structured SQLite database.
It creates tables for books, chapters, verses, and cross-references, and populates them with data.
"""

import os
import re
import sqlite3
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Regular expressions for parsing Bible text
BOOK_PATTERN = re.compile(r'^(?P<book>[1-3A-Za-z\s]+)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+)$')

# Book metadata - maps book names to their testament and position
BOOK_META = {
    'Genesis': {'testament': 'old', 'position': 1},
    'Exodus': {'testament': 'old', 'position': 2},
    'Leviticus': {'testament': 'old', 'position': 3},
    'Numbers': {'testament': 'old', 'position': 4},
    'Deuteronomy': {'testament': 'old', 'position': 5},
    'Joshua': {'testament': 'old', 'position': 6},
    'Judges': {'testament': 'old', 'position': 7},
    'Ruth': {'testament': 'old', 'position': 8},
    '1 Samuel': {'testament': 'old', 'position': 9},
    '2 Samuel': {'testament': 'old', 'position': 10},
    '1 Kings': {'testament': 'old', 'position': 11},
    '2 Kings': {'testament': 'old', 'position': 12},
    '1 Chronicles': {'testament': 'old', 'position': 13},
    '2 Chronicles': {'testament': 'old', 'position': 14},
    'Ezra': {'testament': 'old', 'position': 15},
    'Nehemiah': {'testament': 'old', 'position': 16},
    'Esther': {'testament': 'old', 'position': 17},
    'Job': {'testament': 'old', 'position': 18},
    'Psalms': {'testament': 'old', 'position': 19},
    'Proverbs': {'testament': 'old', 'position': 20},
    'Ecclesiastes': {'testament': 'old', 'position': 21},
    'Song of Solomon': {'testament': 'old', 'position': 22},
    'Isaiah': {'testament': 'old', 'position': 23},
    'Jeremiah': {'testament': 'old', 'position': 24},
    'Lamentations': {'testament': 'old', 'position': 25},
    'Ezekiel': {'testament': 'old', 'position': 26},
    'Daniel': {'testament': 'old', 'position': 27},
    'Hosea': {'testament': 'old', 'position': 28},
    'Joel': {'testament': 'old', 'position': 29},
    'Amos': {'testament': 'old', 'position': 30},
    'Obadiah': {'testament': 'old', 'position': 31},
    'Jonah': {'testament': 'old', 'position': 32},
    'Micah': {'testament': 'old', 'position': 33},
    'Nahum': {'testament': 'old', 'position': 34},
    'Habakkuk': {'testament': 'old', 'position': 35},
    'Zephaniah': {'testament': 'old', 'position': 36},
    'Haggai': {'testament': 'old', 'position': 37},
    'Zechariah': {'testament': 'old', 'position': 38},
    'Malachi': {'testament': 'old', 'position': 39},
    'Matthew': {'testament': 'new', 'position': 40},
    'Mark': {'testament': 'new', 'position': 41},
    'Luke': {'testament': 'new', 'position': 42},
    'John': {'testament': 'new', 'position': 43},
    'Acts': {'testament': 'new', 'position': 44},
    'Romans': {'testament': 'new', 'position': 45},
    '1 Corinthians': {'testament': 'new', 'position': 46},
    '2 Corinthians': {'testament': 'new', 'position': 47},
    'Galatians': {'testament': 'new', 'position': 48},
    'Ephesians': {'testament': 'new', 'position': 49},
    'Philippians': {'testament': 'new', 'position': 50},
    'Colossians': {'testament': 'new', 'position': 51},
    '1 Thessalonians': {'testament': 'new', 'position': 52},
    '2 Thessalonians': {'testament': 'new', 'position': 53},
    '1 Timothy': {'testament': 'new', 'position': 54},
    '2 Timothy': {'testament': 'new', 'position': 55},
    'Titus': {'testament': 'new', 'position': 56},
    'Philemon': {'testament': 'new', 'position': 57},
    'Hebrews': {'testament': 'new', 'position': 58},
    'James': {'testament': 'new', 'position': 59},
    '1 Peter': {'testament': 'new', 'position': 60},
    '2 Peter': {'testament': 'new', 'position': 61},
    '1 John': {'testament': 'new', 'position': 62},
    '2 John': {'testament': 'new', 'position': 63},
    '3 John': {'testament': 'new', 'position': 64},
    'Jude': {'testament': 'new', 'position': 65},
    'Revelation': {'testament': 'new', 'position': 66},
}

# Book abbreviations
BOOK_ABBR = {
    'Genesis': 'GEN',
    'Exodus': 'EXO',
    'Leviticus': 'LEV',
    'Numbers': 'NUM',
    'Deuteronomy': 'DEU',
    'Joshua': 'JOS',
    'Judges': 'JDG',
    'Ruth': 'RUT',
    '1 Samuel': '1SA',
    '2 Samuel': '2SA',
    '1 Kings': '1KI',
    '2 Kings': '2KI',
    '1 Chronicles': '1CH',
    '2 Chronicles': '2CH',
    'Ezra': 'EZR',
    'Nehemiah': 'NEH',
    'Esther': 'EST',
    'Job': 'JOB',
    'Psalms': 'PSA',
    'Proverbs': 'PRO',
    'Ecclesiastes': 'ECC',
    'Song of Solomon': 'SNG',
    'Isaiah': 'ISA',
    'Jeremiah': 'JER',
    'Lamentations': 'LAM',
    'Ezekiel': 'EZK',
    'Daniel': 'DAN',
    'Hosea': 'HOS',
    'Joel': 'JOL',
    'Amos': 'AMO',
    'Obadiah': 'OBA',
    'Jonah': 'JON',
    'Micah': 'MIC',
    'Nahum': 'NAH',
    'Habakkuk': 'HAB',
    'Zephaniah': 'ZEP',
    'Haggai': 'HAG',
    'Zechariah': 'ZEC',
    'Malachi': 'MAL',
    'Matthew': 'MAT',
    'Mark': 'MRK',
    'Luke': 'LUK',
    'John': 'JHN',
    'Acts': 'ACT',
    'Romans': 'ROM',
    '1 Corinthians': '1CO',
    '2 Corinthians': '2CO',
    'Galatians': 'GAL',
    'Ephesians': 'EPH',
    'Philippians': 'PHP',
    'Colossians': 'COL',
    '1 Thessalonians': '1TH',
    '2 Thessalonians': '2TH',
    '1 Timothy': '1TI',
    '2 Timothy': '2TI',
    'Titus': 'TIT',
    'Philemon': 'PHM',
    'Hebrews': 'HEB',
    'James': 'JAS',
    '1 Peter': '1PE',
    '2 Peter': '2PE',
    '1 John': '1JN',
    '2 John': '2JN',
    '3 John': '3JN',
    'Jude': 'JUD',
    'Revelation': 'REV',
}

def create_database_schema(conn):
    """
    Create the database schema for the Bible data.
    
    Args:
        conn: SQLite database connection
    """
    logger.info("Creating database schema...")
    cursor = conn.cursor()
    
    # Create Books table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        abbreviation TEXT NOT NULL,
        testament TEXT NOT NULL,
        position INTEGER NOT NULL,
        UNIQUE(name)
    )
    ''')
    
    # Create Chapters table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chapters (
        id INTEGER PRIMARY KEY,
        book_id INTEGER NOT NULL,
        number INTEGER NOT NULL,
        UNIQUE(book_id, number),
        FOREIGN KEY (book_id) REFERENCES books(id)
    )
    ''')
    
    # Create Verses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS verses (
        id INTEGER PRIMARY KEY,
        chapter_id INTEGER NOT NULL,
        number INTEGER NOT NULL,
        text TEXT NOT NULL,
        UNIQUE(chapter_id, number),
        FOREIGN KEY (chapter_id) REFERENCES chapters(id)
    )
    ''')
    
    # Create cross-references table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cross_references (
        id INTEGER PRIMARY KEY,
        source_verse_id INTEGER NOT NULL,
        target_verse_id INTEGER NOT NULL,
        strength REAL DEFAULT 1.0,
        UNIQUE(source_verse_id, target_verse_id),
        FOREIGN KEY (source_verse_id) REFERENCES verses(id),
        FOREIGN KEY (target_verse_id) REFERENCES verses(id)
    )
    ''')
    
    # Create indices for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_verses_chapter_id ON verses(chapter_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chapters_book_id ON chapters(book_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cross_refs_source ON cross_references(source_verse_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cross_refs_target ON cross_references(target_verse_id)')
    
    conn.commit()
    logger.info("Schema created successfully")

def insert_book(conn, name):
    """
    Insert a book into the database and return its ID.
    
    Args:
        conn: SQLite database connection
        name: Name of the book
        
    Returns:
        The ID of the inserted or existing book
    """
    cursor = conn.cursor()
    
    # Get book metadata
    meta = BOOK_META.get(name, {'testament': 'unknown', 'position': 999})
    abbr = BOOK_ABBR.get(name, name[:3].upper())
    
    # Insert book if it doesn't exist
    cursor.execute(
        '''
        INSERT OR IGNORE INTO books (name, abbreviation, testament, position)
        VALUES (?, ?, ?, ?)
        ''',
        (name, abbr, meta['testament'], meta['position'])
    )
    
    # Get book ID
    cursor.execute('SELECT id FROM books WHERE name = ?', (name,))
    book_id = cursor.fetchone()[0]
    
    return book_id

def insert_chapter(conn, book_id, chapter_num):
    """
    Insert a chapter into the database and return its ID.
    
    Args:
        conn: SQLite database connection
        book_id: ID of the book
        chapter_num: Chapter number
        
    Returns:
        The ID of the inserted or existing chapter
    """
    cursor = conn.cursor()
    
    # Insert chapter if it doesn't exist
    cursor.execute(
        '''
        INSERT OR IGNORE INTO chapters (book_id, number)
        VALUES (?, ?)
        ''',
        (book_id, chapter_num)
    )
    
    # Get chapter ID
    cursor.execute(
        'SELECT id FROM chapters WHERE book_id = ? AND number = ?',
        (book_id, chapter_num)
    )
    chapter_id = cursor.fetchone()[0]
    
    return chapter_id

def insert_verse(conn, chapter_id, verse_num, text):
    """
    Insert a verse into the database and return its ID.
    
    Args:
        conn: SQLite database connection
        chapter_id: ID of the chapter
        verse_num: Verse number
        text: Verse text
        
    Returns:
        The ID of the inserted verse
    """
    cursor = conn.cursor()
    
    # Insert verse
    cursor.execute(
        '''
        INSERT OR IGNORE INTO verses (chapter_id, number, text)
        VALUES (?, ?, ?)
        ''',
        (chapter_id, verse_num, text)
    )
    
    # Get verse ID
    cursor.execute(
        'SELECT id FROM verses WHERE chapter_id = ? AND number = ?',
        (chapter_id, verse_num)
    )
    verse_id = cursor.fetchone()[0]
    
    return verse_id

def process_bible_text(conn, text_file):
    """
    Process the Bible text file and insert data into the database.
    
    Args:
        conn: SQLite database connection
        text_file: Path to the Bible text file
    """
    logger.info(f"Processing Bible text file: {text_file}")
    
    # Stats for logging
    books_count = 0
    chapters_count = 0
    verses_count = 0
    
    # Dictionary to store book_name -> book_id mapping
    book_ids = {}
    
    # Dictionary to store (book_id, chapter_num) -> chapter_id mapping
    chapter_ids = {}
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse the line
            match = BOOK_PATTERN.match(line)
            if not match:
                logger.warning(f"Line doesn't match pattern: {line}")
                continue
                
            book = match.group('book').strip()
            chapter = int(match.group('chapter'))
            verse = int(match.group('verse'))
            text = match.group('text').strip()
            
            # Get or insert book
            if book not in book_ids:
                book_ids[book] = insert_book(conn, book)
                books_count += 1
                
            book_id = book_ids[book]
            
            # Get or insert chapter
            chapter_key = (book_id, chapter)
            if chapter_key not in chapter_ids:
                chapter_ids[chapter_key] = insert_chapter(conn, book_id, chapter)
                chapters_count += 1
                
            chapter_id = chapter_ids[chapter_key]
            
            # Insert verse
            insert_verse(conn, chapter_id, verse, text)
            verses_count += 1
            
            # Commit every 1000 verses to avoid large transactions
            if verses_count % 1000 == 0:
                conn.commit()
                logger.info(f"Processed {verses_count} verses...")
    
    # Final commit
    conn.commit()
    logger.info(f"Processed {books_count} books, {chapters_count} chapters, {verses_count} verses")

def process_cross_references(conn, cross_ref_file):
    """
    Process the cross-references file and insert into the database.
    
    Args:
        conn: SQLite database connection
        cross_ref_file: Path to the cross-references file
    """
    if not os.path.exists(cross_ref_file):
        logger.warning(f"Cross-reference file not found: {cross_ref_file}")
        return
        
    logger.info(f"Processing cross-references file: {cross_ref_file}")
    
    cursor = conn.cursor()
    refs_count = 0
    
    # Function to get verse_id from reference
    def get_verse_id(book, chapter, verse):
        cursor.execute(
            '''
            SELECT v.id
            FROM verses v
            JOIN chapters c ON v.chapter_id = c.id
            JOIN books b ON c.book_id = b.id
            WHERE b.name = ? AND c.number = ? AND v.number = ?
            ''',
            (book, chapter, verse)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    with open(cross_ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('|')
            if len(parts) < 2:
                continue
                
            # Parse source reference
            source_ref = parts[0].strip()
            source_match = BOOK_PATTERN.match(source_ref)
            if not source_match:
                continue
                
            source_book = source_match.group('book').strip()
            source_chapter = int(source_match.group('chapter'))
            source_verse = int(source_match.group('verse'))
            
            # Get source verse ID
            source_id = get_verse_id(source_book, source_chapter, source_verse)
            if not source_id:
                continue
                
            # Process target references
            for target_ref in parts[1:]:
                target_ref = target_ref.strip()
                target_match = BOOK_PATTERN.match(target_ref)
                if not target_match:
                    continue
                    
                target_book = target_match.group('book').strip()
                target_chapter = int(target_match.group('chapter'))
                target_verse = int(target_match.group('verse'))
                
                # Get target verse ID
                target_id = get_verse_id(target_book, target_chapter, target_verse)
                if not target_id:
                    continue
                    
                # Insert cross-reference
                cursor.execute(
                    '''
                    INSERT OR IGNORE INTO cross_references (source_verse_id, target_verse_id)
                    VALUES (?, ?)
                    ''',
                    (source_id, target_id)
                )
                
                refs_count += 1
                
                # Commit every 1000 references
                if refs_count % 1000 == 0:
                    conn.commit()
                    logger.info(f"Processed {refs_count} cross-references...")
    
    # Final commit
    conn.commit()
    logger.info(f"Processed {refs_count} cross-references")

def vacuum_database(conn):
    """
    Vacuum the database to optimize storage.
    
    Args:
        conn: SQLite database connection
    """
    logger.info("Vacuuming database...")
    conn.execute("VACUUM")
    logger.info("Database vacuumed")

def main():
    """Main function to convert Bible text to SQLite database."""
    parser = argparse.ArgumentParser(description='Convert Bible text file to SQLite database')
    parser.add_argument('--input', '-i', required=True, help='Path to the Bible text file')
    parser.add_argument('--output', '-o', required=True, help='Path for the output SQLite database')
    parser.add_argument('--cross-refs', '-c', help='Path to cross-references file (optional)')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_db = args.output
    cross_refs_file = args.cross_refs
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 1
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_db)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Connect to SQLite database
    logger.info(f"Connecting to database: {output_db}")
    conn = sqlite3.connect(output_db)
    
    try:
        # Create schema
        create_database_schema(conn)
        
        # Process Bible text
        process_bible_text(conn, input_file)
        
        # Process cross-references if provided
        if cross_refs_file:
            process_cross_references(conn, cross_refs_file)
            
        # Vacuum database
        vacuum_database(conn)
        
        logger.info(f"Conversion completed successfully. Database saved to: {output_db}")
        
    except Exception as e:
        logger.exception("Error during conversion: %s", str(e))
        return 1
        
    finally:
        conn.close()
        
    return 0

if __name__ == "__main__":
    exit(main())