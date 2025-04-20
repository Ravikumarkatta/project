import os
import sqlite3
import re
from pathlib import Path

def init_db(db_path):
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bible_verses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            translation TEXT,
            book TEXT,
            chapter INTEGER,
            verse INTEGER,
            text TEXT,
            UNIQUE(translation, book, chapter, verse)
        )
    """)
    conn.commit()
    return conn, cursor

def parse_kjv_text(file_path):
    """Parse KJV text file into structured format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip header until Genesis
    content = content[content.find("Genesis"):]
    
    # Regex for verse pattern (e.g. "1:1", "2:14", etc.)
    verse_pattern = re.compile(r'(\d+):(\d+)\s+(.*?)(?=\d+:\d+|\Z)', re.DOTALL)
    
    current_book = None
    verses = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a book title
        if line.isupper() or line.startswith('THE '):
            current_book = line.title().replace('The Book Of ', '').strip()
            continue
            
        # Find verses in the line
        matches = verse_pattern.finditer(line)
        for match in matches:
            chapter = int(match.group(1))
            verse = int(match.group(2))
            text = match.group(3).strip()
            
            if current_book and text:
                verses.append((current_book, chapter, verse, text))
                
    return verses

def load_bible_data(db_path, raw_dir):
    """Load Bible data into SQLite database."""
    conn, cursor = init_db(db_path)
    
    # Process KJV text
    kjv_path = os.path.join(raw_dir, 'bibles', 'kjv.txt')
    if os.path.exists(kjv_path):
        verses = parse_kjv_text(kjv_path)
        
        # Insert verses
        cursor.executemany("""
            INSERT OR REPLACE INTO bible_verses 
            (translation, book, chapter, verse, text)
            VALUES ('KJV', ?, ?, ?, ?)
        """, verses)
        
        conn.commit()
        print(f"Loaded {len(verses)} verses from KJV")
    else:
        print(f"KJV file not found at {kjv_path}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "bible.db"
    raw_dir = project_root / "data" / "raw"
    
    # Ensure database directory exists
    os.makedirs(db_path.parent, exist_ok=True)
    
    # Load data
    load_bible_data(str(db_path), str(raw_dir))