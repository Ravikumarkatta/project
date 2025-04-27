import sqlite3
import sys

db_path = 'data/processed/bible.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cross_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_translation TEXT,
            source_book TEXT,
            source_chapter INTEGER,
            source_verse INTEGER,
            target_translation TEXT,
            target_book TEXT,
            target_chapter INTEGER,
            target_verse INTEGER,
            FOREIGN KEY (source_translation, source_book, source_chapter, source_verse) REFERENCES bible_verses(translation, book, chapter, verse),
            FOREIGN KEY (target_translation, target_book, target_chapter, target_verse) REFERENCES bible_verses(translation, book, chapter, verse)
        )
    """)
    conn.commit()
    print("Cross-references table created or already exists.")
    conn.close()
except Exception as e:
    print(f"Error creating cross_references table: {e}", file=sys.stderr)
