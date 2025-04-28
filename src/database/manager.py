#!/usr/bin/env python3
"""
Database manager for Bible data using SQLite.
This module handles all database operations for the Bible project.
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BibleDatabaseManager:
    """Manager class for Bible SQLite database operations."""
    
    def __init__(self, db_path: Union[str, Path] = None):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Default path
            db_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "bible.db"
        
        self.db_path = Path(db_path)
        
        # Ensure the database exists
        if not self.db_path.exists():
            logger.error(f"Database file not found: {self.db_path}")
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection as a context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
        try:
            yield conn
        finally:
            conn.close()
    
    def get_books(self) -> List[Dict[str, Any]]:
        """Get all Bible books in order."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, book_number, name, abbreviation, testament, position FROM books ORDER BY position"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_book_by_name(self, book_name: str) -> Optional[Dict[str, Any]]:
        """Get a book by its name."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, book_number, name, abbreviation, testament, position FROM books WHERE name = ?",
                (book_name,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def parse_reference(self, reference: str) -> Tuple[str, int, Optional[int]]:
        """Parse a Bible reference string.
        
        Args:
            reference: A string like "Genesis 1:1" or "Genesis 1"
            
        Returns:
            Tuple of (book_name, chapter_number, verse_number or None)
        """
        parts = reference.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid reference format: {reference}")
        
        # Handle multi-word book names
        chapter_verse = parts[-1]
        book_name = " ".join(parts[:-1])
        
        # Parse chapter and verse
        if ":" in chapter_verse:
            chapter, verse = chapter_verse.split(":")
            return book_name, int(chapter), int(verse)
        else:
            return book_name, int(chapter_verse), None
    
    def get_verse(self, reference: str) -> Optional[Dict[str, Any]]:
        """Get a verse by its reference (e.g., 'Genesis 1:1')."""
        try:
            book_name, chapter, verse = self.parse_reference(reference)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT v.id, v.verse_number, v.text, c.chapter_number, b.name as book_name
                    FROM verses v
                    JOIN chapters c ON v.chapter_id = c.id
                    JOIN books b ON c.book_id = b.id
                    WHERE b.name = ? AND c.chapter_number = ? AND v.verse_number = ?
                    """,
                    (book_name, chapter, verse)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except ValueError as e:
            logger.error(f"Error parsing reference: {e}")
            return None
    
    def get_chapter(self, book_name: str, chapter: int) -> List[Dict[str, Any]]:
        """Get all verses in a chapter."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT v.id, v.verse_number, v.text, c.chapter_number, b.name as book_name
                FROM verses v
                JOIN chapters c ON v.chapter_id = c.id
                JOIN books b ON c.book_id = b.id
                WHERE b.name = ? AND c.chapter_number = ?
                ORDER BY v.verse_number
                """,
                (book_name, chapter)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def search_verses(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for verses containing the query text.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching verses with their references
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT v.id, v.verse_number, v.text, c.chapter_number, b.name as book_name,
                       b.name || ' ' || c.chapter_number || ':' || v.verse_number as reference
                FROM verse_search vs
                JOIN verses v ON vs.verse_id = v.id
                JOIN chapters c ON v.chapter_id = c.id
                JOIN books b ON c.book_id = b.id
                WHERE vs.text MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_cross_references(self, reference: str) -> List[Dict[str, Any]]:
        """Get cross-references for a specific verse."""
        try:
            book_name, chapter, verse = self.parse_reference(reference)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        tv.id,
                        tv.verse_number,
                        tv.text,
                        tc.chapter_number,
                        tb.name as book_name,
                        tb.name || ' ' || tc.chapter_number || ':' || tv.verse_number as target_reference,
                        cr.relation_type
                    FROM cross_references cr
                    JOIN verses sv ON cr.source_verse_id = sv.id
                    JOIN chapters sc ON sv.chapter_id = sc.id
                    JOIN books sb ON sc.book_id = sb.id
                    JOIN verses tv ON cr.target_verse_id = tv.id
                    JOIN chapters tc ON tv.chapter_id = tc.id
                    JOIN books tb ON tc.book_id = tb.id
                    WHERE sb.name = ? AND sc.chapter_number = ? AND sv.verse_number = ?
                    """,
                    (book_name, chapter, verse)
                )
                return [dict(row) for row in cursor.fetchall()]
        except ValueError as e:
            logger.error(f"Error parsing reference: {e}")
            return []
    
    def get_verse_embedding(self, verse_id: int) -> Optional[List[float]]:
        """Get the embedding vector for a verse."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT embedding FROM verse_embeddings WHERE verse_id = ?",
                (verse_id,)
            )
            row = cursor.fetchone()
            if row and row['embedding']:
                return json.loads(row['embedding'])
            return None
    
    def get_commentary(self, reference: str, source_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get commentary entries for a verse reference."""
        try:
            book_name, chapter, verse = self.parse_reference(reference)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        ce.id,
                        ce.content,
                        cs.name as source_name,
                        cs.author,
                        cs.theological_tradition
                    FROM commentary_entries ce
                    JOIN commentary_sources cs ON ce.source_id = cs.id
                    JOIN verses v ON ce.verse_id = v.id
                    JOIN chapters c ON v.chapter_id = c.id
                    JOIN books b ON c.book_id = b.id
                    WHERE b.name = ? AND c.chapter_number = ?
                """
                params = [book_name, chapter]
                
                if verse is not None:
                    query += " AND v.verse_number = ?"
                    params.append(verse)
                
                if source_id is not None:
                    query += " AND cs.id = ?"
                    params.append(source_id)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except ValueError as e:
            logger.error(f"Error parsing reference: {e}")
            return []
    
    def get_lexicon_entries_for_verse(self, reference: str) -> List[Dict[str, Any]]:
        """Get lexicon entries for a verse."""
        try:
            book_name, chapter, verse = self.parse_reference(reference)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        le.id,
                        le.original_word,
                        le.language,
                        le.transliteration,
                        le.definition,
                        le.strong_number,
                        vlm.position
                    FROM verse_lexicon_mapping vlm
                    JOIN lexicon_entries le ON vlm.lexicon_id = le.id
                    JOIN verses v ON vlm.verse_id = v.id
                    JOIN chapters c ON v.chapter_id = c.id
                    JOIN books b ON c.book_id = b.id
                    WHERE b.name = ? AND c.chapter_number = ? AND v.verse_number = ?
                    ORDER BY vlm.position
                    """,
                    (book_name, chapter, verse)
                )
                return [dict(row) for row in cursor.fetchall()]
        except ValueError as e:
            logger.error(f"Error parsing reference: {e}")
            return []
    
    def get_verse_count(self) -> int:
        """Get the total number of verses in the Bible."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM verses")
            return cursor.fetchone()['count']
    
    def get_commentary_sources(self) -> List[Dict[str, Any]]:
        """Get all available commentary sources."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, author, year, description, theological_tradition
                FROM commentary_sources
                ORDER BY name
                """
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def add_commentary_entry(self, source_id: int, reference: str, content: str) -> int:
        """Add a new commentary entry."""
        try:
            book_name, chapter, verse = self.parse_reference(reference)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get the verse ID
                verse_id = None
                book_id = None
                chapter_id = None
                
                if verse is not None:
                    cursor.execute(
                        """
                        SELECT v.id
                        FROM verses v
                        JOIN chapters c ON v.chapter_id = c.id
                        JOIN books b ON c.book_id = b.id
                        WHERE b.name = ? AND c.chapter_number = ? AND v.verse_number = ?
                        """,
                        (book_name, chapter, verse)
                    )
                    row = cursor.fetchone()
                    if row:
                        verse_id = row['id']
                
                # If no specific verse, get chapter ID
                if verse_id is None:
                    cursor.execute(
                        """
                        SELECT c.id
                        FROM chapters c
                        JOIN books b ON c.book_id = b.id
                        WHERE b.name = ? AND c.chapter_number = ?
                        """,
                        (book_name, chapter)
                    )
                    row = cursor.fetchone()
                    if row:
                        chapter_id = row['id']
                
                # If no specific chapter, get book ID
                if chapter_id is None:
                    cursor.execute(
                        "SELECT id FROM books WHERE name = ?",
                        (book_name,)
                    )
                    row = cursor.fetchone()
                    if row:
                        book_id = row['id']
                
                # Insert the commentary entry
                cursor.execute(
                    """
                    INSERT INTO commentary_entries 
                    (source_id, verse_id, chapter_id, book_id, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source_id, verse_id, chapter_id, book_id, content)
                )
                entry_id = cursor.lastrowid
                conn.commit()
                return entry_id
        except ValueError as e:
            logger.error(f"Error parsing reference: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a custom SQL query for advanced operations.
        
        Args:
            query: SQL query string
            params: Query parameters as a tuple
            
        Returns:
            List of result rows as dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return [{'rowcount': cursor.rowcount, 'lastrowid': cursor.lastrowid}]

# Singleton instance for global use
_instance = None

def get_db_manager(db_path: Union[str, Path] = None) -> BibleDatabaseManager:
    """Get the singleton instance of the database manager."""
    global _instance
    if _instance is None:
        _instance = BibleDatabaseManager(db_path)
    return _instance