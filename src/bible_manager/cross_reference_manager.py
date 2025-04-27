import sqlite3
from typing import List, Optional

from src.bible_manager.verse_reference import VerseReference

class CrossReferenceManager:
    def __init__(self, db_path: str = 'data/processed/bible.db'):
        self.db_path = db_path

    def _connect(self):
        """Establishes a database connection."""
        return sqlite3.connect(self.db_path)

    def get_cross_references(self, source_ref: VerseReference) -> List[VerseReference]:
        """
        Retrieves cross-references for a given source verse.

        Args:
            source_ref: The VerseReference object for the source verse.

        Returns:
            A list of VerseReference objects that are cross-references to the source verse.
        """
        cross_refs = []
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT target_book, target_chapter, target_verse
                FROM cross_references
                WHERE source_book = ? AND source_chapter = ? AND source_verse = ?
                -- Assuming cross-references are primarily within the same translation for now,
                -- but the table design supports cross-translation references.
                -- WHERE source_translation = ? AND source_book = ? AND source_chapter = ? AND source_verse = ?
            """, (source_ref.book, source_ref.chapter, source_ref.verse)) # Add source_ref.translation if filtering by translation

            rows = cursor.fetchall()
            for row in rows:
                # Note: The current table schema doesn't store end_verse/end_chapter for targets.
                # If cross-reference data includes ranges, the schema and retrieval need adjustment.
                cross_refs.append(VerseReference(book=row[0], chapter=row[1], verse=row[2]))

        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if conn:
                conn.close()

        return cross_refs

    def insert_cross_reference(self, source_ref: VerseReference, target_ref: VerseReference):
        """
        Inserts a single cross-reference link into the database.

        Args:
            source_ref: The VerseReference object for the source verse.
            target_ref: The VerseReference object for the target verse.
        """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cross_references (
                    source_translation, source_book, source_chapter, source_verse,
                    target_translation, target_book, target_chapter, target_verse
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                # Assuming translation info is available in VerseReference or handled elsewhere
                # For now, using placeholder or assuming same translation if not critical
                "N/A", # Placeholder for source_translation
                source_ref.book,
                source_ref.chapter,
                source_ref.verse,
                "N/A", # Placeholder for target_translation
                target_ref.book,
                target_ref.chapter,
                target_ref.verse
            ))
            conn.commit()
            # print(f"Inserted cross-reference: {source_ref.book} {source_ref.chapter}:{source_ref.verse} -> {target_ref.book} {target_ref.chapter}:{target_ref.verse}") # Optional logging
        except sqlite3.Error as e:
            print(f"Database error inserting cross-reference: {e}")
            if conn:
                conn.rollback()
        except Exception as e:
            print(f"An error occurred inserting cross-reference: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    # A method to insert cross-references will be needed later for data loading
    # def insert_cross_references(self, source_ref: VerseReference, target_refs: List[VerseReference]):
    #     pass
