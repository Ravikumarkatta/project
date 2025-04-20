import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class VerseReference:
    """Class representing a Bible verse reference."""

    def __init__(
        self, book: str, chapter: int, verse: int, end_verse: Optional[int] = None
    ):
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.end_verse = end_verse

    def __str__(self) -> str:
        if self.end_verse:
            return f"{self.book} {self.chapter}:{self.verse}-{self.end_verse}"
        return f"{self.book} {self.chapter}:{self.verse}"

    def to_dict(self) -> Dict:
        return {
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "end_verse": self.end_verse,
        }


class VerseResolver:
    """Service for resolving and validating Bible verse references."""

    def __init__(self, bible_data_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Regular expressions for verse detection
        self.verse_patterns = [
            # Standard format: Book Chapter:Verse
            r"(\d?\s*[A-Za-z]+)\s+(\d+):(\d+)(?:-(\d+))?",
            # Chapter and verse only (when book is known): Chapter:Verse
            r"(\d+):(\d+)(?:-(\d+))?",
            # Verse ranges: Book Chapter:Verse-Verse
            r"(\d?\s*[A-Za-z]+)\s+(\d+):(\d+)-(\d+)",
        ]
        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.verse_patterns
        ]

        # Load Bible data for validation
        self.bible_data = self._load_bible_data(bible_data_path)

        # Book name standardization mappings
        self.book_mappings = self._create_book_mappings()

    def _load_bible_data(self, bible_data_path: Optional[str] = None) -> Dict:
        """Load Bible data for validation."""
        if not bible_data_path:
            bible_data_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / "processed"
                / "kjv_processed.json"
            )

        try:
            with open(bible_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load Bible data: {e}")
            return {}

    def _create_book_mappings(self) -> Dict[str, str]:
        """Create mappings for standardizing book names."""
        return {
            # Old Testament
            "gen": "Genesis",
            "genesis": "Genesis",
            "ex": "Exodus",
            "exodus": "Exodus",
            "lev": "Leviticus",
            "leviticus": "Leviticus",
            # ... Add all book mappings
            # New Testament
            "matt": "Matthew",
            "matthew": "Matthew",
            "mk": "Mark",
            "mark": "Mark",
            "lk": "Luke",
            "luke": "Luke",
            "jn": "John",
            "john": "John",
            "acts": "Acts",
            "rom": "Romans",
            "romans": "Romans",
            # ... Add remaining books
        }

    def standardize_book_name(self, book: str) -> str:
        """Convert book name variants to standard form."""
        book = book.lower().strip()
        return self.book_mappings.get(book, book.title())

    def parse_reference(self, text: str) -> Optional[VerseReference]:
        """
        Parse a verse reference from text.

        Args:
            text: Text containing verse reference

        Returns:
            VerseReference object if valid reference found, None otherwise
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) == 4:  # Full reference with possible verse range
                    book = self.standardize_book_name(groups[0])
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    end_verse = int(groups[3]) if groups[3] else None
                    return VerseReference(book, chapter, verse, end_verse)
                elif len(groups) == 2:  # Chapter:Verse format
                    chapter = int(groups[0])
                    verse = int(groups[1])
                    return VerseReference(None, chapter, verse)
        return None

    def validate_reference(self, reference: VerseReference) -> bool:
        """
        Validate if a verse reference exists in the Bible.

        Args:
            reference: VerseReference object to validate

        Returns:
            True if reference is valid, False otherwise
        """
        if not reference.book or reference.book not in self.bible_data:
            return False

        book_data = self.bible_data[reference.book]
        chapter_str = str(reference.chapter)

        if chapter_str not in book_data:
            return False

        verse_str = str(reference.verse)
        if verse_str not in book_data[chapter_str]:
            return False

        if reference.end_verse:
            end_verse_str = str(reference.end_verse)
            return all(
                str(v) in book_data[chapter_str]
                for v in range(reference.verse, reference.end_verse + 1)
            )

        return True

    def resolve_references(self, text: str) -> List[Dict]:
        """
        Find and validate all verse references in text.

        Args:
            text: Input text to search for references

        Returns:
            List of validated verse reference dictionaries
        """
        references = []
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                ref = self.parse_reference(match.group())
                if ref and self.validate_reference(ref):
                    references.append(
                        {
                            "reference": str(ref),
                            "span": match.span(),
                            "details": ref.to_dict(),
                        }
                    )
        return references

    def get_verse_text(self, reference: Union[str, VerseReference]) -> Optional[str]:
        """
        Get the text of a Bible verse.

        Args:
            reference: Verse reference string or VerseReference object

        Returns:
            Verse text if found, None otherwise
        """
        if isinstance(reference, str):
            ref = self.parse_reference(reference)
            if not ref:
                return None
        else:
            ref = reference

        if not self.validate_reference(ref):
            return None

        verse_text = self.bible_data[ref.book][str(ref.chapter)][str(ref.verse)]

        if ref.end_verse:
            # Combine verse range
            verses = [
                self.bible_data[ref.book][str(ref.chapter)][str(v)]
                for v in range(ref.verse, ref.end_verse + 1)
            ]
            verse_text = " ".join(verses)

        return verse_text
