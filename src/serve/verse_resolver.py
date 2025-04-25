# bible/src/serve/verse_resolver.py
"""
Resolves verse references using pre-processed Bible data.
Handles different translations and provides context.
"""

import json
import os
import re
# 3.1, 3.2: Import Path and Optional
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Assuming VerseReference is defined elsewhere, e.g., in bible_manager
# If not, define a placeholder or import correctly
try:
    from src.bible_manager.verse_reference import VerseReference
except ImportError:
    # Placeholder if the actual class isn't available here
    class VerseReference:
        def __init__(self, ref_string: str):
            self.book = "Unknown"
            self.chapter = 0
            self.start_verse = 0
            self.end_verse = None
            if ref_string: # Basic parsing for placeholder
                 parts = ref_string.split()
                 if parts: self.book = parts[0]
                 if len(parts) > 1 and ':' in parts[-1]:
                     try:
                         chap, verse = parts[-1].split(':')
                         self.chapter = int(chap)
                         if '-' in verse:
                             sv, ev = verse.split('-')
                             self.start_verse = int(sv)
                             self.end_verse = int(ev)
                         else:
                             self.start_verse = int(verse)
                     except ValueError:
                         pass # Ignore parsing errors for placeholder

# Assuming logger setup is handled elsewhere
import logging
logger = logging.getLogger(__name__)


class VerseResolver:
    """
    Resolves Bible verse references to their text content for various translations.
    """

    def __init__(self, config_path: Optional[str] = "config/resolver_config.json") -> None:
        """
        Initializes the VerseResolver.

        Args:
            config_path: Path to the configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.bible_data: Dict[str, Dict[str, Any]] = self._load_bible_data()
        self.default_translation: str = self.config.get("default_translation", "KJV") # Default to KJV if not specified

    # 3.3: Correct return type hint
    def _load_config(self, config_path_str: Optional[str]) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        # 3.1: Use Path consistently, handle Optional[Path]
        config_p: Optional[Path] = Path(config_path_str) if config_path_str else None
        default_conf = {
            "data_directory": "data/processed/bibles",
            "default_translation": "KJV",
            "context_verses": 2,
        }

        # 3.2: Guard against None before calling open()
        if config_p is None or not config_p.exists():
            logger.warning(
                f"Config file not found at {config_p}. Using default configuration."
            )
            return default_conf
        try:
            # 3.2: Use the Path object with open
            with open(config_p, "r", encoding="utf-8") as f:
                # 3.3: Ensure return matches the hint (json.load returns Dict[str, Any] typically)
                loaded_conf = json.load(f)
                # Merge defaults with loaded config (loaded overrides defaults)
                default_conf.update(loaded_conf)
                return default_conf
        except (json.JSONDecodeError, IOError, Exception) as e:
            logger.error(f"Error loading config from {config_p}: {e}. Using default.")
            return default_conf

    def _load_bible_data(self) -> Dict[str, Dict[str, Any]]:
        """Loads processed Bible data for available translations."""
        data_dir = Path(self.config.get("data_directory", "data/processed/bibles"))
        bible_data: Dict[str, Dict[str, Any]] = {}
        if not data_dir.exists() or not data_dir.is_dir():
            logger.error(f"Bible data directory not found or not a directory: {data_dir}")
            return bible_data

        logger.info(f"Loading Bible data from: {data_dir}")
        for file_path in data_dir.glob("*.json"):
            try:
                translation_id = file_path.stem # Use filename without extension as ID
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Basic validation: check if it looks like Bible data (e.g., has books)
                    if isinstance(data, dict) and len(data) > 0:
                         # Assume data structure is {book: {chapter: {verse: text}}}
                         bible_data[translation_id.upper()] = data
                         logger.info(f"Loaded translation: {translation_id.upper()}")
                    else:
                         logger.warning(f"Skipping invalid data file: {file_path}")
            except (json.JSONDecodeError, IOError, Exception) as e:
                logger.error(f"Error loading Bible data from {file_path}: {e}")

        if not bible_data:
             logger.warning(f"No valid Bible data loaded from {data_dir}.")

        return bible_data

    def get_available_translations(self) -> List[str]:
        """Returns a list of available translation IDs."""
        return list(self.bible_data.keys())

    def resolve(
        self, reference_str: str, translation: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolves a verse reference string to its text and context.

        Args:
            reference_str: The verse reference (e.g., "John 3:16", "Romans 8:28-30").
            translation: The translation ID (e.g., "KJV", "NIV"). Uses default if None.

        Returns:
            A dictionary containing the resolved verse(s) and context,
            or None if resolution fails.
            Example:
            {
                "reference": "John 3:16",
                "translation": "KJV",
                "verses": {16: "For God so loved the world..."},
                "context_before": {14: "...", 15: "..."},
                "context_after": {17: "...", 18: "..."}
            }
        """
        trans_id = (translation or self.default_translation).upper()
        if trans_id not in self.bible_data:
            logger.error(f"Translation '{trans_id}' not available.")
            return None

        # 3.4: Guard against None before calling VerseReference constructor
        if reference_str is None:
             logger.error("Verse reference string cannot be None.")
             return None

        try:
            # 3.4: Call constructor only with valid string
            vr = VerseReference(reference_str)
        except Exception as e: # Catch potential errors during VerseReference parsing
            logger.error(f"Failed to parse verse reference '{reference_str}': {e}")
            return None

        # Check if parsing was successful (basic check for placeholder)
        if vr.book == "Unknown" or vr.chapter == 0 or vr.start_verse == 0:
             logger.warning(f"Could not fully parse verse reference: {reference_str}")
             # Decide if partial parsing is acceptable or return None
             # return None # Stricter approach

        book_data = self.bible_data[trans_id].get(vr.book)
        if not book_data:
            logger.error(f"Book '{vr.book}' not found in translation '{trans_id}'.")
            return None

        chapter_data = book_data.get(str(vr.chapter)) # Chapters are often stored as string keys
        if not chapter_data:
            logger.error(
                f"Chapter '{vr.chapter}' not found for book '{vr.book}' in translation '{trans_id}'."
            )
            return None

        resolved_verses: Dict[int, str] = {}
        context_before: Dict[int, str] = {}
        context_after: Dict[int, str] = {}
        context_range = self.config.get("context_verses", 2)

        start = vr.start_verse
        # If end_verse is None, it's a single verse reference
        end = vr.end_verse if vr.end_verse is not None else start

        # Collect requested verses
        for v_num in range(start, end + 1):
            verse_text = chapter_data.get(str(v_num)) # Verses often stored as string keys
            if verse_text:
                resolved_verses[v_num] = verse_text
            else:
                logger.warning(
                    f"Verse {vr.book} {vr.chapter}:{v_num} not found in '{trans_id}'."
                )
                # Decide how to handle missing verses in a range (e.g., skip, return None)
                # return None # Stricter approach if any verse in range is missing

        if not resolved_verses:
             logger.error(f"Could not resolve any verses for {reference_str} in '{trans_id}'.")
             return None # Return None if the primary verse(s) couldn't be found

        # Collect context before
        for i in range(1, context_range + 1):
            v_num = start - i
            if v_num <= 0: break # Stop if we go before verse 1
            verse_text = chapter_data.get(str(v_num))
            if verse_text:
                context_before[v_num] = verse_text
            else:
                break # Stop context if a verse is missing

        # Collect context after
        for i in range(1, context_range + 1):
            v_num = end + i
            verse_text = chapter_data.get(str(v_num))
            if verse_text:
                context_after[v_num] = verse_text
            else:
                break # Stop context if verse doesn't exist

        return {
            "reference": reference_str,
            "translation": trans_id,
            "verses": resolved_verses,
            # Sort context dicts by verse number for predictable order
            "context_before": dict(sorted(context_before.items())),
            "context_after": dict(sorted(context_after.items())),
        }

    # 3.3: Add specific return type hint
    def get_verse_text(self, reference_str: str, translation: Optional[str] = None) -> Optional[str]:
         """
         Convenience method to get only the text of the specified verse(s).

         Args:
             reference_str: The verse reference string.
             translation: The translation ID.

         Returns:
             The combined text of the resolved verses, or None if resolution fails.
         """
         resolved_data = self.resolve(reference_str, translation)
         if resolved_data and resolved_data.get("verses"):
             # Join verses, sorted by number, with a space
             verse_texts = [
                 resolved_data["verses"][v_num]
                 for v_num in sorted(resolved_data["verses"])
             ]
             # 3.3: Ensure return matches hint (joining strings results in a string)
             return " ".join(verse_texts)
         return None

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
                / "kjv_structured1.json"
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
