import re
from collections import defaultdict
from typing import Dict, List, Optional
from pydantic import BaseModel


class VerseReference(BaseModel):
    book: str
    chapter: int
    verse: Optional[int] = None
    end_verse: Optional[int] = None
    end_chapter: Optional[int] = None


class VerseReferenceDetector:
    def __init__(self) -> None:
        # Full names to canonical names
        self.book_names: Dict[str, str] = {
            "genesis": "Genesis", "exodus": "Exodus", "leviticus": "Leviticus", 
            "numbers": "Numbers", "deuteronomy": "Deuteronomy", "joshua": "Joshua",
            "judges": "Judges", "ruth": "Ruth", "1 samuel": "1 Samuel", 
            "2 samuel": "2 Samuel", "1 kings": "1 Kings", "2 kings": "2 Kings", 
            "1 chronicles": "1 Chronicles", "2 chronicles": "2 Chronicles", 
            "ezra": "Ezra", "nehemiah": "Nehemiah", "esther": "Esther", "job": "Job",
            "psalms": "Psalms", "psalm": "Psalms", "proverbs": "Proverbs", 
            "ecclesiastes": "Ecclesiastes", "song of solomon": "Song of Solomon", 
            "isaiah": "Isaiah", "jeremiah": "Jeremiah", "lamentations": "Lamentations",
            "ezekiel": "Ezekiel", "daniel": "Daniel", "hosea": "Hosea", "joel": "Joel",
            "amos": "Amos", "obadiah": "Obadiah", "jonah": "Jonah", "micah": "Micah", 
            "nahum": "Nahum", "habakkuk": "Habakkuk", "zephaniah": "Zephaniah", 
            "haggai": "Haggai", "zechariah": "Zechariah", "malachi": "Malachi", 
            "matthew": "Matthew", "mark": "Mark", "luke": "Luke", "john": "John", 
            "acts": "Acts", "romans": "Romans", "1 corinthians": "1 Corinthians", 
            "2 corinthians": "2 Corinthians", "galatians": "Galatians", 
            "ephesians": "Ephesians", "philippians": "Philippians", 
            "colossians": "Colossians", "1 thessalonians": "1 Thessalonians", 
            "2 thessalonians": "2 Thessalonians", "1 timothy": "1 Timothy", 
            "2 timothy": "2 Timothy", "titus": "Titus", "philemon": "Philemon", 
            "hebrews": "Hebrews", "james": "James", "1 peter": "1 Peter", 
            "2 peter": "2 Peter", "1 john": "1 John", "2 john": "2 John", 
            "3 john": "3 John", "jude": "Jude", "revelation": "Revelation"
        }
        
        # Add common abbreviations to the book_names dict
        self.abbreviations = {
            "gen": "Genesis", "exod": "Exodus", "ex": "Exodus", "lev": "Leviticus",
            "num": "Numbers", "deut": "Deuteronomy", "josh": "Joshua", "judg": "Judges",
            "1 sam": "1 Samuel", "2 sam": "2 Samuel", "1 kgs": "1 Kings", "2 kgs": "2 Kings",
            "1 chr": "1 Chronicles", "2 chr": "2 Chronicles", "ps": "Psalms", 
            "prov": "Proverbs", "eccl": "Ecclesiastes", "song": "Song of Solomon", 
            "isa": "Isaiah", "jer": "Jeremiah", "lam": "Lamentations", "ezek": "Ezekiel",
            "dan": "Daniel", "hos": "Hosea", "zech": "Zechariah", "mal": "Malachi",
            "matt": "Matthew", "mk": "Mark", "lk": "Luke", "jn": "John", "rom": "Romans",
            "1 cor": "1 Corinthians", "2 cor": "2 Corinthians", "gal": "Galatians",
            "eph": "Ephesians", "phil": "Philippians", "col": "Colossians",
            "1 thess": "1 Thessalonians", "2 thess": "2 Thessalonians",
            "1 tim": "1 Timothy", "2 tim": "2 Timothy", "heb": "Hebrews", "jas": "James",
            "1 pet": "1 Peter", "2 pet": "2 Peter", "1 jn": "1 John", "2 jn": "2 John",
            "3 jn": "3 John", "rev": "Revelation"
        }
        
        # Merge abbreviations into book_names
        self.book_names.update(self.abbreviations)

        self.patterns = [
            # Genesis 1:1-2:3 (cross-chapter range, allowing : or .) - most specific
            rf"\b({self._book_pattern()})\s+(\d+)[:.](\d+)-(\d+)[:.](\d+)\b",
            # Genesis 1:1-3 (same chapter range, allowing : or .)
            rf"\b({self._book_pattern()})\s+(\d+)[:.](\d+)-(\d+)\b",
            # Genesis 1:1 (single verse, allowing : or .)
            rf"\b({self._book_pattern()})\s+(\d+)[:.](\d+)\b",
            # Genesis 1 (whole chapter, ensuring no : or . follows the chapter number) - least specific
            rf"\b({self._book_pattern()})\s+(\d+)\b(?![:.])"
        ]
        # Sort patterns by length (most specific first) to ensure proper matching
        self.patterns.sort(key=lambda x: len(x), reverse=True)

        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]
        
        # Bible has valid chapter and verse ranges
        # This is a simplified version - you might want to expand with actual limits
        self.valid_ranges = {
            # Mapping of book name to (max_chapter, max_verse_per_chapter)
            # Using a default of (150, 200) for books not explicitly listed
            "default": (150, 200)
        }

    def _book_pattern(self) -> str:
        """Generates the regex pattern for matching book names."""
        return "|".join(map(re.escape, self.book_names.keys()))

    def is_valid_reference(self, book: str, chapter: int, verse: Optional[int] = None) -> bool:
        """
        Check if a verse reference is valid based on Bible structure.
        
        Args:
            book: The book name
            chapter: Chapter number
            verse: Optional verse number
            
        Returns:
            bool: True if the reference is valid, False otherwise
        """
        # Basic validation - chapters and verses should be positive
        if chapter <= 0:
            return False
            
        if verse is not None and verse <= 0:
            return False
            
        # More comprehensive validation could be added here with actual
        # chapter and verse limits for each book
        return True

    def detect_references(self, text: str) -> List[VerseReference]:
        """
        Detect all Bible verse references in the given text.
        Returns a list of VerseReference objects.
        """
        references = []
        seen_spans = set()

        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Skip if we've already processed this span (to avoid duplicate matches)
                span = match.span()
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                groups = match.groups()
                book_name = groups[0].lower()
                
                # Skip if book name is not recognized
                if book_name not in self.book_names:
                    continue
                
                # Normalize book name
                book = self.book_names[book_name]
                
                # Parse the reference based on the pattern
                if len(groups) == 2:  # Whole chapter
                    chapter = int(groups[1])
                    if self.is_valid_reference(book, chapter):
                        references.append(VerseReference(book=book, chapter=chapter))
                
                elif len(groups) == 3:  # Single verse
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    if self.is_valid_reference(book, chapter, verse):
                        references.append(
                            VerseReference(book=book, chapter=chapter, verse=verse)
                        )
                
                elif len(groups) == 4:  # Verse range in same chapter
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    end_verse = int(groups[3])
                    if (self.is_valid_reference(book, chapter, verse) and 
                            self.is_valid_reference(book, chapter, end_verse)):
                        references.append(
                            VerseReference(
                                book=book,
                                chapter=chapter,
                                verse=verse,
                                end_verse=end_verse,
                            )
                        )
                
                elif len(groups) == 5:  # Range across chapters
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    end_chapter = int(groups[3])
                    end_verse = int(groups[4])
                    if (self.is_valid_reference(book, chapter, verse) and 
                            self.is_valid_reference(book, end_chapter, end_verse)):
                        references.append(
                            VerseReference(
                                book=book,
                                chapter=chapter,
                                verse=verse,
                                end_chapter=end_chapter,
                                end_verse=end_verse,
                            )
                        )

        return references

    def normalize_reference(self, reference: str) -> Optional[VerseReference]:
        """
        Convert a string reference to a VerseReference object.
        Returns None if the reference is invalid.
        """
        for pattern in self.compiled_patterns:
            match = pattern.match(reference)
            if match:
                groups = match.groups()
                book_name = groups[0].lower()
                
                # Skip if book name is not recognized
                if book_name not in self.book_names:
                    return None
                    
                book = self.book_names[book_name]

                if len(groups) == 2:
                    chapter = int(groups[1])
                    if not self.is_valid_reference(book, chapter):
                        return None
                    return VerseReference(book=book, chapter=chapter)
                
                elif len(groups) == 3:
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    if not self.is_valid_reference(book, chapter, verse):
                        return None
                    return VerseReference(
                        book=book, chapter=chapter, verse=verse
                    )
                
                elif len(groups) == 4:
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    end_verse = int(groups[3])
                    if not (self.is_valid_reference(book, chapter, verse) and 
                            self.is_valid_reference(book, chapter, end_verse)):
                        return None
                    return VerseReference(
                        book=book,
                        chapter=chapter,
                        verse=verse,
                        end_verse=end_verse,
                    )
                
                elif len(groups) == 5:
                    chapter = int(groups[1])
                    verse = int(groups[2])
                    end_chapter = int(groups[3])
                    end_verse = int(groups[4])
                    if not (self.is_valid_reference(book, chapter, verse) and 
                            self.is_valid_reference(book, end_chapter, end_verse)):
                        return None
                    return VerseReference(
                        book=book,
                        chapter=chapter,
                        verse=verse,
                        end_chapter=end_chapter,
                        end_verse=end_verse,
                    )
        return None

    def format_reference(self, ref: VerseReference) -> str:
        """
        Convert a VerseReference object to a standardized string format.
        """
        if ref.verse is None:
            return f"{ref.book} {ref.chapter}"
        elif ref.end_verse and not ref.end_chapter:
            return f"{ref.book} {ref.chapter}:{ref.verse}-{ref.end_verse}"
        elif ref.end_chapter and ref.end_verse:
            return f"{ref.book} {ref.chapter}:{ref.verse}-{ref.end_chapter}:{ref.end_verse}"
        else:
            return f"{ref.book} {ref.chapter}:{ref.verse}"
