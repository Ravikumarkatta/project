import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel

class VerseReference(BaseModel):
    book: str
    chapter: int
    verse: Optional[int] = None
    end_verse: Optional[int] = None
    end_chapter: Optional[int] = None

class VerseReferenceDetector:
    def __init__(self):
        self.book_names = {
            # Standard names
            "genesis": "Genesis", "exodus": "Exodus", "leviticus": "Leviticus",
            "numbers": "Numbers", "deuteronomy": "Deuteronomy", "joshua": "Joshua",
            "judges": "Judges", "ruth": "Ruth", "1 samuel": "1 Samuel",
            "2 samuel": "2 Samuel", "1 kings": "1 Kings", "2 kings": "2 Kings",
            "1 chronicles": "1 Chronicles", "2 chronicles": "2 Chronicles",
            "ezra": "Ezra", "nehemiah": "Nehemiah", "esther": "Esther",
            "job": "Job", "psalms": "Psalms", "psalm": "Psalms",
            "proverbs": "Proverbs", "ecclesiastes": "Ecclesiastes",
            "song of solomon": "Song of Solomon", "isaiah": "Isaiah",
            "jeremiah": "Jeremiah", "lamentations": "Lamentations",
            "ezekiel": "Ezekiel", "daniel": "Daniel", "hosea": "Hosea",
            "joel": "Joel", "amos": "Amos", "obadiah": "Obadiah",
            "jonah": "Jonah", "micah": "Micah", "nahum": "Nahum",
            "habakkuk": "Habakkuk", "zephaniah": "Zephaniah",
            "haggai": "Haggai", "zechariah": "Zechariah", "malachi": "Malachi",
            "matthew": "Matthew", "mark": "Mark", "luke": "Luke",
            "john": "John", "acts": "Acts", "romans": "Romans",
            "1 corinthians": "1 Corinthians", "2 corinthians": "2 Corinthians",
            "galatians": "Galatians", "ephesians": "Ephesians",
            "philippians": "Philippians", "colossians": "Colossians",
            "1 thessalonians": "1 Thessalonians", "2 thessalonians": "2 Thessalonians",
            "1 timothy": "1 Timothy", "2 timothy": "2 Timothy",
            "titus": "Titus", "philemon": "Philemon", "hebrews": "Hebrews",
            "james": "James", "1 peter": "1 Peter", "2 peter": "2 Peter",
            "1 john": "1 John", "2 john": "2 John", "3 john": "3 John",
            "jude": "Jude", "revelation": "Revelation",
            # Common abbreviations
            "gen": "Genesis", "exo": "Exodus", "lev": "Leviticus",
            "num": "Numbers", "deut": "Deuteronomy", "josh": "Joshua",
            "judg": "Judges", "1 sam": "1 Samuel", "2 sam": "2 Samuel",
            "1 kgs": "1 Kings", "2 kgs": "2 Kings", "1 chr": "1 Chronicles",
            "2 chr": "2 Chronicles", "psa": "Psalms", "ps": "Psalms",
            "prov": "Proverbs", "eccl": "Ecclesiastes", "song": "Song of Solomon",
            "isa": "Isaiah", "jer": "Jeremiah", "lam": "Lamentations",
            "ezek": "Ezekiel", "dan": "Daniel", "hos": "Hosea",
            "matt": "Matthew", "mk": "Mark", "lk": "Luke",
            "jn": "John", "rom": "Romans", "1 cor": "1 Corinthians",
            "2 cor": "2 Corinthians", "gal": "Galatians", "eph": "Ephesians",
            "phil": "Philippians", "col": "Colossians",
            "1 thess": "1 Thessalonians", "2 thess": "2 Thessalonians",
            "1 tim": "1 Timothy", "2 tim": "2 Timothy", "tit": "Titus",
            "phlm": "Philemon", "heb": "Hebrews", "jas": "James",
            "1 pet": "1 Peter", "2 pet": "2 Peter", "1 jn": "1 John",
            "2 jn": "2 John", "3 jn": "3 John", "rev": "Revelation"
        }
        
        # Create regex pattern for book names
        book_pattern = "|".join(map(re.escape, self.book_names.keys()))
        
        # Main regex patterns for different verse reference formats
        self.patterns = [
            # Pattern for range across chapters (e.g., Genesis 1:1-2:3)
            rf"({book_pattern})\s*(\d+):(\d+)-(\d+):(\d+)",
            # Pattern for verse range within same chapter (e.g., Genesis 1:1-3)
            rf"({book_pattern})\s*(\d+):(\d+)-(\d+)",
            # Pattern for single verse (e.g., Genesis 1:1)
            rf"({book_pattern})\s*(\d+):(\d+)",
            # Pattern for whole chapter (e.g., Genesis 1)
            rf"({book_pattern})\s*(\d+)"
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]

    def detect_references(self, text: str) -> List[VerseReference]:
        """
        Detect all Bible verse references in the given text.
        Returns a list of VerseReference objects.
        """
        references = []
        text = text.lower()
        
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()
                
                # Normalize book name
                book = self.book_names[groups[0].lower()]
                
                if len(groups) == 2:  # Whole chapter
                    references.append(VerseReference(
                        book=book,
                        chapter=int(groups[1])
                    ))
                elif len(groups) == 3:  # Single verse
                    references.append(VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2])
                    ))
                elif len(groups) == 4:  # Verse range in same chapter
                    references.append(VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_verse=int(groups[3])
                    ))
                elif len(groups) == 5:  # Range across chapters
                    references.append(VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_chapter=int(groups[3]),
                        end_verse=int(groups[4])
                    ))
        
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
                book = self.book_names[groups[0].lower()]
                
                if len(groups) == 2:
                    return VerseReference(book=book, chapter=int(groups[1]))
                elif len(groups) == 3:
                    return VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2])
                    )
                elif len(groups) == 4:
                    return VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_verse=int(groups[3])
                    )
                elif len(groups) == 5:
                    return VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_chapter=int(groups[3]),
                        end_verse=int(groups[4])
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