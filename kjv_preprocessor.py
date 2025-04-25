#!/usr/bin/env python3
"""
Robust KJV Bible Text Preprocessor

This script processes raw Gutenberg KJV Bible text files into a standardized format
suitable for analysis, searching, and other applications. It handles:
- Gutenberg header/footer removal
- Standardized book title formatting
- Verse reference normalization
- Chapter handling
- Special formatting for Psalms
- Error detection and reporting for malformed verses
- Support for multiple output formats

Usage:
    python kjv_preprocessor.py --input raw_kjv.txt --output preprocessed_kjv.txt [OPTIONS]

Options:
    --format {text,json,csv}   Output format (default: text)
    --include-apocrypha        Include apocryphal books if present
    --fix-line-breaks          Attempt to fix irregular line breaks
    --validate                 Validate the structure of the output
    --verbose                  Enable verbose logging
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Constants
BIBLE_BOOKS = {
    # Old Testament - 39 books
    "Genesis": {"testament": "old", "abbrev": "Gen"},
    "Exodus": {"testament": "old", "abbrev": "Exod"},
    "Leviticus": {"testament": "old", "abbrev": "Lev"},
    "Numbers": {"testament": "old", "abbrev": "Num"},
    "Deuteronomy": {"testament": "old", "abbrev": "Deut"},
    "Joshua": {"testament": "old", "abbrev": "Josh"},
    "Judges": {"testament": "old", "abbrev": "Judg"},
    "Ruth": {"testament": "old", "abbrev": "Ruth"},
    "1 Samuel": {"testament": "old", "abbrev": "1Sam"},
    "2 Samuel": {"testament": "old", "abbrev": "2Sam"},
    "1 Kings": {"testament": "old", "abbrev": "1Kgs"},
    "2 Kings": {"testament": "old", "abbrev": "2Kgs"},
    "1 Chronicles": {"testament": "old", "abbrev": "1Chr"},
    "2 Chronicles": {"testament": "old", "abbrev": "2Chr"},
    "Ezra": {"testament": "old", "abbrev": "Ezra"},
    "Nehemiah": {"testament": "old", "abbrev": "Neh"},
    "Esther": {"testament": "old", "abbrev": "Esth"},
    "Job": {"testament": "old", "abbrev": "Job"},
    "Psalms": {"testament": "old", "abbrev": "Ps"},
    "Proverbs": {"testament": "old", "abbrev": "Prov"},
    "Ecclesiastes": {"testament": "old", "abbrev": "Eccl"},
    "Song of Solomon": {"testament": "old", "abbrev": "Song"},
    "Isaiah": {"testament": "old", "abbrev": "Isa"},
    "Jeremiah": {"testament": "old", "abbrev": "Jer"},
    "Lamentations": {"testament": "old", "abbrev": "Lam"},
    "Ezekiel": {"testament": "old", "abbrev": "Ezek"},
    "Daniel": {"testament": "old", "abbrev": "Dan"},
    "Hosea": {"testament": "old", "abbrev": "Hos"},
    "Joel": {"testament": "old", "abbrev": "Joel"},
    "Amos": {"testament": "old", "abbrev": "Amos"},
    "Obadiah": {"testament": "old", "abbrev": "Obad"},
    "Jonah": {"testament": "old", "abbrev": "Jonah"},
    "Micah": {"testament": "old", "abbrev": "Mic"},
    "Nahum": {"testament": "old", "abbrev": "Nah"},
    "Habakkuk": {"testament": "old", "abbrev": "Hab"},
    "Zephaniah": {"testament": "old", "abbrev": "Zeph"},
    "Haggai": {"testament": "old", "abbrev": "Hag"},
    "Zechariah": {"testament": "old", "abbrev": "Zech"},
    "Malachi": {"testament": "old", "abbrev": "Mal"},
    # New Testament - 27 books
    "Matthew": {"testament": "new", "abbrev": "Matt"},
    "Mark": {"testament": "new", "abbrev": "Mark"},
    "Luke": {"testament": "new", "abbrev": "Luke"},
    "John": {"testament": "new", "abbrev": "John"},
    "Acts": {"testament": "new", "abbrev": "Acts"},
    "Romans": {"testament": "new", "abbrev": "Rom"},
    "1 Corinthians": {"testament": "new", "abbrev": "1Cor"},
    "2 Corinthians": {"testament": "new", "abbrev": "2Cor"},
    "Galatians": {"testament": "new", "abbrev": "Gal"},
    "Ephesians": {"testament": "new", "abbrev": "Eph"},
    "Philippians": {"testament": "new", "abbrev": "Phil"},
    "Colossians": {"testament": "new", "abbrev": "Col"},
    "1 Thessalonians": {"testament": "new", "abbrev": "1Thess"},
    "2 Thessalonians": {"testament": "new", "abbrev": "2Thess"},
    "1 Timothy": {"testament": "new", "abbrev": "1Tim"},
    "2 Timothy": {"testament": "new", "abbrev": "2Tim"},
    "Titus": {"testament": "new", "abbrev": "Titus"},
    "Philemon": {"testament": "new", "abbrev": "Phlm"},
    "Hebrews": {"testament": "new", "abbrev": "Heb"},
    "James": {"testament": "new", "abbrev": "Jas"},
    "1 Peter": {"testament": "new", "abbrev": "1Pet"},
    "2 Peter": {"testament": "new", "abbrev": "2Pet"},
    "1 John": {"testament": "new", "abbrev": "1John"},
    "2 John": {"testament": "new", "abbrev": "2John"},
    "3 John": {"testament": "new", "abbrev": "3John"},
    "Jude": {"testament": "new", "abbrev": "Jude"},
    "Revelation": {"testament": "new", "abbrev": "Rev"},
    # Apocrypha - 14 books (included for completeness)
    "Tobit": {"testament": "apocrypha", "abbrev": "Tob"},
    "Judith": {"testament": "apocrypha", "abbrev": "Jdt"},
    "Wisdom of Solomon": {"testament": "apocrypha", "abbrev": "Wis"},
    "Sirach": {"testament": "apocrypha", "abbrev": "Sir"},
    "Baruch": {"testament": "apocrypha", "abbrev": "Bar"},
    "Letter of Jeremiah": {"testament": "apocrypha", "abbrev": "LJe"},
    "1 Maccabees": {"testament": "apocrypha", "abbrev": "1Macc"},
    "2 Maccabees": {"testament": "apocrypha", "abbrev": "2Macc"},
    "1 Esdras": {"testament": "apocrypha", "abbrev": "1Esd"},
    "Prayer of Manasseh": {"testament": "apocrypha", "abbrev": "PrMan"},
    "Psalm 151": {"testament": "apocrypha", "abbrev": "Ps151"},
    "3 Maccabees": {"testament": "apocrypha", "abbrev": "3Macc"},
    "2 Esdras": {"testament": "apocrypha", "abbrev": "2Esd"},
    "4 Maccabees": {"testament": "apocrypha", "abbrev": "4Macc"},
}

# Alternative names and spellings for books
BOOK_ALIASES = {
    # Common alternative names
    "Psalm": "Psalms",
    "Psalter": "Psalms",
    "Songs": "Song of Solomon",
    "Canticles": "Song of Solomon",
    "Song of Songs": "Song of Solomon",
    "Apocalypse": "Revelation",
    "The Revelation": "Revelation",
    "Saint Matthew": "Matthew",
    "Saint Mark": "Mark",
    "Saint Luke": "Luke",
    "Saint John": "John",
    "The Acts": "Acts",
    "First Samuel": "1 Samuel",
    "Second Samuel": "2 Samuel",
    "First Kings": "1 Kings",
    "Second Kings": "2 Kings",
    "First Chronicles": "1 Chronicles",
    "Second Chronicles": "2 Chronicles",
    "First Corinthians": "1 Corinthians",
    "Second Corinthians": "2 Corinthians",
    "First Thessalonians": "1 Thessalonians",
    "Second Thessalonians": "2 Thessalonians",
    "First Timothy": "1 Timothy",
    "Second Timothy": "2 Timothy",
    "First Peter": "1 Peter",
    "Second Peter": "2 Peter",
    "First John": "1 John",
    "Second John": "2 John",
    "Third John": "3 John",
    "First Book of Moses": "Genesis",
    "Second Book of Moses": "Exodus",
    "Third Book of Moses": "Leviticus",
    "Fourth Book of Moses": "Numbers",
    "Fifth Book of Moses": "Deuteronomy",
}


# Statistics and validation structure
class BibleStatistics:
    def __init__(self):
        self.books_found: Set[str] = set()
        self.chapters_per_book: Dict[str, int] = {}
        self.verses_per_chapter: Dict[str, Dict[int, int]] = {}
        self.total_verses: int = 0
        self.malformed_verses: List[Dict[str, Any]] = []

    def add_book(self, book_name: str) -> None:
        self.books_found.add(book_name)
        if book_name not in self.chapters_per_book:
            self.chapters_per_book[book_name] = 0
            self.verses_per_chapter[book_name] = {}

    def add_chapter(self, book_name: str, chapter_num: int) -> None:
        if book_name in self.chapters_per_book:
            if chapter_num > self.chapters_per_book[book_name]:
                self.chapters_per_book[book_name] = chapter_num

        if book_name not in self.verses_per_chapter:
            self.verses_per_chapter[book_name] = {}

        if chapter_num not in self.verses_per_chapter[book_name]:
            self.verses_per_chapter[book_name][chapter_num] = 0

    def add_verse(self, book_name: str, chapter_num: int, verse_num: int) -> None:
        self.total_verses += 1

        if book_name not in self.verses_per_chapter:
            self.verses_per_chapter[book_name] = {}

        if chapter_num not in self.verses_per_chapter[book_name]:
            self.verses_per_chapter[book_name][chapter_num] = 0

        if verse_num > self.verses_per_chapter[book_name][chapter_num]:
            self.verses_per_chapter[book_name][chapter_num] = verse_num

    def add_malformed_verse(self, line: str, reason: str) -> None:
        self.malformed_verses.append({"line": line, "reason": reason})

    def generate_report(self) -> Dict[str, Any]:
        return {
            "total_books": len(self.books_found),
            "books_found": sorted(list(self.books_found)),
            "total_verses": self.total_verses,
            "chapters_per_book": self.chapters_per_book,
            "malformed_verses_count": len(self.malformed_verses),
            "malformed_verses": self.malformed_verses[:10]
            if self.malformed_verses
            else [],
        }


class BibleProcessor:
    def __init__(self, include_apocrypha: bool = False, fix_line_breaks: bool = False):
        self.include_apocrypha = include_apocrypha
        self.fix_line_breaks = fix_line_breaks
        self.stats = BibleStatistics()
        self.current_book: Optional[str] = None
        self.current_chapter: Optional[int] = None
        self.parsed_bible: Dict[str, Dict[int, Dict[int, str]]] = {}

    def setup_logging(self, verbose: bool) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def remove_gutenberg_wrappers(self, text: str) -> str:
        """Remove Gutenberg header and footer from the text."""
        # Look for multiple possible start markers
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG EBOOK",
            "*END*THE SMALL PRINT",
        ]

        start_index = -1
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                if start_index == -1 or pos < start_index:
                    start_index = pos

        if start_index != -1:
            # Find the next line break after the start marker
            next_line_break = text.find("\n", start_index)
            if next_line_break != -1:
                text = text[next_line_break + 1 :]
                logging.debug("Gutenberg start marker found and removed.")
            else:
                text = text[start_index:]
                logging.debug(
                    "Gutenberg start marker found but no line break after it."
                )
        else:
            logging.warning("No Gutenberg start marker found.")

        # Look for multiple possible end markers
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "***END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg",
        ]

        end_index = -1
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                if end_index == -1 or pos < end_index:
                    end_index = pos

        if end_index != -1:
            text = text[:end_index]
            logging.debug("Gutenberg end marker found and removed.")
        else:
            logging.warning("No Gutenberg end marker found.")

        return text.strip()

    def normalize_book_name(self, name: str) -> Optional[str]:
        """Convert various book name formats to standard names."""
        name = name.strip()

        # Check for direct match
        if name in BIBLE_BOOKS:
            return name

        # Check aliases
        if name in BOOK_ALIASES:
            return BOOK_ALIASES[name]

        # Numbered books handling
        numbered_book_pattern = re.compile(r"^(\d+)\s*(\w+)$")
        match = numbered_book_pattern.match(name)
        if match:
            number, book = match.groups()
            number_word = {
                "1": "1",
                "2": "2",
                "3": "3",
                "I": "1",
                "II": "2",
                "III": "3",
                "First": "1",
                "Second": "2",
                "Third": "3",
            }.get(number)

            if number_word and f"{number_word} {book}" in BIBLE_BOOKS:
                return f"{number_word} {book}"

        # Handle special cases and common variations
        lower_name = name.lower()

        # Special case for Samuel, Kings, Chronicles, etc.
        if lower_name in [
            "samuel",
            "kings",
            "chronicles",
            "corinthians",
            "thessalonians",
            "timothy",
            "peter",
            "john",
        ]:
            logging.warning(f"Ambiguous book name '{name}' - defaulting to first book")
            return f"1 {name}"

        # Fuzzy matching as last resort
        for book in BIBLE_BOOKS:
            if book.lower().replace(" ", "") in lower_name.replace(" ", ""):
                logging.debug(f"Fuzzy matched '{name}' to standard book name '{book}'")
                return book

        # No match found
        logging.warning(f"Could not normalize book name: '{name}'")
        return None

    def identify_verse_reference(self, line: str) -> Optional[Tuple[int, int, str]]:
        """
        Extract chapter, verse and text from a line.
        Returns (chapter_num, verse_num, verse_text) or None if not a verse.
        """
        # Common patterns for verse references
        patterns = [
            # Standard chapter:verse format (e.g., "1:1 In the beginning...")
            r"^(\d+):(\d+)\s+(.+)$",
            # Just verse number at start of line (when chapter is established)
            r"^(\d+)\s+(.+)$",
            # KJV sometimes has verse numbers in brackets like [1]
            r"^\[(\d+)\]\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                groups = match.groups()
                if len(groups) == 3:  # chapter:verse text
                    try:
                        chapter = int(groups[0])
                        verse = int(groups[1])
                        return chapter, verse, groups[2]
                    except ValueError:
                        self.stats.add_malformed_verse(
                            line, "Invalid chapter/verse numbers"
                        )
                        return None
                elif len(groups) == 2:  # verse text (using current chapter)
                    if self.current_chapter is not None:
                        try:
                            verse = int(groups[0])
                            return self.current_chapter, verse, groups[1]
                        except ValueError:
                            self.stats.add_malformed_verse(line, "Invalid verse number")
                            return None
                    else:
                        self.stats.add_malformed_verse(
                            line, "No current chapter established"
                        )
                        return None

        return None

    def identify_book_and_chapter_headers(
        self, line: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Identify book titles and chapter headers.
        Returns (book_name, chapter_number) with None for values not found.
        """
        # Check for our own BOOK: marker from previous processing
        book_marker_match = re.match(r"^BOOK:\s*(.+)$", line.strip())
        if book_marker_match:
            book_name = self.normalize_book_name(book_marker_match.group(1))
            if book_name:
                return book_name, None

        # Check for standard book titles
        book_title_patterns = [
            # First simple pattern for "Book of X" or just "X" where X is a book name
            r"^(?:The\s+)?(?:Book\s+of\s+)?([A-Za-z\s1-3]+)$",
            # Pattern for "X Chapter Y" format
            r"^([A-Za-z\s1-3]+)\s+Chapter\s+(\d+)$",
            # Common KJV format sometimes has this pattern
            r"^THE\s+([A-Za-z\s1-3]+)$",
        ]

        for pattern in book_title_patterns:
            match = re.match(pattern, line.strip())
            if match:
                potential_book = match.group(1).strip()
                normalized_book = self.normalize_book_name(potential_book)
                if normalized_book:
                    # If this pattern also captured a chapter number
                    if len(match.groups()) > 1 and match.group(2):
                        try:
                            chapter = int(match.group(2))
                            return normalized_book, chapter
                        except ValueError:
                            return normalized_book, None
                    return normalized_book, None

        # Check for chapter headers
        chapter_patterns = [
            r"^Chapter\s+(\d+)$",
            r"^CHAPTER\s+(\d+)$",
            r"^\s*(\d+)\s*$",  # Sometimes just the number
        ]

        for pattern in chapter_patterns:
            match = re.match(pattern, line.strip())
            if match:
                try:
                    chapter = int(match.group(1))
                    return None, chapter
                except ValueError:
                    return None, None

        return None, None

    def standardize_book_titles(self, text: str) -> str:
        """
        Replace various book title formats with a consistent marker:
        BOOK: [Standardized Book Name]
        """
        lines = text.splitlines()
        result_lines = []
        processed_line_count = 0

        for line in lines:
            original_line = line
            line = line.strip()

            # Skip empty lines
            if not line:
                result_lines.append("")
                continue

            # Try to identify book or chapter headers
            book_name, chapter_num = self.identify_book_and_chapter_headers(line)

            if book_name:
                # Only include the book if it's in our allowed list
                if book_name in BIBLE_BOOKS and (
                    self.include_apocrypha
                    or BIBLE_BOOKS[book_name]["testament"] != "apocrypha"
                ):
                    result_lines.append(f"BOOK: {book_name}")
                    processed_line_count += 1
                    logging.debug(
                        f"Standardized book title: {original_line} -> {book_name}"
                    )
                else:
                    logging.debug(f"Skipping apocryphal or unknown book: {book_name}")

                # If we also found a chapter, add it on the next line
                if chapter_num:
                    result_lines.append(f"CHAPTER: {chapter_num}")
                    processed_line_count += 1
            elif chapter_num:
                result_lines.append(f"CHAPTER: {chapter_num}")
                processed_line_count += 1
                logging.debug(
                    f"Standardized chapter header: {original_line} -> Chapter {chapter_num}"
                )
            else:
                # Keep the original line
                result_lines.append(original_line)

        logging.info(
            f"Standardized {processed_line_count} book titles and chapter headers"
        )
        return "\n".join(result_lines)

    def process_verse_lines(self, text: str) -> str:
        """
        Process verse references to ensure consistent formatting.
        Also handles proper verse line breaks.
        """
        lines = text.splitlines()
        result_lines = []
        total_lines = len(lines)

        # We'll track current state as we process lines
        for i, line in enumerate(lines):
            if i % max(1, total_lines // 20) == 0:
                logging.info(f"Processing verses: {i/total_lines*100:.1f}% complete")
            line = line.strip()
            if not line:
                result_lines.append("")
                continue

            # Check for book markers first
            if line.startswith("BOOK:"):
                book_name = line[5:].strip()
                self.current_book = book_name
                self.current_chapter = None
                self.stats.add_book(book_name)

                # Initialize the book in our parsed structure if needed
                if book_name not in self.parsed_bible:
                    self.parsed_bible[book_name] = {}

                result_lines.append(line)
                continue

            # Check for chapter markers next
            if line.startswith("CHAPTER:"):
                try:
                    chapter_num = int(line[8:].strip())
                    self.current_chapter = chapter_num

                    if self.current_book:
                        self.stats.add_chapter(self.current_book, chapter_num)

                        # Initialize the chapter in our parsed structure
                        if self.current_book not in self.parsed_bible:
                            self.parsed_bible[self.current_book] = {}
                        self.parsed_bible[self.current_book][chapter_num] = {}

                    result_lines.append(line)
                    continue
                except ValueError:
                    logging.warning(f"Invalid chapter number in line: {line}")

            # Try to identify verse references
            verse_info = self.identify_verse_reference(line)
            if verse_info:
                chapter_num, verse_num, verse_text = verse_info

                # Update current chapter if needed
                if self.current_chapter != chapter_num:
                    self.current_chapter = chapter_num

                    # Add a chapter marker if we detect a new chapter
                    if self.current_book:
                        result_lines.append(f"CHAPTER: {chapter_num}")
                        self.stats.add_chapter(self.current_book, chapter_num)

                        # Initialize the chapter in our parsed structure
                        if self.current_book not in self.parsed_bible:
                            self.parsed_bible[self.current_book] = {}
                        if chapter_num not in self.parsed_bible[self.current_book]:
                            self.parsed_bible[self.current_book][chapter_num] = {}

                # Format the verse consistently and add it
                formatted_verse = f"{chapter_num}:{verse_num} {verse_text}"
                result_lines.append(formatted_verse)

                # Update our statistics
                if self.current_book:
                    self.stats.add_verse(self.current_book, chapter_num, verse_num)

                    # Add to our parsed structure
                    if self.current_book in self.parsed_bible:
                        if chapter_num not in self.parsed_bible[self.current_book]:
                            self.parsed_bible[self.current_book][chapter_num] = {}
                        self.parsed_bible[self.current_book][chapter_num][
                            verse_num
                        ] = verse_text

                continue

            # Handle special cases where verses might be split across lines
            if (
                self.fix_line_breaks
                and i > 0
                and not line.startswith("BOOK:")
                and not line.startswith("CHAPTER:")
            ):
                prev_line = lines[i - 1].strip()
                # If previous line was a verse and this doesn't look like a verse reference
                if (
                    prev_line
                    and not re.match(r"^\d+:\d+", line)
                    and (
                        re.match(r"^\d+:\d+", prev_line)
                        or result_lines
                        and re.match(r"^\d+:\d+", result_lines[-1])
                    )
                ):
                    # Append to previous line instead of adding new line
                    result_lines[-1] += " " + line
                    logging.debug(f"Joined broken verse line: {line}")
                    continue

            # If we get here, just keep the original line
            result_lines.append(line)

        # Count the total processed verses
        verse_count = sum(
            1 for line in result_lines if re.match(r"^\d+:\d+", line.strip())
        )
        logging.info(f"Processed {verse_count} verses")

        return "\n".join(result_lines)

    def validate_bible_structure(self) -> Dict[str, Any]:
        """
        Validate the parsed Bible structure against known patterns.
        Returns a validation report.
        """
        expected_book_count = len(
            [
                b
                for b in BIBLE_BOOKS
                if self.include_apocrypha or BIBLE_BOOKS[b]["testament"] != "apocrypha"
            ]
        )

        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": self.stats.generate_report(),
        }

        # Book count check
        books_found = len(self.stats.books_found)
        if books_found < expected_book_count:
            missing = [
                b
                for b in BIBLE_BOOKS
                if b not in self.stats.books_found
                and (
                    self.include_apocrypha or BIBLE_BOOKS[b]["testament"] != "apocrypha"
                )
            ]
            report["warnings"].append(
                f"Found {books_found} books out of {expected_book_count} expected"
            )
            report["warnings"].append(f"Missing books: {', '.join(missing[:5])}")
            if len(missing) > 5:
                report["warnings"].append(f"... and {len(missing) - 5} more")

        # Verse count check (KJV typically has around 31,000 verses)
        if self.stats.total_verses < 30000:
            report["warnings"].append(
                f"Low verse count: {self.stats.total_verses} (expected ~31,000)"
            )
            report["valid"] = False

        # Check for malformed verses
        if self.stats.malformed_verses:
            report["warnings"].append(
                f"Found {len(self.stats.malformed_verses)} malformed verses"
            )

        return report

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "text",
        validate: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process all .txt files in a directory.
        Returns a list of processing reports.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        reports = []
        txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

        logging.info(f"Found {len(txt_files)} .txt files to process")

        for txt_file in txt_files:
            input_path = os.path.join(input_dir, txt_file)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(txt_file)[0]}.{output_format}"
            )

            logging.info(f"Processing file: {input_path}")
            report = self.process_bible_text(
                input_path, output_path, output_format, validate
            )
            reports.append(report)

        return reports

    def process_bible_text(
        self,
        input_file: str,
        output_file: str,
        output_format: str = "text",
        validate: bool = False,
    ) -> Dict[str, Any]:
        """
        Main method to process the Bible text from input to output.
        Returns a processing report.
        """
        logging.info(f"Reading input file: {input_file}")

        # Process the file in chunks instead of loading it all at once
        try:
            # First determine file size for progress reporting
            file_size = os.path.getsize(input_file)
            logging.info(f"File size: {file_size/1024/1024:.2f} MB")

            # Process in chunks of 1MB
            chunk_size = 1024 * 1024
            processed_text = ""

            with open(input_file, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    processed_text += chunk
                    logging.debug(
                        f"Processed {len(processed_text)/file_size*100:.1f}% of file"
                    )
        except UnicodeDecodeError:
            logging.warning("UTF-8 decoding failed, trying latin-1 encoding")
        with open(input_file, "r", encoding="latin-1") as f:
            processed_text = f.read()

        # Process the text in stages
        processed_text = self.remove_gutenberg_wrappers(processed_text)
        processed_text = self.standardize_book_titles(processed_text)
        processed_text = self.process_verse_lines(processed_text)

        # Write the result in the specified format
        logging.info(f"Writing output to {output_file} in {output_format} format")
        self.write_output(processed_text, output_file, output_format)

        # Validate if requested
        validation_report = None
        if validate:
            validation_report = self.validate_bible_structure()
            log_level = (
                logging.WARNING if validation_report["warnings"] else logging.INFO
            )
            logging.log(
                log_level,
                f"Validation complete: {len(validation_report['warnings'])} warnings",
            )

        # Return processing report
        return {
            "input_file": input_file,
            "output_file": output_file,
            "format": output_format,
            "verses_processed": self.stats.total_verses,
            "books_processed": len(self.stats.books_found),
            "validation": validation_report,
        }

    def format_parsed_bible_as_text(self) -> str:
        """Format the internally parsed Bible structure as text."""
        lines = []

        for book in sorted(self.parsed_bible.keys()):
            lines.append(f"BOOK: {book}")
            lines.append("")

            for chapter in sorted(self.parsed_bible[book].keys()):
                lines.append(f"CHAPTER: {chapter}")
                lines.append("")

                for verse in sorted(self.parsed_bible[book][chapter].keys()):
                    verse_text = self.parsed_bible[book][chapter][verse]
                    lines.append(f"{chapter}:{verse} {verse_text}")

                lines.append("")  # Add blank line after each chapter

            lines.append("")  # Add blank line after each book

        return "\n".join(lines)

    def write_output(
        self, text: str, output_file: str, output_format: str = "text"
    ) -> None:
        """Write the processed Bible to the output file in the specified format."""
        try:
            if output_format == "text":
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)
            elif output_format == "json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(self.parsed_bible, f, indent=2)
            elif output_format == "csv":
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Book", "Chapter", "Verse", "Text"])

                    for book in sorted(self.parsed_bible.keys()):
                        for chapter in sorted(self.parsed_bible[book].keys()):
                            for verse in sorted(
                                self.parsed_bible[book][chapter].keys()
                            ):
                                writer.writerow(
                                    [
                                        book,
                                        chapter,
                                        verse,
                                        self.parsed_bible[book][chapter][verse],
                                    ]
                                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logging.info(f"Successfully wrote output to {output_file}")
        except IOError as e:
            logging.error(f"Failed to write to output file {output_file}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error writing to {output_file}: {e}")
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process KJV Bible text into standardized format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input KJV Bible text file"
    )
    parser.add_argument("--output", required=True, help="Path to the output file")
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--include-apocrypha",
        action="store_true",
        help="Include apocryphal books if present",
    )
    parser.add_argument(
        "--fix-line-breaks",
        action="store_true",
        help="Attempt to fix irregular line breaks",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the structure of the output"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--validation-report", help="Path to save detailed validation report as JSON"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all .txt files in input directory"
    )

    args = parser.parse_args()

    # Initialize the processor
    processor = BibleProcessor(
        include_apocrypha=args.include_apocrypha, fix_line_breaks=args.fix_line_breaks
    )

    # Setup logging
    processor.setup_logging(args.verbose)

    # Batch processing or single file
    if args.batch:
        if not os.path.isdir(args.input):
            parser.error("--batch requires input to be a directory")
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

        reports = processor.process_directory(
            args.input, args.output, args.format, args.validate
        )
        print(f"Processed {len(reports)} files")

        for report in reports:
            print(f"\nFile: {os.path.basename(report['input_file'])}")
            print(f"- Books: {report['books_processed']}")
            print(f"- Verses: {report['verses_processed']}")
    else:
        # Process the Bible text
        report = processor.process_bible_text(
            args.input, args.output, args.format, args.validate
        )

        # Print summary report
        print("\nProcessing Summary:")
        print(f"- Input file: {report['input_file']}")
        print(f"- Output file: {report['output_file']} (format: {report['format']})")
        print(f"- Books processed: {report['books_processed']}")
        print(f"- Verses processed: {report['verses_processed']}")

        if args.validate and report["validation"]:
            print("\nValidation Report:")
            print(
                f"- Valid structure: {'Yes' if report['validation']['valid'] else 'No'}"
            )

            if report["validation"]["warnings"]:
                print(f"- Warnings: {len(report['validation']['warnings'])}")
                for warning in report["validation"]["warnings"][:3]:
                    print(f"  * {warning}")
                if len(report["validation"]["warnings"]) > 3:
                    print(
                        f"  * ... and {len(report['validation']['warnings']) - 3} more warnings"
                    )
            else:
                print("- No warnings")

        # Save detailed validation report if requested
        if args.validation_report and report["validation"]:
            try:
                with open(args.validation_report, "w", encoding="utf-8") as f:
                    json.dump(report["validation"], f, indent=2)
                print(f"Detailed validation report saved to {args.validation_report}")
            except IOError as e:
                print(f"Error saving validation report: {e}")
